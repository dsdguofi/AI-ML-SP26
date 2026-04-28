"""Swiss legal citation retrieval pipeline.

English legal question → LLM translation (DE/FR/IT) → BM25 retrieval →
candidate pool (query-extracted + topic fallback + BM25-mined) →
LLM reranker → grounded citations in abbreviation form.

Library use:
    from Law_evaluation import LegalRetriever
    r = LegalRetriever()
    r.load()
    citations, _ = r.predict("Can a buyer rescind a sale for hidden defects?")

CLI use:
    python Law_evaluation.py --split val --out val_submission.csv
    python Law_evaluation.py --split test --out submission.csv
    python Law_evaluation.py --no-llm   # deterministic fallback only
"""

from __future__ import annotations

import argparse
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from citation_norm import extract_citations, to_abbrev_form
from ollama_client import OllamaClient, OllamaError
from prompts import EXPAND_LEGAL_TERMS, RERANK_CITATIONS, TRANSLATE_QUERY

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
INDEX_DIR = DATA_DIR / "index"

BM25_TOP_K = 200          # chunks retrieved per query variant
CANDIDATES_FOR_RERANK = 30  # how many candidates the LLM reranker sees
DEFAULT_FINAL_N = 30      # citations to predict (gold median ≈ 25)
RERANK_SNIPPET_CHARS = 100  # chars of chunk text shown to reranker per candidate

# Federal-court procedural articles in nearly every appellate gold set.
# Art. 100 Abs. 1 BGG is in 9/10 val gold sets.
ALWAYS_PREDICT = [
    "Art. 100 Abs. 1 BGG",
    "Art. 95 BGG",
    "Art. 105 Abs. 1 BGG",
    "Art. 42 Abs. 2 BGG",
    "Art. 82 BGG",
    "Art. 106 Abs. 2 BGG",
    "Art. 113 BGG",
    "Art. 93 Abs. 1 BGG",
]

# Topic keyword → canonical articles. Each entry: (required_hits, keywords, articles).
# required_hits = minimum number of distinct keywords that must match before the
# fallback fires. This prevents a single incidental word (e.g. "medical") from
# triggering the wrong domain and polluting predictions.
TOPIC_FALLBACKS: dict[str, tuple[int, tuple[str, ...], tuple[str, ...]]] = {
    "criminal_detention": (
        2,
        ("detention", "remand", "pre-trial", "pretrial", "collusion",
         "prosecutor", "accused", "theft", "assault", "offence",
         "offense", "suspect", "interrogation"),
        ("Art. 221 Abs. 1 StPO", "Art. 221 Abs. 2 StPO", "Art. 222 StPO",
         "Art. 227 Abs. 1 StPO", "Art. 212 Abs. 3 StPO", "Art. 393 Abs. 1 StPO",
         "Art. 396 Abs. 1 StPO", "Art. 382 Abs. 1 StPO", "Art. 385 Abs. 1 StPO",
         "Art. 140 Abs. 1 StGB", "Art. 123 StGB", "Art. 144 StGB",
         "Art. 29 Abs. 2 BV"),
    ),
    "insurance_disability": (
        # Require 3 hits — "medical", "insurance", "capacity" are too common in
        # unrelated contexts (e.g. property disputes involving impaired persons).
        3,
        ("invalidity", "disability pension", "insured person", "social insurance",
         "vocational rehabilitation", "earning capacity", "incapacity for work",
         "iv-degree", "lai", "ivg", "atsg", "disability insurance"),
        ("Art. 8 Abs. 1 ATSG", "Art. 8 Abs. 1 IVG", "Art. 17 Abs. 1 IVG",
         "Art. 1 Abs. 1 IVG", "Art. 56 Abs. 1 ATSG", "Art. 60 Abs. 1 ATSG",
         "Art. 61 ATSG", "Art. 28 Abs. 1 IVG", "Art. 4 Abs. 1 IVG",
         "Art. 16 ATSG", "Art. 6 ATSG"),
    ),
    "inheritance": (
        2,
        ("inheritance", "heir", "heirship", "testament", "will", "succession",
         "estate", "bequest", "legacy", "deceased", "testator", "intestate"),
        ("Art. 467 ZGB", "Art. 469 Abs. 1 ZGB", "Art. 471 ZGB", "Art. 505 Abs. 1 ZGB",
         "Art. 458 Abs. 3 ZGB", "Art. 520a ZGB", "Art. 527 ZGB", "Art. 8 ZGB",
         "Art. 20 Abs. 2 OR"),
    ),
    "family_marriage": (
        2,
        ("marriage", "divorce", "spouse", "matrimonial", "child support",
         "custody of child", "alimony", "maintenance"),
        ("Art. 133 Abs. 1 ZGB", "Art. 133 Abs. 2 ZGB", "Art. 285 Abs. 1 ZGB",
         "Art. 274 Abs. 2 ZGB", "Art. 8 ZGB"),
    ),
    "contract": (
        2,
        ("contract", "contractor", "agreement", "offer", "acceptance",
         "performance", "breach", "damages", "supplier", "warranty",
         "rescission", "rescind", "defect"),
        ("Art. 1 Abs. 1 OR", "Art. 18 Abs. 1 OR", "Art. 41 OR", "Art. 97 Abs. 1 OR",
         "Art. 363 OR", "Art. 364 Abs. 1 OR", "Art. 248 Abs. 1 OR",
         "Art. 20 Abs. 2 OR"),
    ),
    "civil_procedure": (
        2,
        ("plaintiff", "defendant", "civil court", "cantonal court", "appeal",
         "summary procedure", "evidence", "burden of proof"),
        ("Art. 8 ZGB", "Art. 28 ZGB"),
    ),
}


@dataclass
class LegalRetriever:
    model: str = "llama3.2:3b"           # used for reranking
    translate_model: str = "llama3.2:3b" # used for translation/expansion
    final_n: int = DEFAULT_FINAL_N
    use_llm: bool = True          # master switch; False = pure deterministic mode

    # populated by .load()
    _ollama: OllamaClient | None = field(default=None, init=False)       # reranker
    _translate_ollama: OllamaClient | None = field(default=None, init=False)  # translator
    _bm25: object = field(default=None, init=False)
    _tokenized: list[list[str]] = field(default_factory=list, init=False)
    _chunks: pd.DataFrame = field(default=None, init=False)
    _valid_citations: set[str] = field(default_factory=set, init=False)

    def load(self) -> None:
        if not INDEX_DIR.exists():
            raise FileNotFoundError(
                f"Index not built. Run: python {PROJECT_ROOT / 'index_builder.py'}"
            )

        self._chunks = pd.read_parquet(INDEX_DIR / "citations.parquet")
        # valid-citation set in abbreviation form — that is what gold uses.
        self._valid_citations = {
            to_abbrev_form(c) for c in self._chunks["citation"].unique()
        }

        with open(INDEX_DIR / "bm25.pkl", "rb") as f:
            bm = pickle.load(f)
        self._bm25 = bm["bm25"]
        self._tokenized = bm["tokenized"]
        print(f"[load] BM25: {len(self._chunks):,} chunks", file=sys.stderr)

        if self.use_llm:
            # Reranker (8b — accuracy)
            self._ollama = OllamaClient(model=self.model, timeout=180)
            try:
                self._ollama.health_check()
                print(f"[load] Ollama reranker OK ({self.model})", file=sys.stderr)
            except OllamaError as e:
                print(f"[warn] Ollama unavailable — falling back to deterministic: {e}",
                      file=sys.stderr)
                self._ollama = None

            if self._ollama is not None:
                self._translate_ollama = OllamaClient(
                    model=self.translate_model, timeout=180
                )
                print(f"[load] Ollama translator OK ({self.translate_model})", file=sys.stderr)

    # ---------- public API ----------

    def predict(self, query: str) -> list[str]:
        """Return a ranked list of Swiss legal citations for the English query.

        Pipeline:
          1. LLM translation → DE/FR/IT query variants (when Ollama available).
          2. LLM legal-term expansion → extra BM25 query variant.
          3. BM25 retrieval over all variants; mine inline citations from chunks.
          4. Deterministic safety net: query-extracted + topic fallback +
             always-predict BGG floor.
          5. LLM reranker picks the best final_n from the merged candidate pool.
             Falls back to deterministic ordering if Ollama fails.
        """
        self._require_loaded()

        # --- deterministic stages (always run) ---
        from_query = [
            to_abbrev_form(c) for c in extract_citations(query)
            if to_abbrev_form(c) in self._valid_citations
        ]
        from_topic = self._topic_fallback(query)
        always = [c for c in ALWAYS_PREDICT if c in self._valid_citations]

        # --- LLM stages (skipped if Ollama is unavailable) ---
        variants = self._build_query_variants(query)   # EN + DE/FR/IT + terms
        retrieved = self._retrieve_citations(variants)

        # Merge: query citations first (precision), then topic priors, then
        # BM25-mined signal, then BGG floor. This is the candidate pool for
        # the reranker.
        candidates = _dedup_keep_order(
            from_query + from_topic + retrieved + always
        )

        # LLM reranker: pick and re-order the best final_n from the pool.
        reranked = self._rerank(query, candidates)
        return reranked

    def run_on_split(self, split: str) -> pd.DataFrame:
        """Generate a submission dataframe for train/val/test."""
        self._require_loaded()
        split_path = DATA_DIR / f"{split}.csv"
        if not split_path.exists():
            raise FileNotFoundError(split_path)
        df = pd.read_csv(split_path)
        query_col = _find_query_column(df)
        rows = []
        for rec in tqdm(df.itertuples(index=False), total=len(df), desc=f"predict[{split}]"):
            qid = getattr(rec, "query_id")
            q = getattr(rec, query_col)
            try:
                citations = self.predict(q)
            except OllamaError as e:
                print(f"[warn] {qid}: {e}", file=sys.stderr)
                citations = []
            rows.append({
                "query_id": qid,
                "predicted_citations": ";".join(citations),
            })
        return pd.DataFrame(rows)

    # ---------- pipeline stages ----------

    def _build_query_variants(self, query: str) -> list[str]:
        """Return [original_EN, DE, FR, IT, expanded_terms] where available.

        Translation uses the fast 3b model; expansion is skipped when 3b
        is also slow (OllamaError caught and warned).
        """
        variants = [query]
        tc = self._translate_ollama  # fast 3b client
        if tc is None:
            return variants

        # Truncate query for LLM calls — first 250 chars captures the legal topic
        # while leaving enough token budget for the model to produce full output.
        short_q = query[:250]

        # Translation (3b) — 600 tokens covers three legal sentences with room to close JSON.
        try:
            tr = tc.generate_json(
                TRANSLATE_QUERY.format(query=short_q), num_predict=600
            )
            for lang in ("de", "fr", "it"):
                v = tr.get(lang) if isinstance(tr, dict) else None
                if isinstance(v, str) and v.strip():
                    variants.append(v.strip())
        except OllamaError as e:
            print(f"[warn] translation failed: {e}", file=sys.stderr)

        # Legal-term expansion (3b) — num_predict=100 covers ~10 short terms.
        try:
            ex = tc.generate_json(
                EXPAND_LEGAL_TERMS.format(query=short_q), num_predict=100
            )
            terms = ex.get("terms", []) if isinstance(ex, dict) else []
            if isinstance(terms, list) and terms:
                variants.append(" ".join(str(t) for t in terms[:15]))
        except OllamaError as e:
            print(f"[warn] expansion failed: {e}", file=sys.stderr)

        return variants

    def _topic_fallback(self, query: str) -> list[str]:
        q = query.lower()
        out: list[str] = []
        for _topic, (required_hits, keywords, articles) in TOPIC_FALLBACKS.items():
            hits = sum(1 for kw in keywords if kw in q)
            if hits >= required_hits:
                for art in articles:
                    if art in self._valid_citations:
                        out.append(art)
        return _dedup_keep_order(out)

    def _retrieve_citations(self, variants: list[str]) -> list[str]:
        """BM25 over each variant; score citations that appear inline in chunks."""
        scores: dict[str, float] = {}
        for variant in variants:
            tok = variant.lower().split()
            if not tok:
                continue
            bm = self._bm25.get_scores(tok)
            top_idx = _topk_indices(bm, BM25_TOP_K)
            for rank, chunk_id in enumerate(top_idx.tolist()):
                row = self._chunks.iloc[int(chunk_id)]
                weight = 1.0 / (rank + 1)
                snippet = _truncate(str(row["text"]), RERANK_SNIPPET_CHARS)
                for cit in extract_citations(snippet):
                    cit = to_abbrev_form(cit)
                    if cit in self._valid_citations:
                        scores[cit] = scores.get(cit, 0.0) + weight
        return [c for c, _ in sorted(scores.items(), key=lambda kv: -kv[1])]

    def _rerank(self, query: str, candidates: list[str]) -> list[str]:
        """Ask Ollama to pick and rank the best citations from the candidate pool.

        Falls back to deterministic order (first final_n candidates) if Ollama
        is unavailable or returns nothing usable.
        """
        fallback = candidates[: self.final_n]

        if self._ollama is None or not candidates:
            return fallback

        # Build numbered listing with a short snippet per citation.
        pool = candidates[: CANDIDATES_FOR_RERANK]
        lookup = self._chunks.set_index("citation")

        lines: list[str] = []
        for i, cit in enumerate(pool, 1):
            # Look up by SR-numeric form (how corpus stores it) or abbrev.
            snippet = ""
            for key in (cit, _sr_key(cit)):
                if key in lookup.index:
                    row = lookup.loc[key]
                    text = row["text"] if isinstance(row, pd.Series) else row.iloc[0]["text"]
                    snippet = _truncate(str(text), RERANK_SNIPPET_CHARS)
                    break
            lines.append(f"{i}. [{cit}] {snippet}")

        prompt = RERANK_CITATIONS.format(
            query=query,
            candidates="\n".join(lines),
            top_n=self.final_n,
        )

        try:
            parsed = self._ollama.generate_json(prompt, num_predict=512)
            raw = parsed.get("ranked_citations", []) if isinstance(parsed, dict) else []
        except OllamaError as e:
            print(f"[warn] rerank failed, using deterministic order: {e}", file=sys.stderr)
            return fallback

        # Ground: only keep citations that exist in the corpus.
        pool_set = set(pool)
        grounded: list[str] = []
        seen: set[str] = set()
        for c in raw:
            if not isinstance(c, str):
                continue
            c = c.strip().strip("[]")
            if c in pool_set and c not in seen:
                grounded.append(c)
                seen.add(c)
            if len(grounded) >= self.final_n:
                break

        if not grounded:
            print("[warn] reranker returned nothing usable, using deterministic order",
                  file=sys.stderr)
            return fallback

        # Append any candidates the reranker skipped (preserves recall).
        extras = [c for c in pool if c not in seen]
        return (grounded + extras)[: self.final_n]

    # ---------- helpers ----------

    def _require_loaded(self) -> None:
        if self._bm25 is None:
            raise RuntimeError("Retriever not loaded. Call .load() first.")


def _sr_key(abbrev_citation: str) -> str:
    """Convert abbreviation-form citation back to SR-numeric for corpus lookup."""
    from citation_norm import ABBREV_TO_SR, _TAIL_RE
    m = _TAIL_RE.match(abbrev_citation)
    if not m:
        return abbrev_citation
    tail = m.group("tail")
    if tail in ABBREV_TO_SR:
        return f"{m.group('head').strip()} {ABBREV_TO_SR[tail]}"
    return abbrev_citation


def _find_query_column(df: pd.DataFrame) -> str:
    for candidate in ("query", "question", "question_en", "text"):
        if candidate in df.columns:
            return candidate
    non_id = [c for c in df.columns if c != "query_id" and c != "gold_citations"]
    if not non_id:
        raise ValueError(f"No query column found in columns: {list(df.columns)}")
    return non_id[0]


def _truncate(text: str, n: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _dedup_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for v in items:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    n = len(scores)
    k = min(k, n)
    if k <= 0:
        return np.empty(0, dtype=np.int64)
    part = np.argpartition(-scores, k - 1)[:k]
    return part[np.argsort(-scores[part])]


# ---------------- CLI ----------------

def _cli() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--split", choices=["train", "val", "test"],
                   help="Run batch prediction on a split.")
    p.add_argument("--out", type=Path, help="Output submission CSV path.")
    p.add_argument("--model", default="llama3.2:3b",
                   help="Ollama model for reranking.")
    p.add_argument("--translate-model", default="llama3.2:3b",
                   help="Ollama model for translation/expansion (speed).")
    p.add_argument("--final-n", type=int, default=DEFAULT_FINAL_N,
                   help="Max citations to predict per query.")
    p.add_argument("--no-llm", action="store_true",
                   help="Skip Ollama entirely — deterministic fallback only.")
    p.add_argument("--interactive", action="store_true",
                   help="REPL: paste a case, get citations.")
    args = p.parse_args()

    retriever = LegalRetriever(
        model=args.model,
        translate_model=args.translate_model,
        final_n=args.final_n,
        use_llm=not args.no_llm,
    )
    retriever.load()

    if args.interactive:
        _interactive_loop(retriever)
        return

    if not args.split:
        p.error("Provide either --split or --interactive.")
    out = args.out or PROJECT_ROOT / f"{args.split}_submission.csv"
    df = retriever.run_on_split(args.split)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} predictions to {out}")


def _interactive_loop(retriever: LegalRetriever) -> None:
    print("Interactive mode. Paste an English case description, end with a blank line.")
    print("Ctrl-C to exit.\n")
    try:
        while True:
            print("case> ", end="", flush=True)
            lines = []
            for line in sys.stdin:
                if line.strip() == "":
                    break
                lines.append(line)
            if not lines:
                break
            query = "".join(lines).strip()
            if not query:
                continue
            citations = retriever.predict(query)
            print("\n--- Citations ---")
            for c in citations:
                print(f"  {c}")
            print()
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    _cli()
