"""Build dense (FAISS) + sparse (BM25) indexes over the Swiss legal corpus.

Run once after downloading data:
    python Law_Retrieval/index_builder.py

Outputs under Law_Retrieval/data/index/:
    faiss.bin          — FAISS IndexFlatIP over L2-normalized embeddings
    bm25.pkl           — pickled (BM25Okapi, tokenized_corpus)
    citations.parquet  — chunk_id -> (citation, source, text)
    meta.json          — embedder name, chunk config, counts
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
INDEX_DIR = DATA_DIR / "index"

# Default to the smaller multilingual model for fast CPU builds.
# Switch to "BAAI/bge-m3" via --embedder once you have a GPU or time for a full rebuild.
DEFAULT_EMBEDDER = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_TOKENS = 512
CHUNK_OVERLAP = 64
BATCH_SIZE = 64


def _load_corpus(use_court: bool = True) -> pd.DataFrame:
    frames = [_load_laws()]
    if use_court:
        court = _load_court()
        if court is not None:
            frames.append(court)
    df = pd.concat(frames, ignore_index=True)
    return _clean(df)


def _load_laws() -> pd.DataFrame:
    laws_path = DATA_DIR / "laws_de.csv"
    if not laws_path.exists():
        raise FileNotFoundError(
            f"Missing {laws_path}. Download data per Law_Retrieval/README.md."
        )
    laws = pd.read_csv(laws_path)
    return _normalize_columns(laws, source="laws")


def _load_court() -> pd.DataFrame | None:
    court_path = DATA_DIR / "court_considerations.csv"
    if not court_path.exists():
        print(f"[warn] {court_path} not found.", file=sys.stderr)
        return None
    court = pd.read_csv(court_path)
    return _normalize_columns(court, source="court")


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["citation", "text"])
    df["citation"] = df["citation"].astype(str).str.strip()
    df["text"] = df["text"].astype(str)
    df = df[df["text"].str.len() > 0]
    return df.reset_index(drop=True)


def _normalize_columns(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Map whatever citation/text columns exist into a common schema."""
    cols = {c.lower(): c for c in df.columns}
    citation_col = next(
        (cols[k] for k in ("citation", "canonical_citation", "id", "key") if k in cols),
        None,
    )
    text_col = next(
        (cols[k] for k in ("text", "content", "snippet", "body") if k in cols),
        None,
    )
    if citation_col is None or text_col is None:
        raise ValueError(
            f"Could not infer citation/text columns in {source} corpus. Columns: {list(df.columns)}"
        )
    out = pd.DataFrame({
        "citation": df[citation_col],
        "text": df[text_col],
        "source": source,
    })
    return out


def _chunk_text(text: str, approx_tokens: int = CHUNK_TOKENS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Word-based chunking (≈1 token per word for German/French/Italian legal text is close enough)."""
    words = text.split()
    if len(words) <= approx_tokens:
        return [text]
    chunks = []
    step = approx_tokens - overlap
    for start in range(0, len(words), step):
        piece = words[start:start + approx_tokens]
        if not piece:
            break
        chunks.append(" ".join(piece))
        if start + approx_tokens >= len(words):
            break
    return chunks


def _build_chunks(df: pd.DataFrame) -> pd.DataFrame:
    citations: list[str] = []
    sources: list[str] = []
    texts: list[str] = []
    citation_vals = df["citation"].to_numpy()
    source_vals = df["source"].to_numpy()
    text_vals = df["text"].to_numpy()
    for i in tqdm(range(len(df)), desc="chunking"):
        for chunk in _chunk_text(text_vals[i]):
            citations.append(citation_vals[i])
            sources.append(source_vals[i])
            texts.append(chunk)
    return pd.DataFrame({"citation": citations, "source": sources, "text": texts})


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def build_indexes(use_court: bool = True, embedder_name: str = DEFAULT_EMBEDDER) -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading corpus...")
    corpus = _load_corpus(use_court=use_court)
    print(f"  {len(corpus):,} raw records")

    print("Chunking...")
    chunks = _build_chunks(corpus)
    print(f"  {len(chunks):,} chunks")

    print("Embedding (this is the slow step)...")
    embeddings = _embed_texts(chunks["text"].tolist(), embedder_name)

    print("Building FAISS index...")
    import faiss
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, str(INDEX_DIR / "faiss.bin"))
    print(f"  FAISS: {index.ntotal:,} vectors, dim={dim}")

    print("Building BM25 index...")
    from rank_bm25 import BM25Okapi
    tokenized = [_tokenize(t) for t in tqdm(chunks["text"].tolist(), desc="tokenize")]
    bm25 = BM25Okapi(tokenized)
    with open(INDEX_DIR / "bm25.pkl", "wb") as f:
        pickle.dump({"bm25": bm25, "tokenized": tokenized}, f)

    print("Writing citations parquet...")
    chunks.reset_index(drop=True).to_parquet(INDEX_DIR / "citations.parquet", index=False)

    meta = {
        "embedder": embedder_name,
        "chunk_tokens": CHUNK_TOKENS,
        "chunk_overlap": CHUNK_OVERLAP,
        "num_chunks": int(len(chunks)),
        "num_unique_citations": int(chunks["citation"].nunique()),
        "dim": int(dim),
        "use_court": use_court,
    }
    (INDEX_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Done. Indexes in {INDEX_DIR}")


def build_court_tfidf(max_words_per_doc: int = 800, max_features: int = 200_000) -> None:
    """Sparse TF-IDF index over court_considerations.csv.

    Uses scikit-learn's TfidfVectorizer (vectorized C/numpy) which is roughly
    50-100x faster than rank-bm25 at this scale. Each row in the CSV is treated
    as one document (citations are already at consideration granularity, e.g.
    "BGE 137 IV 122 E. 6.2"). Text is truncated per row to keep the matrix
    sparse and the pickle small.
    """
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading court corpus...")
    court = _load_court()
    if court is None:
        raise FileNotFoundError("court_considerations.csv missing")
    court = _clean(court)
    print(f"  {len(court):,} raw court records")

    print(f"Truncating to first {max_words_per_doc} words per row...")
    court["text"] = court["text"].map(lambda t: " ".join(t.split()[:max_words_per_doc]))
    chunks = court[["citation", "source", "text"]].reset_index(drop=True)

    print("Fitting TF-IDF (sparse, vectorized)...")
    from sklearn.feature_extraction.text import TfidfVectorizer
    import scipy.sparse as sp
    vectorizer = TfidfVectorizer(
        lowercase=True,
        max_features=max_features,
        ngram_range=(1, 1),
        sublinear_tf=True,
        dtype=np.float32,
    )
    matrix = vectorizer.fit_transform(chunks["text"].tolist())
    print(f"  TF-IDF matrix: {matrix.shape}, nnz={matrix.nnz:,}")

    sp.save_npz(INDEX_DIR / "court_tfidf.npz", matrix)
    with open(INDEX_DIR / "court_tfidf_vec.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    chunks.to_parquet(INDEX_DIR / "court_citations.parquet", index=False)

    meta = {
        "num_chunks": int(len(chunks)),
        "num_unique_citations": int(chunks["citation"].nunique()),
        "vocab_size": int(len(vectorizer.vocabulary_)),
        "kind": "tfidf_court",
    }
    (INDEX_DIR / "court_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Done. Court TF-IDF index in {INDEX_DIR}")


def _embed_texts(texts: list[str], embedder_name: str) -> np.ndarray:
    """Embed with BGE-M3 (or any sentence-transformers model), L2-normalized."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(embedder_name)
    emb = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return emb.astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--no-court", action="store_true", help="Skip the court_considerations corpus.")
    p.add_argument("--court-bm25-only", action="store_true",
                   help="Build ONLY a sparse TF-IDF index over court_considerations.csv "
                        "(no embedding). Use when you already have the laws_de dense+BM25 index.")
    p.add_argument("--embedder", default=DEFAULT_EMBEDDER)
    args = p.parse_args()
    if args.court_bm25_only:
        build_court_tfidf()
    else:
        build_indexes(use_court=not args.no_court, embedder_name=args.embedder)


if __name__ == "__main__":
    main()
