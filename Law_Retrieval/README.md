# Swiss Legal Citation Retrieval

Given an English legal question, retrieve the relevant Swiss legal citations from a German/French/Italian corpus. Scored on **Macro F1** — per-query precision/recall F1 averaged across all queries.

**Current val score: Macro F1 ≈ 0.25** (10 queries, exact-string set match)

---

## How It Works

The pipeline has five stages, run for each query:

1. **Citation extraction** — regex-scans the English query for any citations mentioned verbatim (e.g. `Art. 221 Abs. 1 StPO`). These are high-precision and go first in the candidate pool.

2. **Topic fallback** — keyword heuristics detect the legal domain (criminal detention, disability insurance, inheritance, contract, etc.) and inject canonical statute articles for that domain. Requires ≥2 keyword matches to avoid misfiring on incidental words.

3. **LLM translation** — `llama3.2:3b` (via Ollama) translates the English query to German, French, and Italian so BM25 can match against the multilingual corpus.

4. **BM25 retrieval** — sparse retrieval over 176k statute chunks, run once per query variant (EN + DE + FR + IT + expanded legal terms). Inline citations found in the retrieved chunks are harvested and scored by frequency across variants.

5. **LLM reranker** — `llama3.2:3b` picks the best 30 citations from the merged candidate pool, grounded to corpus membership only (no hallucination).

A BGG procedural floor (`Art. 100 Abs. 1 BGG` etc.) is always appended since these appear in nearly every Swiss Federal Court appellate gold set.

All predictions are emitted in **abbreviation form** (e.g. `Art. 221 Abs. 1 StPO`) to match the exact-string gold standard. Falls back to deterministic ordering if Ollama is unavailable.

---

## Setup

```bash
git clone https://github.com/dsdguofi/AI-ML-SP26
cd AI-ML-SP26/Law_Retrieval

python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Install and start [Ollama](https://ollama.com), then pull the model:

```bash
ollama serve          # in a separate terminal
ollama pull llama3.2:3b
```

---

## Data

Download the data zip and unzip into `data/`:

**[Download Data](https://drive.google.com/file/d/1o0HX5qUgPh79Vjd6UqLZeUocANWp7OSV/view)**

```
data/
├── train.csv                  # Training queries with gold citations
├── val.csv                    # 10 English validation queries with gold citations
├── test.csv                   # 40 English test queries — no labels
├── laws_de.csv                # Swiss federal law snippets, keyed by citation
├── court_considerations.csv   # Swiss Federal Court decisions (~30 years)
└── sample_submission.csv      # Required submission format
```

Build the BM25 index once (~2 minutes):

```bash
python index_builder.py
```

---

## Usage

**Batch prediction:**

```bash
# Val set (with per-query F1 breakdown)
python Law_evaluation.py --split val --out val_submission.csv
python evaluation/evaluate.py val_submission.csv --split val -v

# Test submission
python Law_evaluation.py --split test --out submission.csv

# Skip Ollama — fast deterministic fallback only
python Law_evaluation.py --split val --no-llm
```

**Interactive mode** — paste a case description, get citations:

```bash
python Law_evaluation.py --interactive
```

**Run tests:**

```bash
pytest tests/
```

---

## Project Structure

```
├── Law_evaluation.py       # Main pipeline (retriever + CLI)
├── citation_norm.py        # Citation extraction regex + SR↔abbreviation conversion
├── ollama_client.py        # Thin wrapper around Ollama HTTP API
├── prompts.py              # LLM prompt templates (translate, expand, rerank)
├── index_builder.py        # Builds BM25 index from laws_de.csv
├── notebook.ipynb          # Interactive demo notebook
├── evaluation/
│   ├── metrics.py          # Macro F1, Micro F1, MAP, NDCG
│   └── evaluate.py         # CLI scoring script
├── data/                   # All data files (gitignored)
└── tests/
    ├── conftest.py
    └── test_metrics.py
```

---

## Submission Format

One row per query, citations semicolon-separated:

```
query_id,predicted_citations
val_001,"Art. 221 Abs. 1 StPO;Art. 100 Abs. 1 BGG;BGE 141 I 141 E. 3.5"
val_002,"Art. 8 ZGB;Art. 97 Abs. 1 OR"
```

The test set has 40 English queries — 20 scored on the public leaderboard, 20 on the private.

---

## Scoring

Primary metric: **Macro F1**

- **Precision** = correct citations predicted / all citations predicted  
- **Recall** = correct citations predicted / all gold citations  
- **F1** = harmonic mean, averaged across all queries

> Citations must exactly match the corpus abbreviation form (e.g. `Art. 41 OR`, `BGE 116 Ia 56 E. 2b`). The pipeline grounds all predictions to the corpus — it never generates citation strings freely.
