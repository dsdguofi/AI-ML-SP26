# Swiss Legal Citation Retrieval

Given an English legal question, retrieve the relevant Swiss legal citations from a German/French/Italian corpus. Scored on **Macro F1** — per-query precision/recall F1 averaged across all test queries.

**Current val score: Macro F1 = 0.2528** (10 queries, exact-string set match)

---

## How It Works

The pipeline has five stages:

1. **Query extraction** — regex-extract any citations mentioned verbatim in the English query (e.g. `Art. 221 Abs. 1 StPO`). These are high-precision and go first.
2. **LLM translation** — `llama3.2:3b` translates the query to German, French, and Italian so BM25 can match against the German corpus.
3. **BM25 retrieval** — sparse retrieval over 176k statute and court-decision chunks, run once per query variant. Inline citations found in retrieved chunks are harvested and scored by frequency.
4. **Topic fallback** — keyword heuristics add canonical articles for well-defined legal domains (criminal detention, disability insurance, inheritance, etc.) when ≥2 domain keywords match.
5. **LLM reranker** — `llama3.2:3b` picks the best 30 citations from the merged candidate pool, grounded to corpus membership.

All predictions are emitted in **abbreviation form** (e.g. `Art. 221 Abs. 1 StPO`) to match the exact-string gold standard.

---

## Setup

```bash
git clone https://github.com/alanyom/AI-ML-SP26
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

Build the BM25 index once (takes ~2 minutes):

```bash
python index_builder.py
```

---

## Usage

**Batch prediction (val or test):**

```bash
# Score against val set
python Law_evaluation.py --split val --out val_submission.csv

# Generate test submission
python Law_evaluation.py --split test --out submission.csv

# Skip LLM (fast deterministic fallback only)
python Law_evaluation.py --split val --no-llm
```

**Interactive mode** (paste a case, get citations):

```bash
python Law_evaluation.py --interactive
```

**Evaluate locally:**

```bash
python evaluation/evaluate.py val_submission.csv --split val -v
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
├── prompts.py              # LLM prompt templates
├── index_builder.py        # Builds BM25 index from laws_de.csv + court_considerations.csv
├── notebook.ipynb          # Interactive demo notebook
├── evaluation/
│   ├── metrics.py          # Macro F1, Micro F1, MAP, NDCG
│   └── evaluate.py         # CLI scoring script
├── data/                   # All data files go here
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

For each query:
- **Precision** = correct citations predicted / all citations predicted
- **Recall** = correct citations predicted / all gold citations
- **F1** = harmonic mean of precision and recall

Final score = average F1 across all queries.

> Citations must exactly match the corpus abbreviation form (e.g. `Art. 41 OR`, `BGE 116 Ia 56 E. 2b`). Never generate citations freely with an LLM — always ground them against the corpus.
