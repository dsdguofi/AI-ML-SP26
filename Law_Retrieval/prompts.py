"""Prompt templates for the legal retrieval pipeline."""

TRANSLATE_QUERY = """Translate this legal question to German, French, and Italian. Be concise.

Question: {query}

Reply with ONLY this JSON (no markdown, no explanation):
{{"de": "...", "fr": "...", "it": "..."}}"""


EXPAND_LEGAL_TERMS = """List Swiss legal terms and statute abbreviations relevant to this question (e.g. ZGB, OR, StGB, BGG).

Question: {query}

Reply with ONLY this JSON (no markdown, no explanation):
{{"terms": ["term1", "term2", "term3"]}}"""


RERANK_CITATIONS = """You are a Swiss legal research assistant. You will be given an English legal question and a numbered list of Swiss legal citations, each with a short snippet from the relevant statute or court decision (in German/French/Italian).

Your task: select the citations that are directly relevant to answering the question and return them ranked from most to least relevant.

Question:
{query}

Candidates:
{candidates}

Return ONLY a JSON object with one key "ranked_citations" whose value is a list of citation strings copied VERBATIM from the candidates above.

Rules:
- Copy each citation string EXACTLY as it appears in square brackets (e.g. "Art. 221 Abs. 1 StPO").
- Rank by relevance to the question — most relevant first.
- Include UP TO {top_n} citations. Include fewer if most candidates are not relevant.
- Do not invent or modify citations.
- No commentary, no markdown fences, no explanation.

Example output:
{{"ranked_citations": ["Art. 221 Abs. 1 StPO", "Art. 100 Abs. 1 BGG"]}}"""


ANSWER_WITH_CITATIONS = """You are a Swiss legal advisor. Answer the following case in English, citing ONLY from the provided Swiss legal sources. Do not invent citations. If the sources are insufficient, say so plainly.

Case:
{query}

Sources (citation → snippet):
{sources}

Write a concise legal-advice response (3-6 short paragraphs). Inline-cite using the exact citation string in square brackets, e.g. [Art. 41 OR]. End with a "Citations:" line listing every citation you used, semicolon-separated."""
