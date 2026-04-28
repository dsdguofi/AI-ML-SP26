"""Map between Swiss-law SR numbers (used in the corpus) and code abbreviations
(used in the gold citations). Also expand each predicted citation into all
equivalent surface forms so set-overlap scoring can match either side.
"""

from __future__ import annotations

import re

# Canonical Swiss-law abbreviations -> SR number (Systematische Rechtssammlung).
# Sourced from the official SR; covers every abbreviation observed in the
# competition's gold citations (StGB, StPO, ZGB, OR, BGG, BV, ...).
ABBREV_TO_SR: dict[str, str] = {
    # Federal Constitution & key public law
    "BV":     "101",
    "ParlG":  "171.10",
    "RVOG":   "172.010",
    "VwVG":   "172.021",
    "BGG":    "173.110",
    "VGG":    "173.32",
    "StBOG":  "173.71",
    "BPG":    "172.220.1",
    # Civil & commercial
    "ZGB":    "210",
    "PartG":  "211.231",
    "OR":     "220",
    "URG":    "231.1",
    "MSchG":  "232.11",
    "PatG":   "232.14",
    "DSG":    "235.1",
    "SchKG":  "281.1",
    # Procedure
    "ZPO":    "272",
    "IPRG":   "291",
    # Criminal law & procedure
    "StGB":   "311.0",
    "MStG":   "321.0",
    "JStG":   "311.1",
    "StPO":   "312.0",
    "JStPO":  "312.1",
    # Tax / social
    "DBG":    "642.11",
    "MWSTG":  "641.20",
    "AHVG":   "831.10",
    "IVG":    "831.20",
    "BVG":    "831.40",
    "UVG":    "832.20",
    "KVG":    "832.10",
    "ATSG":   "830.1",
    "AVIG":   "837.0",
    # Migration / asylum
    "AIG":    "142.20",
    "AsylG":  "142.31",
    "BüG":    "141.0",
    "BueG":   "141.0",
    # Environment / energy
    "USG":    "814.01",
    "GSchG":  "814.20",
    "EnG":    "730.0",
    # Traffic
    "SVG":    "741.01",
    "VRV":    "741.11",
    # Misc important
    "FINMAG": "956.1",
    "BankG":  "952.0",
    "GwG":    "955.0",
    "KAG":    "951.31",
    "FusG":   "221.301",
    "KG":     "251",
    "BÜPF":   "780.1",
}

SR_TO_ABBREV: dict[str, str] = {sr: abbr for abbr, sr in ABBREV_TO_SR.items()}

# Match the trailing law identifier in a citation. Either:
#   * a numeric SR like "210" or "311.0" or "172.220.1"
#   * an alphabetic abbreviation like "ZGB" / "StPO"
_TAIL_RE = re.compile(r"^(?P<head>.+?)\s+(?P<tail>(?:\d+(?:\.\d+)*|[A-Za-zÄÖÜäöüß]+))\s*$")

# Court citations sometimes have the issuance date wedged into the "E." field,
# e.g. "BGE 139 I 2 E. 1.12.2011" (12 December 2011). Gold cites the legal
# consideration number instead, e.g. "BGE 139 I 2 E. 5.1". We strip dd.mm.yyyy
# patterns so the bare consideration form stays.
_BGE_DATE_RE = re.compile(
    r"^(?P<prefix>(?:BGE\s+\S+\s+\S+\s+\S+|[\dA-Za-z]+_\d+/\d+))\s+E\.\s+\d{1,2}\.\d{1,2}\.\d{4}\s*$"
)
# Generic court-citation prefix detector (BGE / 1B_ / 4A_ / etc.)
_COURT_PREFIX_RE = re.compile(r"^(?:BGE\s|\d+[A-Za-z]+_\d+/\d+)")


def normalize_citation(citation: str) -> list[str]:
    """Return all equivalent surface forms of the citation.

    Always returns a list whose first element is the original (cleaned) form;
    additional forms are added when SR<->abbreviation conversion applies, or
    when court-citation date suffixes can be dropped.
    """
    c = re.sub(r"\s+", " ", citation.strip())
    out = [c]

    # Court citation with a date suffix in E.: drop the date so the bare
    # decision form (e.g. "BGE 139 I 2") matches gold considerations.
    if _BGE_DATE_RE.match(c):
        out.append(_BGE_DATE_RE.match(c).group("prefix"))

    m = _TAIL_RE.match(c)
    if not m:
        return _dedup(out)
    head = m.group("head").strip()
    tail = m.group("tail")

    # Numeric SR -> abbreviation
    if tail in SR_TO_ABBREV:
        out.append(f"{head} {SR_TO_ABBREV[tail]}")
    # Abbreviation -> numeric SR
    elif tail in ABBREV_TO_SR:
        out.append(f"{head} {ABBREV_TO_SR[tail]}")

    return _dedup(out)


def to_abbrev_form(citation: str) -> str:
    """Return the abbreviation-form of a statute citation, or the input unchanged.

    Gold uses abbreviation form (`Art. 221 Abs. 1 StPO`) but `laws_de` keys are
    SR-numeric (`Art. 221 Abs. 1 312.0`). Eval is exact-string set match, so
    predictions must be in abbreviation form.
    """
    c = re.sub(r"\s+", " ", citation.strip())
    m = _TAIL_RE.match(c)
    if not m:
        return c
    tail = m.group("tail")
    if tail in SR_TO_ABBREV:
        return f"{m.group('head').strip()} {SR_TO_ABBREV[tail]}"
    return c


def _dedup(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for v in items:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


# --- Citation extraction from free text ---

# Article-form: "Art. 41 OR" / "Art. 221 Abs. 1 lit. b StPO" / "Art. 41 Abs. 1 OR"
# Captures groups so we can emit broader forms (drop lit., drop Abs.) — gold
# usually cites at the article or Abs. level even when the source text adds lit.
_ART_RE = re.compile(
    r"Art\.?\s*(?P<num>\d+[a-z]?)"
    r"(?:\s+Abs\.?\s*(?P<abs>\d+[a-z]?))?"
    r"(?:\s+lit\.?\s*[a-z]+)?"
    r"(?:\s+Ziff\.?\s*\d+)?"
    r"\s+(?P<abbr>" + "|".join(re.escape(a) for a in ABBREV_TO_SR.keys()) + r")"
)
# BGE: "BGE 137 IV 122 E. 6.2"
_BGE_RE = re.compile(r"BGE\s+\d+\s+[IVX]+\s+\d+(?:\s+E\.?\s*\d+(?:\.\d+)*[a-z]?)?")
# Federal court ID: "1B_15/2023 E. 3.1"
_FED_RE = re.compile(r"\d+[A-Za-z]+_\d+/\d+(?:\s+E\.?\s*\d+(?:\.\d+)*[a-z]?)?")


def extract_citations(text: str) -> list[str]:
    """Find citation-shaped substrings in free text, in order of appearance.

    For statute (Art.) citations, also emit broader forms: an extracted
    `Art. X Abs. Y lit. z ABBR` yields `Art. X Abs. Y ABBR` and `Art. X ABBR`
    too, since gold often cites at the article level.
    """
    found: list[str] = []
    for m in _ART_RE.finditer(text):
        num = m.group("num")
        abs_ = m.group("abs")
        abbr = m.group("abbr")
        # Always emit article-level form.
        found.append(f"Art. {num} {abbr}")
        # If the source had Abs., also emit the Art.+Abs. form (drops lit./Ziff.).
        if abs_:
            found.append(f"Art. {num} Abs. {abs_} {abbr}")
        # And emit the verbatim match (with lit./Ziff. preserved) as a third form.
        verbatim = re.sub(r"\s+", " ", m.group(0).strip())
        found.append(verbatim)
    for pat in (_BGE_RE, _FED_RE):
        for m in pat.finditer(text):
            found.append(re.sub(r"\s+", " ", m.group(0).strip()))
    return _dedup(found)


def expand_citations(citations: list[str], cap: int | None = None) -> list[str]:
    """Expand a ranked citation list into all equivalent forms.

    Preserves rank order: the original citation comes first, then its
    abbreviation/SR alias (if any). This widens overlap with gold without
    sacrificing precision on the original prediction.
    """
    out: list[str] = []
    seen: set[str] = set()
    for c in citations:
        for form in normalize_citation(c):
            if form not in seen:
                seen.add(form)
                out.append(form)
                if cap is not None and len(out) >= cap:
                    return out
    return out
