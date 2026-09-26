"""
Phase 2 — script-agnostic normalisation.

Extracted from notebooks/normalize_phase2.ipynb (authored as a notebook, so the
demo/`print` scaffolding has been dropped and the definitions kept verbatim).
Tests live in tests/test_normalize.py.

Design constraints, from PLAN.md and reports/eda.md:

* Handles Latin (including French-accented), Devanagari, Tamil, Telugu, Kannada,
  Bengali, Gujarati, Malayalam, Oriya and Gurmukhi — all confirmed present.
* **Never** strips Indic combining marks. Their vowel signs are Unicode
  categories Mn/Mc, which `\\w` does not match, so a `[^\\w\\s]` stripper shatters
  प्राइवेट into "प र इव ट". That bug was measured in our own EDA
  (reports/eda.md §7); this module strips by Unicode *category* instead —
  removing P*/S*/C*, keeping L*/N*/M*.
* No per-country rule tables as the primary mechanism. High-frequency
  stopword/legal-suffix removal is *derived* at runtime from per-partition token
  document frequency, which is why France's sarl/sas/eurl fall out in the same
  band as US llc/inc with no French-specific code. A hand-written suffix list
  exists only behind a feature flag, for Phase 7 to ablate.
* Does not transliterate Indic scripts. Per reports/eda.md §5-6 the ADDRESS
  channel recovers transliterated-name matches (98-99%, versus 0.2-1.0% from the
  name alone), because digits survive script differences untouched.
"""



from __future__ import annotations

import re
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

try:
    import regex as _re_engine  # type: ignore
    _HAS_REGEX_MODULE = True
    _STRIP_RE = _re_engine.compile(r"[\p{P}\p{S}\p{C}]+")
except ImportError:
    _HAS_REGEX_MODULE = False
    _STRIP_RE = None


_KEEP_CATEGORY_PREFIXES = ("L", "N", "M")


def _strip_by_category_stdlib(s: str) -> str:
    """Fallback punctuation/symbol stripper using unicodedata.category().
    Equivalent to the `regex` \\p{P}\\p{S}\\p{C} pass, character by
    character. Correctly reports Devanagari vowel signs as 'Mn'/'Mc',
    which we explicitly keep."""
    out = []
    prev_was_space = False
    for ch in s:
        cat = unicodedata.category(ch)
        if cat[0] in _KEEP_CATEGORY_PREFIXES:
            out.append(ch)
            prev_was_space = False
        else:
            if not prev_was_space:
                out.append(" ")
                prev_was_space = True
    return "".join(out)


def strip_punct_symbols_keep_marks(s: str) -> str:
    """Strip P*/S*/C* Unicode categories, keep L*/N*/M*. Script-agnostic."""
    if _HAS_REGEX_MODULE:
        return _STRIP_RE.sub(" ", s)
    return _strip_by_category_stdlib(s)


# quick sanity check: the exact bug from reports/eda.md Sec.7


_INDIC_RANGES: Tuple[Tuple[str, int, int], ...] = (
    ("devanagari", 0x0900, 0x097F),
    ("bengali", 0x0980, 0x09FF),
    ("gurmukhi", 0x0A00, 0x0A7F),
    ("gujarati", 0x0A80, 0x0AFF),
    ("oriya", 0x0B00, 0x0B7F),
    ("tamil", 0x0B80, 0x0BFF),
    ("telugu", 0x0C00, 0x0C7F),
    ("kannada", 0x0C80, 0x0CFF),
    ("malayalam", 0x0D00, 0x0D7F),
)


def _char_class(ch: str) -> str:
    """'indic' (any of the 9 scripts) or 'other' (Latin incl. accented,
    digits, punctuation, whitespace, or any non-Indic script -- French,
    Cyrillic, etc. all fall here and get the same diacritic-stripping
    treatment, a no-op for scripts with nothing to strip)."""
    cp = ord(ch)
    for _name, lo, hi in _INDIC_RANGES:
        if lo <= cp <= hi:
            return "indic"
    return "other"


def _split_runs(s: str) -> List[Tuple[str, str]]:
    """Group consecutive same-class characters into (class, text) runs."""
    if not s:
        return []
    runs: List[Tuple[str, str]] = []
    cur_class = _char_class(s[0])
    cur_chars = [s[0]]
    for ch in s[1:]:
        cls = _char_class(ch)
        if cls == cur_class:
            cur_chars.append(ch)
        else:
            runs.append((cur_class, "".join(cur_chars)))
            cur_class, cur_chars = cls, [ch]
    runs.append((cur_class, "".join(cur_chars)))
    return runs


def _strip_diacritics_latin(s: str) -> str:
    """NFKD-decompose and drop combining marks. Only ever called on
    'other'-class runs (Latin/accented-Latin/etc), never on Indic runs."""
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def clean_text(raw: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        return ""
    s = unicodedata.normalize("NFKC", raw)
    runs = _split_runs(s)
    pieces = []
    for cls, text in runs:
        if cls == "other":
            pieces.append(_strip_diacritics_latin(text))
        else:
            pieces.append(text)  # indic run, untouched
    joined = "".join(pieces)
    joined = strip_punct_symbols_keep_marks(joined)
    joined = joined.casefold()
    joined = re.sub(r"\s+", " ", joined).strip()
    return joined


def tokenize(clean: str) -> List[str]:
    """Whitespace tokenization over an already-cleaned string."""
    return clean.split() if clean else []


_CODE_CHUNK_RE = re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*")


def extract_numeric_keys(raw: str) -> Set[str]:
    """'AF-0684' -> {'af684'}; '1056-1060' -> {'1056','1060'} (range);
    '1056c' -> {'1056c'}."""
    if not isinstance(raw, str) or not raw.strip():
        return set()
    keys: Set[str] = set()
    for m in _CODE_CHUNK_RE.finditer(raw):
        chunk = m.group(0)
        if not any(c.isdigit() for c in chunk):
            continue
        parts = chunk.lower().split("-")
        if len(parts) > 1 and all(p.isdigit() for p in parts):
            for p in parts:
                keys.add(p.lstrip("0") or "0")
        else:
            joined = "".join(
                p.lstrip("0") if p.isdigit() and p.lstrip("0") else (p if not p.isdigit() else "0")
                for p in parts
            )
            if joined:
                keys.add(joined)
    return keys


def char_ngrams(clean: str, n: int = 3) -> List[str]:
    """Character n-grams over the cleaned (space-collapsed) string."""
    s = clean.replace(" ", "")
    if len(s) < n:
        return [s] if s else []
    return [s[i : i + n] for i in range(len(s) - n + 1)]


_SUPPLEMENTARY_LEGAL_SUFFIXES: Dict[str, str] = {
    # India -- abbreviations
    "ltd": "limited", "pvt": "private", "co": "company", "llp": "llp",
    # India -- full words (added: these are the higher-DF forms, per EDA)
    "limited": "limited", "private": "private", "company": "company",
    # US
    "llc": "llc", "inc": "incorporated", "corp": "corporation",
    "pc": "pc", "l": "l", "p": "p",
    "corporation": "corporation", "incorporated": "incorporated",
    # France -- abbreviations
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "eurl": "eurl",
    "sci": "sci", "sa": "sa",
    # France -- full/expanded forms seen in French company law
    "societe": "societe",
}


def apply_supplementary_suffixes(tokens: List[str]) -> List[str]:
    return [_SUPPLEMENTARY_LEGAL_SUFFIXES.get(t, t) for t in tokens]


@dataclass
class DocumentFrequencyModel:
    """Streaming per-partition (e.g. per-country) token document frequency."""

    doc_count: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    token_doc_count: Dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))

    def update(self, partition: str, tokens: Iterable[str]) -> None:
        """Feed one record's (partition, unique-token-set). Streams --
        no need to hold the corpus in RAM."""
        uniq = set(tokens)
        if not uniq:
            return
        self.doc_count[partition] += 1
        c = self.token_doc_count[partition]
        for t in uniq:
            c[t] += 1

    def df_fraction(self, partition: str, token: str) -> float:
        n = self.doc_count.get(partition, 0)
        if n == 0:
            return 0.0
        return self.token_doc_count[partition].get(token, 0) / n

    def build_stoplists(self, min_df_fraction: float = 0.05) -> Dict[str, Set[str]]:
        """Per-partition stoplist: tokens whose DF exceeds min_df_fraction.

        Default raised from 0.02 to 0.05 on measured evidence (reports/phase2.md
        Sec.3): at 0.02 the US stoplist also swallows `care`, `associates`,
        `center`, `group`, `partners` and `corp`, which are discriminative
        business words. At 0.05 it catches legal-form boilerplate and country
        names only. Phase 3 should still tune this against end-to-end F0.5.
        A threshold, not a fixed top-K, so it scales to however many
        distinct legal-suffix/function-word tokens a country's corpus has."""
        stoplists: Dict[str, Set[str]] = {}
        for partition, n in self.doc_count.items():
            if n == 0:
                stoplists[partition] = set()
                continue
            thresh = n * min_df_fraction
            stoplists[partition] = {
                tok for tok, cnt in self.token_doc_count[partition].items() if cnt >= thresh
            }
        return stoplists

    def filter_high_df(self, partition: str, tokens: List[str], stoplists: Dict[str, Set[str]]) -> List[str]:
        stop = stoplists.get(partition, set())
        kept = [t for t in tokens if t not in stop]
        return kept if kept else list(tokens)


# demo: France sarl/sas should land in the same high-DF band as US llc/inc,
# derived purely from counts -- no country-specific code anywhere above.


@dataclass
class NormalizedRecord:
    clean: str
    tokens: List[str]
    core_tokens: List[str]
    joined: str
    char_ngrams: List[str]
    numeric_keys: Set[str] = field(default_factory=set)


def normalize_record(
    raw_name: str,
    raw_address: str = "",
    *,
    partition: Optional[str] = None,
    dfm: Optional[DocumentFrequencyModel] = None,
    stoplists: Optional[Dict[str, Set[str]]] = None,
    use_supplementary_suffixes: bool = False,
    ngram_n: int = 3,
) -> NormalizedRecord:
    """Normalize one (name, address) pair into the five required outputs.
    core_tokens needs a fitted DocumentFrequencyModel + stoplists for the
    record's partition (country); without one it falls back to tokens
    unchanged (useful for ad-hoc calls with no corpus to fit against)."""
    clean = clean_text(raw_name)
    tokens = tokenize(clean)

    if use_supplementary_suffixes:
        tokens = apply_supplementary_suffixes(tokens)

    if dfm is not None and stoplists is not None and partition is not None:
        core_tokens = dfm.filter_high_df(partition, tokens, stoplists)
    else:
        core_tokens = list(tokens)

    joined = "".join(tokens)
    ngrams = char_ngrams(clean, n=ngram_n)
    numeric_keys = extract_numeric_keys(raw_address) | extract_numeric_keys(raw_name)

    return NormalizedRecord(
        clean=clean, tokens=tokens, core_tokens=core_tokens,
        joined=joined, char_ngrams=ngrams, numeric_keys=numeric_keys,
    )


def measure_throughput(samples: List[str], iterations: int = 1) -> float:
    """Returns records/sec for clean_text() over `samples`."""
    t0 = time.perf_counter()
    for _ in range(iterations):
        for s in samples:
            clean_text(s)
    elapsed = time.perf_counter() - t0
    total = len(samples) * iterations
    return total / elapsed if elapsed > 0 else float("inf")

