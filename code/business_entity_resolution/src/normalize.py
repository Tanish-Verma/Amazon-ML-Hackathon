"""
Normalization utilities for the Business Entity Resolution Challenge.

Handles:
  - Legal-suffix stripping (Pvt/Private, Ltd/Limited, Corp/Corporation, LLC, LLP, Inc ...)
  - Domain / handle style names (bnpgroup.com, @jexfirst)
  - Punctuation, casing, bracket noise ([LLC], [Partners])
  - Address abbreviation expansion (Rd -> Road, St -> Street, ...)
  - Regional-script Indian state names -> canonical English
  - Landmark phrase stripping ("Near SBI ATM", "opp ...")

Design note: this module is deliberately conservative. It normalizes for
*comparison* purposes (tokenizing, blocking, feature computation) - it never
mutates or discards the original raw fields, which should always be kept
alongside the normalized versions for auditing / the methodology doc.
"""

import re
import unicodedata

# ---------------------------------------------------------------------------
# Legal suffix tokens to strip when building the "core name" for blocking.
# Kept as a set of lowercase tokens (post-tokenization, post-punctuation-strip).
# ---------------------------------------------------------------------------
LEGAL_SUFFIXES = {
    "pvt", "private", "ltd", "limited", "llp", "llc", "inc", "incorporated",
    "corp", "corporation", "co", "company", "companies", "group",
    "enterprises", "enterprise", "industries", "industry", "holdings",
    "holding", "services", "service", "solutions", "solution", "associates",
    "partners", "partnership", "trust", "foundation", "plc", "gmbh", "sa",
    "inc.", "co.", "corp.", "ltd.",
}

# Generic connector / stopword tokens that carry no discriminating signal
# for blocking (but are NOT legal suffixes).
STOPWORDS = {"of", "the", "and", "&", "for", "a", "an", "de", "la"}

# ---------------------------------------------------------------------------
# Address abbreviation expansion. Maps common variants -> one canonical form.
# Applied on whole tokens after lowercasing + punctuation strip.
# ---------------------------------------------------------------------------
ADDRESS_ABBREV = {
    "rd": "road", "st": "street", "str": "street", "ave": "avenue",
    "av": "avenue", "blvd": "boulevard", "dr": "drive", "ln": "lane",
    "hwy": "highway", "apt": "apartment", "fl": "floor", "flr": "floor",
    "no": "number", "sec": "sector", "twp": "township", "pkwy": "parkway",
    "ct": "court", "cir": "circle", "sq": "square", "ste": "suite",
}

# ---------------------------------------------------------------------------
# Regional-script -> English state name lookup. Only needs to be big enough
# to normalize the country's own official-language variants; extend as more
# scripts are observed in the real data (e.g. add Tamil, Telugu, Bengali
# entries as they show up).
# ---------------------------------------------------------------------------
STATE_SCRIPT_MAP = {
    # Hindi (Devanagari)
    "राजस्थान": "rajasthan", "उत्तर प्रदेश": "uttar pradesh", "महाराष्ट्र": "maharashtra",
    "गुजरात": "gujarat", "पंजाब": "punjab", "हरियाणा": "haryana", "बिहार": "bihar",
    "पश्चिम बंगाल": "west bengal", "केरल": "kerala", "कर्नाटक": "karnataka",
    "तमिलनाडु": "tamil nadu", "तेलंगाना": "telangana", "आंध्र प्रदेश": "andhra pradesh",
    "मध्य प्रदेश": "madhya pradesh", "दिल्ली": "delhi", "असम": "assam",
    "ओडिशा": "odisha", "झारखंड": "jharkhand", "छत्तीसगढ़": "chhattisgarh",
    # Tamil
    "தமிழ்நாடு": "tamil nadu", "கேரளா": "kerala", "கர்நாடகா": "karnataka",
    "ஆந்திரா பிரதேஷ்": "andhra pradesh", "தில்லி": "delhi",
}

_bracket_re = re.compile(r"\[[^\]]*\]|\([^)]*\)")
_handle_prefix_re = re.compile(r"^(www\.|@)")   # strip only the leading marker
_tld_suffix_re = re.compile(r"\.(com|in|org|net|co)\b", re.IGNORECASE)  # strip only the TLD, keep the name
_punct_re = re.compile(r"[^\w\s]")
_ws_re = re.compile(r"\s+")
_landmark_re = re.compile(
    r"\b(near|opp\.?|opposite|behind|beside|adjacent to)\b.*?(?=,|$)",
    re.IGNORECASE,
)


def _strip_diacritics_safe(text: str) -> str:
    """Strip Latin diacritics (café -> cafe) but never touch non-Latin scripts
    (Devanagari, Tamil, etc.) - NFKD-decomposing those would corrupt them."""
    out = []
    for ch in text:
        if ord(ch) < 0x2000:  # Latin/Latin-extended/punctuation range
            decomposed = unicodedata.normalize("NFKD", ch)
            out.append("".join(c for c in decomposed if not unicodedata.combining(c)))
        else:
            out.append(ch)
    return "".join(out)


def normalize_name(raw_name: str) -> dict:
    """
    Returns a dict:
      raw            - original string, untouched
      clean          - lowercased, punctuation/bracket/domain stripped
      tokens         - full token list (post-clean, stopwords removed)
      core_tokens    - tokens with legal suffixes ALSO removed (for blocking)
      has_legal_suffix - bool, useful as a feature later
    """
    if raw_name is None or (isinstance(raw_name, float)):  # NaN
        return {"raw": "", "clean": "", "tokens": [], "core_tokens": [],
                "has_legal_suffix": False}

    raw = str(raw_name)
    text = raw.strip()
    text = _bracket_re.sub(" ", text)          # strip [LLC], (Partners)
    text = _handle_prefix_re.sub("", text)     # strip leading www./@ (keep the handle itself)
    text = _tld_suffix_re.sub("", text)        # strip trailing .com/.in/etc (keep the name)
    text = _strip_diacritics_safe(text)
    text = text.lower()
    text = _punct_re.sub(" ", text)            # strip punctuation (keeps unicode letters)
    text = _ws_re.sub(" ", text).strip()

    tokens = [t for t in text.split(" ") if t and t not in STOPWORDS]
    has_suffix = any(t in LEGAL_SUFFIXES for t in tokens)
    core_tokens = [t for t in tokens if t not in LEGAL_SUFFIXES]

    return {
        "raw": raw,
        "clean": text,
        "tokens": tokens,
        "core_tokens": core_tokens if core_tokens else tokens,  # never empty out completely
        "has_legal_suffix": has_suffix,
    }


def normalize_address(raw_address) -> dict:
    """
    Returns a dict:
      raw        - original string (or "" if missing)
      clean      - lowercased, landmark-stripped, abbreviation-expanded
      tokens     - token list of the cleaned address
      is_missing - True if the input was NaN/empty
    """
    if raw_address is None or (isinstance(raw_address, float)):  # NaN
        return {"raw": "", "clean": "", "tokens": [], "is_missing": True}

    raw = str(raw_address)
    text = raw.strip()
    if not text:
        return {"raw": raw, "clean": "", "tokens": [], "is_missing": True}

    # Map any regional-script state name to English BEFORE lowercasing
    # (these scripts have no case, so order doesn't matter for them, but
    # keep the substitution ahead of the diacritic/lowercase pass for clarity)
    for script_form, english in STATE_SCRIPT_MAP.items():
        if script_form in text:
            text = text.replace(script_form, english)

    text = _landmark_re.sub(" ", text)
    text = _strip_diacritics_safe(text)
    text = text.lower()
    text = _punct_re.sub(" ", text)
    text = _ws_re.sub(" ", text).strip()

    tokens = []
    for tok in text.split(" "):
        if not tok:
            continue
        tokens.append(ADDRESS_ABBREV.get(tok, tok))

    return {"raw": raw, "clean": " ".join(tokens), "tokens": tokens, "is_missing": False}


if __name__ == "__main__":
    # Quick self-check against a few of the actual noisy examples we were shown.
    samples = [
        "of Group Bnp Companies-Delhi Limited",
        "bnpgroupcompaniesdelhi.com",
        "Svl Ttuaesb Pvt Ltd",
        "VANGUARD DATA COLLECTIVE [LLC]",
        "@jexfirst",
    ]
    for s in samples:
        print(s, "->", normalize_name(s))

    addr_samples = [
        "House No.-591, G.F, Bhai Parmanand Colony, Delhi, North West Delhi, Delhi",
        "राजस्थान, DOOR NO 187 P NO 13, BAJRANG DHAM RAIPURA, KOTA",
        "PLOT NO: ##51 FIRST FLOOR, GOVINDAN STREET, AYYAVOO COLONY, AMINJIK, ARAI, CHENNAI, தமிழ்நாடு",
        None,
    ]
    for a in addr_samples:
        print(a, "->", normalize_address(a))
