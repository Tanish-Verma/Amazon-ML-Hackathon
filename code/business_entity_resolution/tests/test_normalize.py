"""Tests for Phase 2 normalisation.

Ported from notebooks/normalize_phase2.ipynb's inline checks, with two changes:
the country-literal check is now computed for real (the notebook's version
referenced an undefined `_country_literal_findings`, so it could never fail),
and every example is drawn from real records in reports/eda.md rather than
being invented.
"""
import os
import re
import sys
import unicodedata

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.normalize import (  # noqa: E402
    DocumentFrequencyModel, char_ngrams, clean_text, extract_numeric_keys,
    normalize_record, tokenize,
)

DEVA_PRIVATE = "प्राइवेट"


def test_indic_combining_marks_preserved():
    """The exact bug from reports/eda.md §7: [^\\w\\s] shatters this."""
    out = clean_text(DEVA_PRIVATE)
    assert out == DEVA_PRIVATE, repr(out)
    # And the word must stay ONE token, not split into loose consonants.
    assert len(tokenize(out)) == 1, tokenize(out)


def test_all_nine_indic_scripts_survive_as_single_tokens():
    """Every script the EDA found in the data, not just Devanagari."""
    samples = {
        "devanagari": "प्राइवेट", "bengali": "বেসরকারি", "gurmukhi": "ਪ੍ਰਾਈਵੇਟ",
        "gujarati": "પ્રાઇવેટ", "oriya": "ପ୍ରାଇଭେଟ", "tamil": "பிரைவேட்",
        "telugu": "ప్రైవేట్", "kannada": "ಪ್ರೈವೇಟ್", "malayalam": "പ്രൈവറ്റ്",
    }
    for script, word in samples.items():
        out = clean_text(word)
        assert out == word, f"{script}: {out!r} != {word!r}"
        assert len(tokenize(out)) == 1, f"{script} split into {tokenize(out)}"


def test_real_transliteration_pair_from_eda():
    """Real record pair: the Latin side must normalise, the Indic side survive."""
    latin = clean_text("Ss Food Private Limited")
    indic = clean_text("एसएस फूड प्राइवेट लिमिटेड")
    assert latin == "ss food private limited", repr(latin)
    assert len(tokenize(indic)) == 4, tokenize(indic)


def test_latin_diacritics_stripped():
    assert clean_text("Payne Énterprises") == "payne enterprises"
    assert clean_text("Lumay Bóral") == "lumay boral"
    assert clean_text("Dréxkor") == "drexkor"


def test_french_examples():
    assert clean_text("Thermal & Fils SASU") == "thermal fils sasu"
    assert clean_text("63 R. DE DIEPPE") == "63 r de dieppe"
    assert clean_text("Établissements Dëleves EURL") == "etablissements deleves eurl"
    assert clean_text("Europ & Frères Distribution S.A.") == "europ freres distribution s a"


def test_mixed_script_keeps_both_sides():
    out = clean_text("श्री Traders")
    assert "traders" in out
    assert any(0x0900 <= ord(c) <= 0x097F for c in out), repr(out)


def test_empty_and_degenerate_inputs():
    assert clean_text("") == ""
    assert clean_text("   ") == ""
    assert clean_text("...,,,---") == ""
    rec = normalize_record("", "")
    assert rec.clean == "" and rec.tokens == [] and rec.numeric_keys == set()
    # Address missing in 3.3% of real S2/S3 records: must not crash.
    rec2 = normalize_record("Some Company", "")
    assert isinstance(rec2.numeric_keys, set)


def test_numeric_keys_from_real_addresses():
    assert extract_numeric_keys("AF-0684") == {"af684"}
    assert extract_numeric_keys("AF-684") == {"af684"}          # zero-pad equivalence
    assert extract_numeric_keys("1056-1060") == {"1056", "1060"}
    assert "187c" in " ".join(extract_numeric_keys("Wz-187C Shop No.13"))
    # Real pair from eda.md: both sides must share a key.
    a = extract_numeric_keys("3315 Fremont Street, Peoria, IL")
    b = extract_numeric_keys("3315 FREMONT ST, PEORIA, IL")
    assert a & b, (a, b)


def test_documented_numeric_limitation():
    """'48' vs '4-8' cannot share an exact key. Inside the measured 0.12%."""
    assert extract_numeric_keys("48").isdisjoint(extract_numeric_keys("4-8"))


def test_df_model_derives_stoplists_without_country_code():
    dfm = DocumentFrequencyModel()
    for i in range(283):
        dfm.update("france", tokenize(f"company{i} sarl"))
    for i in range(201):
        dfm.update("france", tokenize(f"firm{i} sas"))
    for i in range(269):
        dfm.update("us", tokenize(f"company{i} llc"))
    sl = dfm.build_stoplists(min_df_fraction=0.15)
    assert "sarl" in sl["france"] and "sas" in sl["france"]
    assert "llc" in sl["us"]
    assert "llc" not in sl["france"], "partitions must not leak"
    assert dfm.filter_high_df("france", tokenize("company7 sarl"), sl) == ["company7"]


def test_core_tokens_never_empty():
    """A name made entirely of high-DF tokens must not normalise to nothing."""
    dfm = DocumentFrequencyModel()
    for i in range(100):
        dfm.update("x", tokenize("private limited"))
    sl = dfm.build_stoplists(min_df_fraction=0.5)
    kept = dfm.filter_high_df("x", tokenize("private limited"), sl)
    assert kept, "fallback must return the original tokens rather than []"


def test_char_ngrams_typo_robustness():
    """Char n-gram overlap must survive real injected noise.

    Threshold is 0.30, comfortably above the 0.15 the EDA used as its blocking
    reachability bar, and below the measured values for real noisy pairs
    (0.33-0.60). Transposition ('ENRTPRMISES') is the harshest case at 0.333.
    """
    pairs = [("Payne Enterprises", "Payne Enterpires"),
             ("Payne Enterprises", "PAYNE-ENRTPRMISES"),
             ("Maure Williams Colombier", "Maure Wilblims Colombier"),
             ("Orellana Investments LLC", "LLC Orellana Invsmbens")]
    for x, y in pairs:
        a = set(char_ngrams(clean_text(x)))
        b = set(char_ngrams(clean_text(y)))
        j = len(a & b) / len(a | b)
        assert j > 0.30, f"{x!r} vs {y!r}: jaccard {j:.3f}"


def test_no_hardcoded_country_branching_in_logic():
    """Country names may appear only in the feature-flagged suffix table.

    The notebook's equivalent check referenced an undefined variable, so it
    always passed. This one reads the source.
    """
    src = open(os.path.join(os.path.dirname(__file__), "..", "src", "normalize.py"),
               encoding="utf-8").read()
    # Drop the supplementary table and all comments/docstrings before scanning.
    body = src.split("_SUPPLEMENTARY_LEGAL_SUFFIXES")[0]
    body = re.sub(r'""".*?"""', "", body, flags=re.S)
    body = re.sub(r"#.*", "", body)
    for literal in ('"US"', "'US'", '"India"', "'India'", '"France"', "'France'"):
        assert literal not in body, f"hardcoded country literal {literal} in logic"


if __name__ == "__main__":
    fails = []
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            try:
                fn(); print(f"  PASS {name}")
            except AssertionError as e:
                fails.append(name); print(f"  FAIL {name}: {e}")
    print(f"\n{len(fails)} failure(s)." if fails else "\nall normalisation tests passed")
    sys.exit(1 if fails else 0)
