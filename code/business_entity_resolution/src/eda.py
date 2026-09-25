"""
Phase 1 -- exploratory data analysis.

The headline question this has to answer, because it decides how much of the
remaining budget goes into the embedding channel:

    What fraction of true matches carry NO usable lexical signal at all --
    neither a name a character-n-gram index could find, nor an address a token
    index could find?

If that number is near zero, embeddings are an optimisation and a lexical
pipeline can carry us. If it is several percent, embeddings are load-bearing
and Phase 3 must budget GPU time for them.

Everything here is computed from the provided training data only. Nothing is
conditioned on a fixed set of country labels: statistics are computed per
whatever `country` values appear, so the same code profiles France unchanged.
"""

from __future__ import annotations

import argparse
import os
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict

import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --------------------------------------------------------------------------
# Script detection -- generic over Unicode blocks, not a fixed language list
# --------------------------------------------------------------------------

def dominant_script(text: str) -> str:
    """Most common Unicode script among the letters of ``text``.

    Uses unicodedata's own character names rather than hardcoded codepoint
    ranges, so a script we have not seen before (any new country's writing
    system) is still classified correctly instead of falling into 'Other'.
    """
    counts: Counter[str] = Counter()
    for ch in text:
        if not ch.isalpha():
            continue
        try:
            name = unicodedata.name(ch)
        except ValueError:
            continue
        counts[name.split(" ")[0]] += 1
    if not counts:
        return "None"       # digits/punctuation only
    return counts.most_common(1)[0][0].title()


# --------------------------------------------------------------------------
# Minimal normalisation, local to the EDA
# --------------------------------------------------------------------------
# Deliberately separate from src/normalize.py: Phase 2 redesigns that module,
# and the EDA's job is to measure the data as it is, not to measure whatever
# the normaliser happens to do this week.

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+")


def basic_norm(text: str) -> str:
    """NFKC -> strip Latin diacritics -> casefold -> depunctuate."""
    text = unicodedata.normalize("NFKC", text)
    # Strip combining marks only where the base is Latin; decomposing Indic
    # scripts would destroy them, since their vowel signs are combining marks.
    out = []
    for ch in text:
        if ord(ch) < 0x0500:
            d = unicodedata.normalize("NFKD", ch)
            out.append("".join(c for c in d if not unicodedata.combining(c)))
        else:
            out.append(ch)
    text = "".join(out).casefold()
    return _WS.sub(" ", _PUNCT.sub(" ", text)).strip()


def char_ngrams(s: str, n: int = 3) -> set[str]:
    s = s.replace(" ", "")
    return {s[i:i + n] for i in range(len(s) - n + 1)} if len(s) >= n else ({s} if s else set())


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def tokens(s: str) -> set[str]:
    return {t for t in s.split(" ") if t}


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_store(store_dir: str, split: str, source: str) -> dict[str, tuple]:
    """Load one source into {entity_id: (name, address, country)}."""
    recs: dict[str, tuple] = {}
    for fn in sorted(os.listdir(store_dir)):
        if not fn.startswith(f"{split}_{source}_") or not fn.endswith(".parquet"):
            continue
        t = pq.read_table(os.path.join(store_dir, fn))
        for eid, name, addr, ctry in zip(
            t.column("entity_id").to_pylist(), t.column("business_name").to_pylist(),
            t.column("business_address").to_pylist(), t.column("country").to_pylist()):
            recs[eid] = (name, addr, ctry)
    return recs


def load_gt(path: str) -> dict[str, list[str]]:
    gt = {}
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            s1, _, rest = line.partition("\t")
            rest = rest.rstrip("\n")
            gt[s1] = rest.split(",") if rest else []
    return gt


# --------------------------------------------------------------------------
# Analyses
# --------------------------------------------------------------------------

# Thresholds for calling a true match "reachable by a lexical index". These are
# deliberately generous -- the point is to establish an OPTIMISTIC ceiling for a
# lexical pipeline, so that anything falling outside is genuinely hard rather
# than merely below a tight cutoff.
NAME_REACHABLE = 0.15    # char-3gram Jaccard between normalised names
ADDR_REACHABLE = 0.20    # token Jaccard between normalised addresses
NUM_RE = re.compile(r"\d+")


def numeric_keys(addr_norm: str) -> set[str]:
    """Digit runs in an address (street/door/PIN numbers).

    These survive translation, transliteration and word reordering untouched,
    which makes them the most script-invariant blocking signal available.
    """
    return set(NUM_RE.findall(addr_norm))


def analyse_pairs(gt, s1, s23, sample_entities, seed=0):
    """Classify a sample of true matches by which lexical channel could find them."""
    rng = random.Random(seed)
    matched = [k for k, v in gt.items() if v and k in s1]
    rng.shuffle(matched)
    chosen = matched[:sample_entities]

    stats = Counter()
    by_country = defaultdict(Counter)
    by_script = defaultdict(Counter)
    hard_examples = []

    for s1_id in chosen:
        n1, a1, c1 = s1[s1_id]
        n1n, a1n = basic_norm(n1), basic_norm(a1)
        g1, t1, k1 = char_ngrams(n1n), tokens(a1n), numeric_keys(a1n)

        for m_id in gt[s1_id]:
            rec = s23.get(m_id)
            if rec is None:
                continue
            n2, a2, _ = rec
            n2n, a2n = basic_norm(n2), basic_norm(a2)

            name_sim = jaccard(g1, char_ngrams(n2n))
            addr_sim = jaccard(t1, tokens(a2n)) if a2n else 0.0
            num_hit = bool(k1 & numeric_keys(a2n)) if a2n else False

            name_ok = name_sim >= NAME_REACHABLE
            addr_ok = (addr_sim >= ADDR_REACHABLE) or num_hit
            script = dominant_script(n2)

            stats["pairs"] += 1
            if not a2n:
                stats["addr_empty"] += 1
            if name_ok:
                stats["name_reachable"] += 1
            if addr_ok:
                stats["addr_reachable"] += 1
            if name_ok or addr_ok:
                stats["reachable"] += 1
            else:
                stats["UNREACHABLE"] += 1
                if len(hard_examples) < 12:
                    hard_examples.append((s1_id, n1, a1, m_id, n2, a2, c1))
            # Cases where only the address saves us -- these are what a
            # name-only pipeline would lose.
            if addr_ok and not name_ok:
                stats["addr_only"] += 1
            if name_ok and not addr_ok:
                stats["name_only"] += 1

            by_country[c1]["pairs"] += 1
            by_country[c1]["reachable"] += int(name_ok or addr_ok)
            by_script[script]["pairs"] += 1
            by_script[script]["reachable"] += int(name_ok or addr_ok)
            by_script[script]["name_ok"] += int(name_ok)

    return stats, by_country, by_script, hard_examples, len(chosen)


def script_census(recs, label, out):
    """Distribution of writing systems in names, per country."""
    per = defaultdict(Counter)
    for name, _addr, ctry in recs.values():
        per[ctry][dominant_script(name)] += 1
    out.append(f"\n**{label}**\n")
    out.append("| Country | " + " | ".join(["Script mix (names)"]) + " |")
    out.append("|---|---|")
    for ctry in sorted(per):
        tot = sum(per[ctry].values())
        mix = ", ".join(f"{s} {100*c/tot:.1f}%" for s, c in per[ctry].most_common(5))
        out.append(f"| {ctry} | {mix} |")


def df_profile(recs, out, top=18):
    """Highest document-frequency name tokens per country.

    This is the empirical basis for deriving stopword/legal-suffix handling from
    corpus statistics instead of a hand-written per-country table. If France's
    SARL/SAS/EURL land in the same frequency band as US/India's LLC/LTD/PVT,
    then a DF-based rule generalises to an unseen country for free.
    """
    per = defaultdict(Counter)
    ndocs = Counter()
    for name, _addr, ctry in recs.values():
        ndocs[ctry] += 1
        for t in set(tokens(basic_norm(name))):
            per[ctry][t] += 1
    out.append("\n| Country | Most frequent name tokens (document frequency %) |")
    out.append("|---|---|")
    for ctry in sorted(per):
        n = ndocs[ctry]
        s = ", ".join(f"`{t}` {100*c/n:.1f}" for t, c in per[ctry].most_common(top))
        out.append(f"| {ctry} | {s} |")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", required=True, help="parquet store dir from 'cli.py prep'")
    ap.add_argument("--gt", required=True, help="train_ground_truth.tsv")
    ap.add_argument("--out", default="reports/eda.md")
    ap.add_argument("--sample-entities", type=int, default=150_000)
    args = ap.parse_args()

    out: list[str] = ["# Phase 1 — EDA", ""]

    print("loading train sources...", flush=True)
    s1 = load_store(args.store, "train", "s1")
    s2 = load_store(args.store, "train", "s2")
    s3 = load_store(args.store, "train", "s3")
    s23 = {**s2, **s3}
    gt = load_gt(args.gt)
    print(f"  s1={len(s1):,} s2={len(s2):,} s3={len(s3):,} gt={len(gt):,}", flush=True)

    # ---- 1. Ground-truth shape -------------------------------------------
    sizes = Counter(len(v) for v in gt.values())
    n_sing = sizes[0]
    matched_ids = [m for v in gt.values() for m in v]
    out += ["## 1. Ground-truth shape", "",
            f"* {len(gt):,} Source-1 entities; **{n_sing:,} singletons ({100*n_sing/len(gt):.2f}%)**",
            f"* {len(matched_ids):,} matched IDs total; mean {len(matched_ids)/(len(gt)-n_sing):.2f} per matched entity",
            f"* max list size {max(sizes)}",
            "",
            "| List size | Entities |", "|---|---|"]
    for k in sorted(sizes)[:13]:
        out.append(f"| {k} | {sizes[k]:,} |")

    # ---- 2. Distractors ---------------------------------------------------
    claimed = set(matched_ids)
    d2 = sum(1 for k in s2 if k not in claimed)
    d3 = sum(1 for k in s3 if k not in claimed)
    out += ["", "## 2. Distractors (records matching nothing)", "",
            f"* Source 2: {d2:,} of {len(s2):,} ({100*d2/len(s2):.1f}%) match no Source-1 entity",
            f"* Source 3: {d3:,} of {len(s3):,} ({100*d3/len(s3):.1f}%)",
            "",
            "These are the precision hazard: they sit in the candidate pool looking plausible,",
            "and every one we accept costs a full point on a singleton or dilutes a matched entity."]

    # ---- 3. Script census -------------------------------------------------
    out += ["", "## 3. Writing systems"]
    script_census(s1, "Source 1 (train)", out)
    script_census(s2, "Source 2 (train)", out)
    script_census(s3, "Source 3 (train)", out)

    # ---- 4. Token DF profiles --------------------------------------------
    out += ["", "## 4. Name-token document frequency, per country", "",
            "Basis for deriving stopwords from corpus statistics rather than a hand-written",
            "per-country table."]
    df_profile(s1, out)

    # ---- 5. THE HEADLINE: lexical reachability ---------------------------
    print("analysing pair reachability...", flush=True)
    stats, by_country, by_script, hard, n_ent = analyse_pairs(
        gt, s1, s23, args.sample_entities)
    p = stats["pairs"]
    out += ["", "## 5. Lexical reachability of true matches  **(the headline)**", "",
            f"Sample: {n_ent:,} matched Source-1 entities, **{p:,} true pairs**.",
            f"A pair counts as reachable if normalised name char-3gram Jaccard >= {NAME_REACHABLE}",
            f"OR address token Jaccard >= {ADDR_REACHABLE} OR the addresses share a digit run.",
            "Thresholds are deliberately generous: this is an optimistic ceiling for a purely",
            "lexical pipeline, so whatever falls outside is genuinely hard.", "",
            "| Channel | Pairs | % of true matches |", "|---|---|---|",
            f"| Name reachable | {stats['name_reachable']:,} | {100*stats['name_reachable']/p:.2f}% |",
            f"| Address reachable | {stats['addr_reachable']:,} | {100*stats['addr_reachable']/p:.2f}% |",
            f"| **Either (lexical ceiling)** | **{stats['reachable']:,}** | **{100*stats['reachable']/p:.2f}%** |",
            f"| Address rescues a hopeless name | {stats['addr_only']:,} | {100*stats['addr_only']/p:.2f}% |",
            f"| Name rescues a hopeless address | {stats['name_only']:,} | {100*stats['name_only']/p:.2f}% |",
            f"| **Neither — needs embeddings** | **{stats['UNREACHABLE']:,}** | **{100*stats['UNREACHABLE']/p:.2f}%** |",
            f"| (of which, address was empty) | {stats['addr_empty']:,} | {100*stats['addr_empty']/p:.2f}% |",
            "", "### By country", "", "| Country | Pairs | Lexically reachable |", "|---|---|---|"]
    for c in sorted(by_country):
        cc = by_country[c]
        out.append(f"| {c} | {cc['pairs']:,} | {100*cc['reachable']/cc['pairs']:.2f}% |")

    out += ["", "### By writing system of the matched record's name", "",
            "| Script | Pairs | Name channel alone | Either channel |", "|---|---|---|---|"]
    for s, cc in sorted(by_script.items(), key=lambda kv: -kv[1]["pairs"])[:10]:
        out.append(f"| {s} | {cc['pairs']:,} | {100*cc['name_ok']/cc['pairs']:.1f}% | "
                   f"{100*cc['reachable']/cc['pairs']:.1f}% |")

    out += ["", "### Examples of unreachable pairs", "", "```"]
    for s1id, n1, a1, mid, n2, a2, c in hard:
        out += [f"[{c}] {s1id}  {n1!r}", f"          {a1!r}",
                f"     -> {mid}  {n2!r}", f"          {a2!r}", ""]
    out += ["```"]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print(f"\nwrote {args.out}")
    print(f"HEADLINE: lexical ceiling {100*stats['reachable']/p:.2f}%, "
          f"unreachable {100*stats['UNREACHABLE']/p:.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
