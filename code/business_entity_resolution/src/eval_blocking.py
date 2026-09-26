"""
Blocking quality measurement: recall ceiling, reduction ratio, candidate-set size
distribution -- reported per country, since an unseen country is the main risk.

Macro recall is the number that bounds everything downstream: it is the most the
matcher could possibly achieve even if it were perfect.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Mapping, Sequence


def evaluate(candidates: Mapping[str, Sequence[str]],
             truth: Mapping[str, Sequence[str]],
             country_of: Mapping[str, str],
             corpus_sizes: Mapping[str, int]) -> str:
    """Return a markdown report. `truth` defines the evaluation set."""
    per_country: Dict[str, Counter] = defaultdict(Counter)
    per_country_recall: Dict[str, List[float]] = defaultdict(list)
    sizes: Dict[str, List[int]] = defaultdict(list)
    missed_all = 0
    lines: List[str] = []

    for s1, true_iter in truth.items():
        true_set = set(true_iter)
        cands = set(candidates.get(s1, ()))
        ctry = country_of.get(s1, "?")
        sizes[ctry].append(len(cands))
        c = per_country[ctry]
        c["entities"] += 1
        if not true_set:
            c["singletons"] += 1
            continue
        found = len(true_set & cands)
        c["true"] += len(true_set)
        c["found"] += found
        per_country_recall[ctry].append(found / len(true_set))
        if found == 0:
            missed_all += 1

    lines.append("| Country | Entities | Macro recall | Micro recall | Avg cands | Max | Reduction ratio |")
    lines.append("|---|---|---|---|---|---|---|")
    tot_macro: List[float] = []
    tot_true = tot_found = 0
    for ctry in sorted(per_country):
        c = per_country[ctry]
        rec = per_country_recall[ctry]
        macro = sum(rec) / len(rec) if rec else float("nan")
        micro = c["found"] / c["true"] if c["true"] else float("nan")
        avg = sum(sizes[ctry]) / len(sizes[ctry])
        n_corpus = corpus_sizes.get(ctry, 0)
        rr = 1 - (avg / n_corpus) if n_corpus else float("nan")
        lines.append(f"| {ctry} | {c['entities']:,} | {macro:.4f} | {micro:.4f} | "
                     f"{avg:.1f} | {max(sizes[ctry])} | {rr:.6f} |")
        tot_macro += rec
        tot_true += c["true"]; tot_found += c["found"]

    overall_macro = sum(tot_macro) / len(tot_macro) if tot_macro else float("nan")
    overall_micro = tot_found / tot_true if tot_true else float("nan")
    all_sizes = [s for v in sizes.values() for s in v]
    lines.append(f"| **ALL** | {sum(c['entities'] for c in per_country.values()):,} | "
                 f"**{overall_macro:.4f}** | {overall_micro:.4f} | "
                 f"{sum(all_sizes)/len(all_sizes):.1f} | {max(all_sizes)} | |")
    lines.append("")
    lines.append(f"* Entities whose every true match was missed: **{missed_all:,}**")
    lines.append(f"* Singletons in the evaluation set: "
                 f"{sum(c['singletons'] for c in per_country.values()):,}")
    return "\n".join(lines)
