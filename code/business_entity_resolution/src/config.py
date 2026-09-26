"""
Single source of truth for the blocking / reranking configuration.

Every number here was chosen from a measurement, and the measurement is named so
that a future change can be checked against the same evidence. See
reports/phase3_blocking.md for the full tables.

Production code imports from this module. Nothing in src/ should import
configuration from a diagnostic script -- that coupling existed briefly and made
the diagnostics load-bearing, which is the wrong dependency direction.
"""

from __future__ import annotations

from src.blocking import (
    ChannelSpec, feats_address, feats_digit_keys, feats_name_ngrams, feats_name_tokens,
)

# Retrieval depth per channel. Depth is nearly free (the same matmul with a deeper
# cut), whereas loosening max_df multiplies the matmul itself: measured, depth-1000
# at max_df 0.01 reaches 97.93% pool recall in 26 min/partition, while max_df 0.05
# at depth 50 reaches only 94.91% in 113 min. Going broad beats going loose.
DEPTH = 1000

# Reciprocal-rank-fusion constant. RRF replaced a weighted sum of raw cosines,
# which cost ~16 points of recall: trigram cosines are far larger in magnitude
# than address-token cosines, so name candidates evicted address candidates during
# the final cap -- fatal for transliterated records, whose only route is the address.
RRF_K0 = 20

# Threads for sp_matmul_topn. 24 measured fastest; 48 was SLOWER (4.2x vs 4.9x),
# which is the memory-bandwidth ceiling of this box, not a scheduling artefact.
THREADS = 24

# Candidates written to candidate_pairs.tsv. Measured rerank recall at K=50:
# 97.4% India / 99.3% US, against a pool ceiling of 98.7% / 99.4%.
TOP_K = 50

# max_df_frac 0.01: features present in more than 1% of the corpus are dropped.
# The original 0.002 was far too aggressive -- character trigrams are so skewed
# that the ~1,500 commonest ones carry most occurrences, so a tight threshold left
# queries with only 2.3 usable features each and cost ~5 points of recall.
MAX_DF_FRAC = 0.01

# The digit-key channel is deliberately tight: a digit run appearing in >0.2% of
# addresses ('1', '100') carries no signal, and this channel's whole value is its
# selectivity.
DIGIT_MAX_DF_FRAC = 0.002


def channel_specs(depth: int = DEPTH) -> tuple[ChannelSpec, ...]:
    """The four retrieval channels, in fusion order.

    Their measured solo recall@1000 on India: address 92.64%, name_ngram 71.93%,
    name_token 67.80%, digit_key 43.93%. No channel is redundant -- the union
    reaches 98.8% because they fail on different records. The address channel is
    what recovers transliterated Indic names (the name channel gets 0.2-1.0% of
    those) because street numbers survive a change of script untouched.
    """
    return (
        ChannelSpec("name_ngram", feats_name_ngrams, 1.0, depth, MAX_DF_FRAC, 20),
        ChannelSpec("address",    feats_address,     1.0, depth, MAX_DF_FRAC, 20),
        ChannelSpec("name_token", feats_name_tokens, 0.7, depth, MAX_DF_FRAC, 14),
        ChannelSpec("digit_key",  feats_digit_keys,  1.0, depth, DIGIT_MAX_DF_FRAC, 16),
    )
