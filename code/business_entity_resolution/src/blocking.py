"""
Phase 3 — lexical blocking / candidate generation.

This stage sets the **recall ceiling** for the whole system: a true match not
retrieved here is unrecoverable no matter how good the matcher is. Phase 1
measured that ceiling at 99.88% for a lexical approach (reports/eda.md §5), so
the job here is to realise it in a real index at scale.

Design, following the measurements rather than intuition:

* **Three independent channels, unioned.** The EDA is unambiguous that no single
  channel suffices: name alone reaches 90.43%, address alone 94.82%, the union
  99.88%. The address channel is what recovers transliterated Indic names (name
  alone gets 0.2-1.0% of those, the union 98-99%), because digit runs survive
  script changes untouched.
* **No embedding channel.** Measured worth: 0.12% extra recall, which cannot
  repay GPU hours under an F0.5 objective that weights precision 2x.
* **Per country partition.** Ground truth confirms no true match crosses a
  country boundary. This partitions on whatever label is present, so France
  works with no code change.

Retrieval mechanism. For each channel we build an IDF-weighted, L2-normalised
sparse matrix over the corpus, and for each query keep only its **rarest few
features**. That bound is what makes this tractable: a character trigram present
in a million records carries no signal but would dominate the cost. Retrieval is
then one sparse matmul per chunk, which runs at C speed, and top-K per row.
"""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import scipy.sparse as sp

from src.normalize import char_ngrams, clean_text, extract_numeric_keys, tokenize

# --------------------------------------------------------------------------
# Feature extractors -- one per channel
# --------------------------------------------------------------------------
# Each returns the feature strings for one record. Namespacing by prefix keeps
# channels from colliding if they are ever merged into one space.


def feats_name_ngrams(name: str, addr: str) -> List[str]:
    """Character trigrams of the name: script-agnostic and typo-robust."""
    return char_ngrams(clean_text(name), 3)


def feats_name_tokens(name: str, addr: str) -> List[str]:
    """Whole name tokens: catches word reordering, which n-grams blur."""
    return tokenize(clean_text(name))


def feats_address(name: str, addr: str) -> List[str]:
    """Digit runs plus address word tokens.

    Digit runs are listed first-class because reports/eda.md §5 found them the
    strongest script-invariant signal available -- a street number reads the same
    in Devanagari, Tamil and Latin. IDF then does the rest: a door number is rare
    and scores high, while 'road' is common and scores near zero, so there is no
    need for a hand-written address stopword table.
    """
    if not addr.strip():
        return []
    out = ["d:" + k for k in extract_numeric_keys(addr)]
    out += ["a:" + t for t in tokenize(clean_text(addr))]
    return out


def feats_digit_keys(name: str, addr: str) -> List[str]:
    """Digit runs from the address ONLY -- an exact-key channel.

    Separated from `feats_address` deliberately. In the combined address channel
    the digit features share L2 normalisation with dozens of address word tokens,
    so a shared door number is diluted by unshared street/city words. On its own
    the feature space is tiny and highly selective, which makes this both the
    cheapest channel to retrieve and the one the EDA says carries the most
    script-invariant signal: a street number reads identically in Devanagari,
    Tamil, French and Latin, so this is the channel that finds transliterated
    records. IDF then discards useless keys like '1' automatically.
    """
    if not addr.strip():
        return []
    return ["d:" + k for k in extract_numeric_keys(addr)]


@dataclass(frozen=True)
class ChannelSpec:
    name: str
    extract: Callable[[str, str], List[str]]
    weight: float          # contribution to the fused score
    top_k: int             # candidates retrieved from this channel alone
    max_df_frac: float     # features above this corpus fraction are dropped
    query_terms: int       # keep only this many rarest features per query


DEFAULT_CHANNELS: Tuple[ChannelSpec, ...] = (
    ChannelSpec("name_ngram", feats_name_ngrams, 1.0, 40, 0.002, 14),
    ChannelSpec("address",    feats_address,     1.0, 40, 0.002, 14),
    ChannelSpec("name_token", feats_name_tokens, 0.7, 25, 0.005, 10),
)


# --------------------------------------------------------------------------
# Sparse index
# --------------------------------------------------------------------------

class ChannelIndex:
    """IDF-weighted sparse index over one corpus for one feature family."""

    def __init__(self, spec: ChannelSpec, corpus_feats: Sequence[Sequence[str]]):
        self.spec = spec
        n_docs = len(corpus_feats)

        # Document frequency over the corpus, then prune non-discriminative
        # features. A feature present in a large fraction of the corpus cannot
        # separate anything and would dominate retrieval cost.
        df: Counter[str] = Counter()
        for fs in corpus_feats:
            df.update(set(fs))
        max_df = max(2, int(spec.max_df_frac * n_docs))
        # Filter FIRST, then enumerate: enumerating df.items() and filtering
        # inline leaves gaps in the index space and overruns the idf array.
        kept = [f for f, c in df.items() if c <= max_df]
        vocab = {f: i for i, f in enumerate(kept)}

        self.vocab = vocab
        self.n_docs = n_docs
        self.n_pruned = len(df) - len(vocab)

        # Smoothed IDF. Rare features dominate, which is the behaviour we want:
        # a shared door number should outweigh a shared 'ltd'.
        idf = np.zeros(len(vocab), dtype=np.float32)
        for f, i in vocab.items():
            idf[i] = np.log(n_docs / (1.0 + df[f])) + 1.0
        self.idf = idf

        # Stored transposed as (features x docs) so a (queries x features)
        # block multiplies straight into (queries x docs) scores.
        self.matrix = self._build(corpus_feats, prune_to_rarest=None).T.tocsr()

    def _build(self, feats: Sequence[Sequence[str]], prune_to_rarest: int | None) -> sp.csr_matrix:
        """Rows = records, cols = features, values = L2-normalised IDF."""
        indptr = [0]
        indices: List[int] = []
        data: List[float] = []
        vocab, idf = self.vocab, self.idf
        for fs in feats:
            cols = {vocab[f] for f in fs if f in vocab}
            if prune_to_rarest is not None and len(cols) > prune_to_rarest:
                # Keep only the rarest (highest-IDF) features of this query --
                # the bound that makes retrieval affordable.
                cols_arr = np.fromiter(cols, dtype=np.int32, count=len(cols))
                keep = np.argpartition(-idf[cols_arr], prune_to_rarest)[:prune_to_rarest]
                cols = cols_arr[keep].tolist()
            if cols:
                w = idf[np.asarray(sorted(cols), dtype=np.int32)]
                norm = np.linalg.norm(w)
                indices.extend(sorted(cols))
                data.extend((w / norm).tolist())
            indptr.append(len(indices))
        return sp.csr_matrix(
            (np.asarray(data, dtype=np.float32), np.asarray(indices, dtype=np.int32),
             np.asarray(indptr, dtype=np.int64)),
            shape=(len(feats), len(vocab)),
        )

    def query(self, query_feats: Sequence[Sequence[str]], chunk: int = 2000):
        """Yield (row_offset, indices_2d, scores_2d) top-K blocks per chunk."""
        Q = self._build(query_feats, prune_to_rarest=self.spec.query_terms)
        k = self.spec.top_k
        for start in range(0, Q.shape[0], chunk):
            block = Q[start:start + chunk]
            S = (block @ self.matrix).tocsr()
            yield start, _topk_rows(S, k)


def _topk_rows(S: sp.csr_matrix, k: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Top-k (doc_id, score) per row of a sparse score matrix."""
    out = []
    indptr, indices, data = S.indptr, S.indices, S.data
    for i in range(S.shape[0]):
        a, b = indptr[i], indptr[i + 1]
        idx, val = indices[a:b], data[a:b]
        if idx.size > k:
            sel = np.argpartition(-val, k)[:k]
            idx, val = idx[sel], val[sel]
        out.append((idx, val))
    return out
