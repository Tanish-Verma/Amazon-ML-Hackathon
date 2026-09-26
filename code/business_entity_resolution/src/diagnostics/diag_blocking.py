"""
Diagnostic for the Phase 3 recall shortfall.

The first full run returned macro recall 0.7102 against a measured lexical
"ceiling" of 99.88%. That gap has to be explained before tuning anything, and
there are two candidate explanations which imply completely different fixes:

  (a) RANKING. The true match IS reachable but is not inside the top-K, because
      other records outrank it. Fix: better scoring, and/or a larger K.
  (b) RETRIEVAL. The true match never enters the channel's candidate stream at
      all -- e.g. its features were pruned away. Fix: change pruning.

So this measures, per channel, the *rank* of every true match rather than just
whether it made top-30, and reports recall@K curves. It also tests one specific
suspicion about the scoring: the corpus vectors are L2-normalised over all their
features while queries are normalised over only their rarest few, which
systematically favours SHORT corpus names that happen to share one rare feature
over genuine long matches. The 'overlap' variant drops document normalisation to
see whether that is really what is costing us.

Runs on a sample so a cycle is ~minutes, not the 78 minutes a full run costs.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, ".")
from src.blocking import DEFAULT_CHANNELS, ChannelIndex  # noqa: E402
from src.candidates import extract_features_parallel, load_partition  # noqa: E402

KS = (10, 30, 50, 100, 200, 500)
PROBE_K = 500      # how deep we look before declaring a match unretrieved


def build_matrix(index: ChannelIndex, feats, prune, normalise_docs: bool) -> sp.csr_matrix:
    """Rebuild a matrix in the same feature space, optionally unnormalised."""
    indptr, indices, data = [0], [], []
    vocab, idf = index.vocab, index.idf
    for fs in feats:
        cols = {vocab[f] for f in fs if f in vocab}
        if prune is not None and len(cols) > prune:
            arr = np.fromiter(cols, dtype=np.int32, count=len(cols))
            keep = np.argpartition(-idf[arr], prune)[:prune]
            cols = arr[keep].tolist()
        if cols:
            cols = sorted(cols)
            w = idf[np.asarray(cols, dtype=np.int32)]
            if normalise_docs:
                w = w / np.linalg.norm(w)
            indices.extend(cols)
            data.extend(w.tolist())
        indptr.append(len(indices))
    return sp.csr_matrix(
        (np.asarray(data, dtype=np.float32), np.asarray(indices, dtype=np.int32),
         np.asarray(indptr, dtype=np.int64)), shape=(len(feats), len(vocab)))


def ranks_for(Q: sp.csr_matrix, C_T: sp.csr_matrix, q_rows, truth_rows, chunk=500):
    """Rank of each true match within each query's score list (None if > PROBE_K)."""
    out = defaultdict(list)
    for start in range(0, Q.shape[0], chunk):
        S = (Q[start:start + chunk] @ C_T).tocsr()
        for r in range(S.shape[0]):
            qi = start + r
            a, b = S.indptr[qi - start], S.indptr[qi - start + 1]
            idx, val = S.indices[a:b], S.data[a:b]
            if idx.size == 0:
                for _ in truth_rows[q_rows[qi]]:
                    out[q_rows[qi]].append(None)
                continue
            top = idx[np.argsort(-val)[:PROBE_K]]
            pos = {d: i for i, d in enumerate(top.tolist())}
            for t in truth_rows[q_rows[qi]]:
                out[q_rows[qi]].append(pos.get(t))
    return out


def report(label, rank_map):
    allr = [r for v in rank_map.values() for r in v]
    n = len(allr)
    line = f"  {label:<34}"
    for k in KS:
        hit = sum(1 for r in allr if r is not None and r < k)
        line += f"{100*hit/n:>8.2f}"
    unret = sum(1 for r in allr if r is None)
    line += f"{100*unret/n:>10.2f}"
    print(line)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--country", default="India")
    ap.add_argument("--sample", type=int, default=8000)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    q_ids, q_names, q_addrs = load_partition(args.store, "train", "s1", args.country)
    c_ids, c_names, c_addrs = [], [], []
    for src in ("s2", "s3"):
        i, n, a = load_partition(args.store, "train", src, args.country)
        c_ids += i; c_names += n; c_addrs += a
    crow = {e: i for i, e in enumerate(c_ids)}
    print(f"{args.country}: queries={len(q_ids):,} corpus={len(c_ids):,}")

    gt = {}
    with open(args.gt, encoding="utf-8") as f:
        next(f)
        for line in f:
            s1, _, rest = line.partition("\t")
            rest = rest.rstrip("\n")
            if rest:
                gt[s1] = rest.split(",")

    qset = {e: i for i, e in enumerate(q_ids)}
    rng = random.Random(0)
    cand = [e for e in q_ids if e in gt and any(m in crow for m in gt[e])]
    rng.shuffle(cand)
    cand = cand[:args.sample]
    sel = [qset[e] for e in cand]
    truth_rows = {qi: [crow[m] for m in gt[q_ids[qi]] if m in crow] for qi in sel}
    n_pairs = sum(len(v) for v in truth_rows.values())
    print(f"sample: {len(sel):,} queries, {n_pairs:,} true pairs\n")

    sub_names = [q_names[i] for i in sel]
    sub_addrs = [q_addrs[i] for i in sel]

    t0 = time.perf_counter()
    corpus_feats = extract_features_parallel(c_names, c_addrs, DEFAULT_CHANNELS, args.workers)
    query_feats = extract_features_parallel(sub_names, sub_addrs, DEFAULT_CHANNELS, args.workers)
    print(f"features in {time.perf_counter()-t0:.0f}s\n")

    hdr = "  " + "channel / scoring".ljust(34) + "".join(f"{f'@{k}':>8}" for k in KS) + f"{'unret%':>10}"
    print(hdr); print("  " + "-" * (len(hdr) - 2))

    for spec, cf, qf in zip(DEFAULT_CHANNELS, corpus_feats, query_feats):
        index = ChannelIndex(spec, cf)
        for norm_docs, tag in ((True, "cosine (current)"), (False, "overlap (no doc-norm)")):
            C_T = build_matrix(index, cf, None, norm_docs).T.tocsr()
            Q = build_matrix(index, qf, spec.query_terms, True)
            rm = ranks_for(Q, C_T, sel, truth_rows)
            report(f"{spec.name} · {tag}", rm)
            del C_T, Q
        # Does keeping ALL query features (no rarest-N pruning) help?
        C_T = build_matrix(index, cf, None, True).T.tocsr()
        Q = build_matrix(index, qf, None, True)
        report(f"{spec.name} · cosine, all query terms", ranks_for(Q, C_T, sel, truth_rows))
        del C_T, Q, index
    return 0


if __name__ == "__main__":
    sys.exit(main())
