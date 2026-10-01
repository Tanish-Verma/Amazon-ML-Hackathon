"""Phase 4 train/validation feature extraction, country by country.

Requires Phase 3 *enriched* pair TSV with columns:
source1_entity_id, candidate_entity_id, rrf_score, n_channels_hit,
best_channel_rank, reranker_score

The currently produced 2-column cand_*.tsv files are insufficient: Phase 3
throws away retrieval metadata and reranker scores. Do not silently fabricate
those columns. Extend the Phase 3 writer to export an enriched sidecar, or
regenerate candidates with the metadata retained.

Outputs per-country parquet feature chunks and per-feature validation AUC.
Uses SQLite on the user's home filesystem for full-partition reverse ranks;
never writes /tmp or deletes pre-existing files.
"""
from __future__ import annotations

import argparse
import csv
import os
import sqlite3
import time
from pathlib import Path

import numpy as np
import psutil
import pyarrow as pa
import pyarrow.parquet as pq

from src.candidates import load_partition
from src.features import (FEATURE_NAMES, address_document_frequencies,
                          build_features, feature_auc)
from src.io_utils import read_ground_truth
from src.rerank_vec import build_store_streaming

COLUMNS = ["source1_entity_id", "candidate_entity_id", "rrf_score",
           "n_channels_hit", "best_channel_rank", "reranker_score"]


def check_memory(min_free_gb):
    mem = psutil.virtual_memory()
    swap = psutil.swap_memory()
    print(f"available RAM {mem.available/1e9:.1f}GB; swap used {swap.used/1e9:.1f}GB", flush=True)
    if mem.available < min_free_gb * 1e9:
        raise RuntimeError(f"Need {min_free_gb}GB free RAM; aborting before allocation")


def read_pairs(path):
    with open(path, newline="", encoding="utf8") as f:
        r = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        if r.fieldnames != COLUMNS:
            raise ValueError(f"{path} must have exactly {COLUMNS}; got {r.fieldnames}. "
                             "The existing 2-column Phase 3 output loses required metadata.")
        for row in r:
            yield row


def create_rank_db(path, pairs, valid_queries, valid_candidates):
    """First pass: all contenders of one country; SQL window gives global rank."""
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}; choose a new --out directory")
    db = sqlite3.connect(str(path))
    db.execute("PRAGMA temp_store=FILE")  # SQLite temp files: place inside home via TMPDIR
    db.execute("CREATE TABLE contenders (qid TEXT, cid TEXT, score REAL, rrf REAL, hits REAL, br REAL, PRIMARY KEY(qid,cid))")
    buf = []
    count = 0
    for row in read_pairs(pairs):
        q, c = row["source1_entity_id"], row["candidate_entity_id"]
        if q not in valid_queries:
            continue
        if c not in valid_candidates:
            raise ValueError(f"Candidate {c} absent from this country's S2/S3 corpus")
        vals = (q, c, float(row["reranker_score"]), float(row["rrf_score"]),
                float(row["n_channels_hit"]), float(row["best_channel_rank"]))
        if not np.isfinite(np.asarray(vals[2:], dtype=float)).all():
            raise ValueError(f"Nonfinite metadata for {q} / {c}")
        buf.append(vals)
        if len(buf) == 20000:
            db.executemany("INSERT INTO contenders VALUES (?,?,?,?,?,?)", buf)
            db.commit()
            count += len(buf)
            buf.clear()
    if buf:
        db.executemany("INSERT INTO contenders VALUES (?,?,?,?,?,?)", buf)
        db.commit()
        count += len(buf)
    db.execute("CREATE INDEX idx_contenders_cid ON contenders(cid,score DESC,qid)")
    db.commit()
    print(f"staged {count:,} pairs for global reverse ranks", flush=True)
    return db, count


def process_country(args, country, truth):
    """Do not batch reverse ranking: stage the entire country before extracting."""
    t0 = time.perf_counter()
    check_memory(args.min_free_gb)
    q_ids, q_names, q_addr = load_partition(args.store, args.split, "s1", country)
    c_ids, c_names, c_addr, is_s3 = [], [], [], []
    for source in ("s2", "s3"):
        ids, names, addrs = load_partition(args.store, args.split, source, country)
        c_ids.extend(ids); c_names.extend(names); c_addr.extend(addrs)
        is_s3.extend([float(source == "s3")] * len(ids))
    if not q_ids:
        raise ValueError(f"No queries for country {country}")
    if len(set(c_ids)) != len(c_ids):
        raise ValueError("S2/S3 candidate IDs collide; source-qualify IDs before running")
    qmap, cmap = {x: i for i, x in enumerate(q_ids)}, {x: i for i, x in enumerate(c_ids)}
    out = Path(args.out) / country
    out.mkdir(parents=True, exist_ok=False)
    db, n = create_rank_db(out / "contenders.sqlite", args.pairs, qmap, cmap)
    if not n:
        raise ValueError("No pairs matched the requested country")
    check_memory(args.min_free_gb)
    cstore = build_store_streaming(c_names, c_addr, args.workers)
    df = address_document_frequencies(cstore)
    print(f"corpus sparse store {cstore.nbytes():.2f}GB; extracting features", flush=True)
    # Rank once over the WHOLE country's contender set, not individual batches.
    db.execute("CREATE TABLE reverse (qid TEXT, cid TEXT, rr INTEGER, PRIMARY KEY(qid,cid))")
    db.execute("INSERT INTO reverse SELECT qid,cid, ROW_NUMBER() OVER "
               "(PARTITION BY cid ORDER BY score DESC,qid ASC) FROM contenders")
    db.commit()
    qstore = None
    # All candidates for each S1 are always fetched together, even at batch edges.
    query_rows = db.execute("SELECT DISTINCT qid FROM contenders ORDER BY qid")
    query_batch, part, auc_X, auc_y, n_processed = [], 0, [], [], 0

    def emit(qids, part):
        nonlocal qstore, n_processed
        if not qids:
            return
        check_memory(args.min_free_gb)
        qidx = [qmap[q] for q in qids]
        qstore = build_store_streaming([q_names[i] for i in qidx],
                                       [q_addr[i] for i in qidx], args.workers,
                                       vocabs=cstore.vocabs)
        local = {q: i for i, q in enumerate(qids)}
        # SQLite join reads full query groups and their precomputed global ranks.
        # Query IDs are exact keys; no country name is ever a model column.
        rows = []
        for qid in qids:
            rows.extend(db.execute("SELECT c.cid,c.rrf,c.hits,c.br,c.score,r.rr "
                                   "FROM contenders c JOIN reverse r USING(qid,cid) "
                                   "WHERE c.qid=? ORDER BY c.score DESC,c.cid", (qid,)).fetchall())
        sizes = [db.execute("SELECT COUNT(*) FROM contenders WHERE qid=?", (qid,)).fetchone()[0]
                 for qid in qids]
        qi = np.repeat(np.arange(len(qids), dtype=np.int32), sizes)
        ci = np.array([cmap[r[0]] for r in rows], dtype=np.int32)
        rrf = np.array([r[1] for r in rows], dtype=np.float32)
        hits = np.array([r[2] for r in rows], dtype=np.float32)
        br = np.array([r[3] for r in rows], dtype=np.float32)
        scores = np.array([r[4] for r in rows], dtype=np.float32)
        rr = np.array([r[5] for r in rows], dtype=np.float32)
        X = build_features(qstore, cstore, qi, ci, rrf, hits, br,
                           np.asarray(is_s3, dtype=np.float32)[ci], scores, rr, df,
                           workers=args.workers,
                           q_raw_addrs=[q_addr[i] for i in qidx], c_raw_addrs=c_addr)
        labels = np.array([int(rows[j][0] in truth.get(qids[qi[j]], set()))
                           for j in range(len(rows))], dtype=np.uint8) if truth is not None else None
        columns = {name: pa.array(X[:, j]) for j, name in enumerate(FEATURE_NAMES)}
        columns["source1_entity_id"] = pa.array([qids[i] for i in qi])
        columns["candidate_entity_id"] = pa.array([r[0] for r in rows])
        if labels is not None:
            columns["label"] = pa.array(labels)
            # Validation AUC: bounded sample, not an 87M-pair RAM allocation.
            if sum(len(a) for a in auc_y) < args.auc_sample:
                remaining = args.auc_sample - sum(len(a) for a in auc_y)
                auc_X.append(X[:remaining].copy()); auc_y.append(labels[:remaining].copy())
        path = out / f"features_{part:05d}.parquet"
        pq.write_table(pa.table(columns), path, compression="zstd")
        n_processed += len(rows)
        print(f"{country}: {n_processed:,} pairs; {n_processed/max(1,time.perf_counter()-t0):,.0f} pairs/s; "
              f"output {path.name}", flush=True)

    for (qid,) in query_rows:
        query_batch.append(qid)
        if len(query_batch) >= args.query_batch:
            emit(query_batch, part)
            query_batch = []; part += 1
    if query_batch:
        emit(query_batch, part)
    duration = time.perf_counter() - t0
    if auc_X:
        X = np.vstack(auc_X); y = np.concatenate(auc_y)
        if len(np.unique(y)) == 2:
            aucs = feature_auc(X, y)
            with open(out / "feature_auc.tsv", "w") as f:
                f.write("feature\tauc\tdirection_neutral_auc\n")
                for name, a, adjusted in aucs:
                    f.write(f"{name}\t{a:.6f}\t{adjusted:.6f}\n")
            print(f"AUC written from {len(y):,} sampled labelled pairs", flush=True)
        else:
            print("AUC unavailable: sampled pairs contain only one class", flush=True)
    print(f"{country}: {n_processed:,} pairs in {duration:.1f}s; "
          f"{n_processed/max(duration,1):,.0f} pairs/s; "
          f"87M extrapolation {87e6/max(n_processed/max(duration,1),1)/3600:.1f}h", flush=True)
    db.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", required=True, help="ENRICHED Phase 3 TSV, not 2-column candidate list")
    ap.add_argument("--store", required=True)
    ap.add_argument("--split", choices=["train", "test"], default="train")
    ap.add_argument("--country", required=True, help="one country per run")
    ap.add_argument("--gt", help="train ground truth TSV; omit for test")
    ap.add_argument("--out", required=True, help="new output directory under your home")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--query-batch", type=int, default=100)
    ap.add_argument("--auc-sample", type=int, default=500000)
    ap.add_argument("--min-free-gb", type=float, default=25.0)
    args = ap.parse_args()
    if args.split == "train" and not args.gt:
        ap.error("--gt required for train AUC")
    if args.query_batch < 1:
        ap.error("--query-batch must be positive")
    if not str(Path(args.out).resolve()).startswith(str(Path.home().resolve()) + os.sep):
        ap.error("--out must be inside your home directory")
    # Prevent SQLite spill outside home. Must be set before SQLite creates temp files.
    os.environ["TMPDIR"] = str(Path(args.out).resolve().parent)
    process_country(args, args.country, read_ground_truth(args.gt) if args.gt else None)


if __name__ == "__main__":
    main()
