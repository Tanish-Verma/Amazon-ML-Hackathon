"""
I/O primitives for the entity-resolution pipeline.

Everything here is written against two facts about the deployment environment:

  * The pipeline runs inside one user's home directory on a shared lab machine.
    Disk is treated as limited and this process NEVER deletes anything -- it
    reports pressure and lets a human act on it (see ``assert_free_space``).
  * The sources are large (up to 5.3M rows / 500 MB each), so every reader here
    either streams or uses Arrow's columnar reader. Nothing builds an
    intermediate list-of-dicts over a whole source.

The challenge files are tab-separated with NO quoting -- addresses contain
commas and apostrophes, and a stray double quote must be read literally rather
than treated as a quote character. Every reader below therefore disables
quote handling explicitly; using a default CSV reader here silently corrupts
rows that contain a quote character.
"""

from __future__ import annotations

import csv
import os
import shutil
import sys
from typing import Iterator, Sequence

import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

# The exact schema the problem statement specifies for every source file.
SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
GROUND_TRUTH_COLUMNS = ["source1_entity_id", "matched_entity_ids"]

DELIM = "\t"


# --------------------------------------------------------------------------
# Disk safety
# --------------------------------------------------------------------------

def free_bytes(path: str) -> int:
    """Bytes free on the filesystem holding ``path``."""
    return shutil.disk_usage(path).free


def assert_free_space(path: str, need_gb: float) -> None:
    """Fail loudly *before* a bulk write rather than filling a shared volume.

    This project never deletes files to make room -- that decision belongs to
    the machine's owner -- so the only safe behaviour on disk pressure is to
    stop and report.
    """
    have = free_bytes(path) / 1e9
    if have < need_gb:
        raise RuntimeError(
            f"Refusing to write: {path} has {have:.1f} GB free but this step "
            f"needs ~{need_gb:.1f} GB. Nothing has been deleted automatically. "
            f"Free space manually and re-run."
        )


# --------------------------------------------------------------------------
# Streaming readers
# --------------------------------------------------------------------------

def stream_tsv(path: str, columns: Sequence[str] | None = None) -> Iterator[dict]:
    """Yield one dict per data row, streaming, with no quote interpretation.

    ``csv.reader`` with ``QUOTE_NONE`` is used rather than ``DictReader`` so the
    header can be validated against ``columns`` before any row is produced.
    """
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter=DELIM, quoting=csv.QUOTE_NONE)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"{path} is empty")
        if columns is not None and header != list(columns):
            raise ValueError(f"{path}: header {header} != expected {list(columns)}")
        width = len(header)
        for row in reader:
            # Guard against ragged rows: a short row means an embedded newline
            # or a lost tab, and silently zipping it would mis-assign fields.
            if len(row) != width:
                raise ValueError(f"{path}: ragged row with {len(row)} fields, expected {width}: {row[:2]}")
            yield dict(zip(header, row))


def read_source_table(path: str) -> pa.Table:
    """Read a whole source file into an Arrow table (columnar, low overhead).

    Arrow keeps strings in contiguous buffers rather than as Python objects, so
    a 5M-row source costs a few hundred MB here versus several GB via pandas'
    object dtype. Everything is read as string: entity IDs are prefixed tokens,
    not integers, and addresses must not be type-inferred.
    """
    table = pacsv.read_csv(
        path,
        parse_options=pacsv.ParseOptions(delimiter=DELIM, quote_char=False, newlines_in_values=False),
        convert_options=pacsv.ConvertOptions(
            column_types={c: pa.string() for c in SOURCE_COLUMNS},
            strings_can_be_null=False,
        ),
        read_options=pacsv.ReadOptions(use_threads=True),
    )
    if table.column_names != SOURCE_COLUMNS:
        raise ValueError(f"{path}: columns {table.column_names} != {SOURCE_COLUMNS}")
    return table


def read_ground_truth(path: str) -> dict[str, set[str]]:
    """Load ``train_ground_truth.tsv`` into ``{source1_id: set(matched_ids)}``.

    An empty ``matched_entity_ids`` field is a genuine singleton and maps to an
    empty set -- distinct from a missing key, which would mean the entity was
    absent from the file entirely.
    """
    out: dict[str, set[str]] = {}
    for row in stream_tsv(path, GROUND_TRUTH_COLUMNS):
        raw = row["matched_entity_ids"]
        out[row["source1_entity_id"]] = set(raw.split(",")) if raw else set()
    return out


# --------------------------------------------------------------------------
# Columnar store
# --------------------------------------------------------------------------

def build_columnar_store(tsv_path: str, out_dir: str, split: str, source: str) -> list[str]:
    """Convert one source TSV to Parquet, partitioned by country.

    Partitioning by ``country`` is safe and generalising: it partitions on
    whatever label is present in the data rather than on a fixed set, so an
    unseen country (France, in the test set) creates its own partition with no
    code change. Ground truth confirms matches never cross a country boundary,
    so this partition is also a correctness-preserving restriction of the
    candidate space.
    """
    os.makedirs(out_dir, exist_ok=True)
    table = read_source_table(tsv_path)
    countries = pa_unique(table, "country")
    written = []
    for country in countries:
        mask = pa.compute.equal(table.column("country"), pa.scalar(country))
        part = table.filter(mask)
        # Country labels go into a path, so sanitise rather than trusting them.
        safe = "".join(ch if ch.isalnum() else "_" for ch in country)
        path = os.path.join(out_dir, f"{split}_{source}_{safe}.parquet")
        pq.write_table(part, path, compression="zstd", compression_level=3)
        written.append(path)
    return written


def pa_unique(table: pa.Table, column: str) -> list[str]:
    """Sorted unique values of a string column."""
    return sorted(pa.compute.unique(table.column(column)).to_pylist())


# --------------------------------------------------------------------------
# Submission writers
# --------------------------------------------------------------------------

def write_id_list_tsv(path: str, header: Sequence[str], rows: Iterator[tuple[str, Sequence[str]]]) -> int:
    """Write a two-column submission file (matching_results / candidate_pairs).

    Both output files share this shape: an S1 id, then a comma-separated ID
    list that is empty for singletons. Written with an explicit tab delimiter
    and no quoting, because the scorer splits on tabs and commas literally.
    """
    n = 0
    tmp = path + ".partial"
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        fh.write(DELIM.join(header) + "\n")
        for s1_id, ids in rows:
            fh.write(f"{s1_id}{DELIM}{','.join(ids)}\n")
            n += 1
    os.replace(tmp, path)  # atomic: never leave a half-written submission
    return n
