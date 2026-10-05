#!/usr/bin/env python3
"""
Filter the raw SocialBM0311 bookmark dataset down to only the bookmarks
that have at least one tag in a given list (e.g. vegan / plant-based tags).

Each input line looks like:
    url_md5    user_id    url    unix_timestamp    tag1    tag2    tag3 ...
(tab-separated, variable number of trailing tags)

A line is kept if ANY of its tags exactly matches a tag in the tags file
(one tag per line). Matching lines are written out unchanged, in whatever
order the parallel workers finish (order is not preserved).

Usage:
    python filter_bookmarks_by_tag.py dataset.txt tags.txt 6_filtered_rows.tsv
    python filter_bookmarks_by_tag.py part1.txt part2.txt.gz tags.txt out.tsv -j 8
"""
import argparse
import gzip
import os
import sys
from multiprocessing import Pool, cpu_count

NUM_FIXED_FIELDS = 4  # url_md5, user_id, url, unix_timestamp -- rest is tags

# Populated in each worker process via the Pool initializer, so the tag set
# is built once per process rather than re-parsed/re-pickled per chunk.
_WANTED_TAGS = None


def _init_worker(tags_path):
    global _WANTED_TAGS
    with open(tags_path, "r", encoding="utf-8", errors="replace") as f:
        _WANTED_TAGS = {line.strip().encode("utf-8") for line in f if line.strip()}


def _filter_plain_chunk(path_start_end):
    path, start, end = path_start_end
    kept = []
    with open(path, "rb") as f:
        f.seek(start)
        if start != 0:
            f.readline()  # discard partial line; the previous chunk owns it
        while f.tell() < end:
            line = f.readline()
            if not line:
                break
            fields = line.rstrip(b"\n").split(b"\t")
            if len(fields) > NUM_FIXED_FIELDS and any(
                t in _WANTED_TAGS for t in fields[NUM_FIXED_FIELDS:]
            ):
                kept.append(line if line.endswith(b"\n") else line + b"\n")
    return kept


def _filter_whole_file(path):
    kept = []
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as f:
        for line in f:
            fields = line.rstrip(b"\n").split(b"\t")
            if len(fields) > NUM_FIXED_FIELDS and any(
                t in _WANTED_TAGS for t in fields[NUM_FIXED_FIELDS:]
            ):
                kept.append(line if line.endswith(b"\n") else line + b"\n")
    return kept


def build_jobs(files, jobs):
    units = []
    for path in files:
        if path.endswith(".gz"):
            units.append(("whole", path))
        else:
            size = os.path.getsize(path)
            n_chunks = max(1, min(jobs, max(1, size // (64 * 1024 * 1024))))
            chunk_size = size // n_chunks
            start = 0
            for i in range(n_chunks):
                end = size if i == n_chunks - 1 else start + chunk_size
                units.append(("plain", (path, start, end)))
                start = end
    return units


def _run_job(unit):
    kind, payload = unit
    if kind == "whole":
        return _filter_whole_file(payload)
    return _filter_plain_chunk(payload)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", help="dataset file(s) except the last arg, which is the tags file")
    parser.add_argument("-j", "--jobs", type=int, default=cpu_count(),
                         help="parallel worker processes (default: number of CPU cores)")
    args = parser.parse_args()

    if len(args.files) < 3:
        parser.error("need at least: <dataset file(s)...> <tags.txt> <output.tsv>")

    *dataset_files, tags_path, out_path = args.files

    jobs = build_jobs(dataset_files, args.jobs)

    total_matched = 0
    with Pool(processes=max(1, min(args.jobs, len(jobs))),
              initializer=_init_worker, initargs=(tags_path,)) as pool, \
         open(out_path, "wb") as fout:
        for i, kept_lines in enumerate(pool.imap(_run_job, jobs), 1):
            fout.writelines(kept_lines)
            total_matched += len(kept_lines)
            print(f"processed chunk {i}/{len(jobs)} - {total_matched:,} matching bookmarks so far",
                  file=sys.stderr)

    print(f"done: wrote {total_matched:,} matching bookmarks to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
