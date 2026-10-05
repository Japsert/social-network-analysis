#!/usr/bin/env python3
"""
Count tag occurrences across the SocialBM0311 Delicious bookmark dataset
and print all tags sorted by how often they occur (most frequent first).

Each line of the dataset looks like:
    url_md5    user_id    url    unix_timestamp    tag1    tag2    tag3 ...
(tab-separated, with a variable number of trailing tags per line)

The script never loads the whole file into memory - it streams through it
line by line, and (for plain-text files) splits the work into byte-aligned
chunks that are processed in parallel across CPU cores.

Usage:
    python count_tags.py dataset.txt
    python count_tags.py dataset_part1.txt dataset_part2.txt.gz
    python count_tags.py dataset.txt -j 8 -n 100 -o top_tags.tsv
"""
import argparse
import gzip
import os
import sys
from collections import Counter
from multiprocessing import Pool, cpu_count

NUM_FIXED_FIELDS = 4  # url_md5, user_id, url, unix_timestamp -- rest is tags


def _count_plain_chunk(path_start_end):
    """Count tags in a byte-aligned chunk of a plain-text file."""
    path, start, end = path_start_end
    counts = Counter()
    with open(path, 'rb') as f:
        f.seek(start)
        if start != 0:
            f.readline()  # discard partial line; the previous chunk owns it
        while f.tell() < end:
            line = f.readline()
            if not line:
                break
            fields = line.rstrip(b'\n').split(b'\t')
            if len(fields) > NUM_FIXED_FIELDS:
                for tag in fields[NUM_FIXED_FIELDS:]:
                    if tag:
                        counts[tag] += 1
    return counts


def _count_whole_file(path):
    """Count tags in an entire file, read sequentially (used for .gz files,
    since gzip doesn't support cheap random access for byte-chunking)."""
    counts = Counter()
    opener = gzip.open if path.endswith('.gz') else open
    with opener(path, 'rb') as f:
        for line in f:
            fields = line.rstrip(b'\n').split(b'\t')
            if len(fields) > NUM_FIXED_FIELDS:
                for tag in fields[NUM_FIXED_FIELDS:]:
                    if tag:
                        counts[tag] += 1
    return counts


def build_jobs(files, jobs):
    """Turn the input files into a list of work units for the process pool."""
    units = []
    for path in files:
        if path.endswith('.gz'):
            units.append(('whole', path))
        else:
            size = os.path.getsize(path)
            # roughly one chunk per job, but don't over-split small files
            n_chunks = max(1, min(jobs, max(1, size // (64 * 1024 * 1024))))
            chunk_size = size // n_chunks
            start = 0
            for i in range(n_chunks):
                end = size if i == n_chunks - 1 else start + chunk_size
                units.append(('plain', (path, start, end)))
                start = end
    return units


def _run_job(unit):
    kind, payload = unit
    if kind == 'whole':
        return _count_whole_file(payload)
    return _count_plain_chunk(payload)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('files', nargs='+', help='Dataset file(s), plain text or .gz')
    parser.add_argument('-j', '--jobs', type=int, default=cpu_count(),
                         help='Parallel worker processes (default: number of CPU cores)')
    parser.add_argument('-o', '--output', default=None,
                         help='Write TSV output here instead of stdout')
    parser.add_argument('-n', '--top', type=int, default=None,
                         help='Only print the N most common tags')
    args = parser.parse_args()

    jobs = build_jobs(args.files, args.jobs)

    total = Counter()
    with Pool(processes=max(1, min(args.jobs, len(jobs)))) as pool:
        for i, partial in enumerate(pool.imap_unordered(_run_job, jobs), 1):
            total.update(partial)
            print(f"processed chunk {i}/{len(jobs)} - {len(total):,} unique tags so far",
                  file=sys.stderr)

    out = open(args.output, 'w', encoding='utf-8') if args.output else sys.stdout
    try:
        for tag, count in total.most_common(args.top):
            out.write(f"{count}\t{tag.decode('utf-8', errors='replace')}\n")
    finally:
        if out is not sys.stdout:
            out.close()


if __name__ == '__main__':
    main()
