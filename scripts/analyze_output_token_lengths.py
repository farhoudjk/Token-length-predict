#!/usr/bin/env python3
"""Analyze output token lengths in an inference CSV.

Reads a CSV (default `out/out_v2.csv`) and prints counts and percentages
for these ranges:
 - 1 <= tokens < 9      : error
 - 10 <= tokens <= 500  : short answer
 - 501 <= tokens <= 2000: medium answer
 - 2001 <= tokens <= 4096: exhaustive answers

Any rows with missing or out-of-range token counts are reported under
`missing` or `other` respectively.
"""
import argparse
import csv
import os
import sys
from collections import Counter


def parse_int(v):
    if v is None:
        return None
    v = v.strip()
    if v == "":
        return None
    try:
        # allow floats in CSV but convert to int
        return int(float(v))
    except Exception:
        return None


def analyze(path):
    if not os.path.exists(path):
        print(f"Error: file not found: {path}")
        return 2

    total = 0
    counts = Counter()

    # read as binary and strip NULs to avoid _csv.Error: line contains NUL
    import io
    with open(path, 'rb') as fb:
        data = fb.read().replace(b'\x00', b' ')
    text = data.decode('utf-8', errors='replace')
    reader = csv.DictReader(io.StringIO(text))
    # guess common field names
    common_keys = [
        'output_tokens', 'output_token', 'output_token_count', 'output_tokens_count'
    ]

    for row in reader:
        total += 1
        tok = None
        for k in common_keys:
            if k in row:
                tok = parse_int(row[k])
                break
        # fallback: try any key that contains 'output' and 'token'
        if tok is None:
            for k in row:
                lk = k.lower()
                if 'output' in lk and 'token' in lk:
                    tok = parse_int(row[k])
                    break

        if tok is None:
            counts['missing'] += 1
            continue

        if 1 <= tok < 9:
            counts['error'] += 1
        elif 10 <= tok <= 500:
            counts['short'] += 1
        elif 501 <= tok <= 2000:
            counts['medium'] += 1
        elif 2001 <= tok <= 4096:
            counts['exhaustive'] += 1
        else:
            counts['other'] += 1

    if total == 0:
        print("No rows found in the CSV.")
        return 0

    def pct(n):
        return 100.0 * n / total

    print(f"Analyzed: {path}")
    print(f"Total rows: {total}")
    print("")
    print(f"1 <= tokens < 9 (error): {counts['error']} ({pct(counts['error']):.2f}%)")
    print(f"10 <= tokens <= 500 (short answer): {counts['short']} ({pct(counts['short']):.2f}%)")
    print(f"501 <= tokens <= 2000 (medium answer): {counts['medium']} ({pct(counts['medium']):.2f}%)")
    print(f"2001 <= tokens <= 4096 (exhaustive answer): {counts['exhaustive']} ({pct(counts['exhaustive']):.2f}%)")
    print("")
    print(f"missing token count: {counts['missing']} ({pct(counts['missing']):.2f}%)")
    print(f"other (out-of-range / unexpected): {counts['other']} ({pct(counts['other']):.2f}%)")

    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', '-c', default='out/out_v2.csv', help='Path to output CSV')
    args = parser.parse_args()
    sys.exit(analyze(args.csv))


if __name__ == '__main__':
    main()
