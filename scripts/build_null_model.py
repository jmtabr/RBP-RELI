#!/usr/bin/env python3
"""Build per-base expanded null models with cumulative genome positions.

Expands BG BED intervals into per-base cumulative genome coordinates for
RELI's permutation null model. Includes validation to ensure positions
span the full genome (not limited to a single chromosome).

Usage:
    python build_null_model.py --input-dir inputs_hg19/ --genome-build data/GenomeBuild/hg19.txt

Input:
    BG_*.bed files in --input-dir (BED4 format: chr, start, end, name)
    Genome build file (tab-separated: chrom, size)

Output:
    Null_Model_BG_{region} files (one position per line, sorted ascending)
    dummy_dbsnp (empty file required by RELI)

The core functions (load_chrom_offsets, expand_bed_to_cumulative_positions)
are designed to be importable by other scripts (e.g., extract_inputs_deseq2.py).
"""

import argparse
import os
import re
import sys


# ---------------------------------------------------------------------------
# Core functions (importable)
# ---------------------------------------------------------------------------

def load_chrom_offsets(genome_build_path):
    """Load chromosome cumulative offsets from a genome build file.

    The genome build file is tab-separated with columns: chrom, size.
    Cumulative offsets convert (chrom, position) to a single integer that
    preserves genomic order across the entire genome.

    Args:
        genome_build_path: Path to hg19.txt or similar genome build file.

    Returns:
        dict: {chrom_name: cumulative_offset}
    """
    offsets = {}
    cumsum = 0
    with open(genome_build_path) as fh:
        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                offsets[parts[0]] = cumsum
                cumsum += int(parts[1])
    return offsets


def expand_bed_to_cumulative_positions(bed_path, chrom_offsets):
    """Expand BED intervals to per-base cumulative genome positions.

    For each interval [start, end), computes chrom_offset + position for
    every base. Returns a sorted list of integers.

    Args:
        bed_path: Path to a BED file (at least 3 columns: chr, start, end).
        chrom_offsets: dict from load_chrom_offsets().

    Returns:
        list[int]: Sorted cumulative genome positions.
    """
    positions = []
    skipped_chroms = set()
    with open(bed_path, "r") as fh:
        for line in fh:
            fields = line.strip().split("\t")
            if len(fields) < 3:
                continue
            chrom = fields[0]
            start = int(fields[1])
            end = int(fields[2])
            if chrom not in chrom_offsets:
                skipped_chroms.add(chrom)
                continue
            offset = chrom_offsets[chrom]
            for pos in range(start, end):
                positions.append(offset + pos)
    if skipped_chroms:
        print(f"  WARNING: Skipped {len(skipped_chroms)} unknown chromosomes: "
              f"{', '.join(sorted(skipped_chroms)[:5])}")
    positions.sort()
    return positions


def write_null_model(positions, output_path):
    """Write null model file: line 1 is 0 (header), then sorted positions.

    Both C++ RELI and GPU-RELI skip line 1, so the leading 0 acts as a
    dummy header row.

    Args:
        positions: Sorted list of cumulative genome positions.
        output_path: Path to write the null model file.
    """
    with open(output_path, "w", newline="\n") as fh:
        fh.write("0\n")
        for pos in positions:
            fh.write(f"{pos}\n")


def validate_null_model(positions, region_name, chrom_offsets):
    """Validate that a null model has correct cumulative genome positions.

    Checks:
        1. Max value > 249,250,621 (chr1 size) -- confirms multi-chromosome.
        2. Positions span >= 5 chromosome ranges in cumsum space.
        3. Values are sorted ascending.

    Args:
        positions: Sorted list of cumulative genome positions.
        region_name: Name for error messages (e.g., "AltEX").
        chrom_offsets: dict from load_chrom_offsets().

    Returns:
        tuple: (is_valid: bool, summary: str)
    """
    CHR1_SIZE = 249_250_621
    errors = []

    if len(positions) == 0:
        return False, f"ERROR [{region_name}]: No positions generated"

    max_val = positions[-1]
    min_val = positions[0]

    # Check 1: multi-chromosome range
    if max_val <= CHR1_SIZE:
        errors.append(
            f"Max value {max_val:,} <= chr1 size {CHR1_SIZE:,} -- "
            f"positions may be missing cumulative chromosome offsets"
        )

    # Check 2: span at least 5 chromosome ranges
    # Build sorted boundary list from offsets
    boundaries = sorted(chrom_offsets.values())
    # Count how many chromosome boundaries fall within our position range
    n_chroms_spanned = 0
    for i, boundary in enumerate(boundaries):
        # A chromosome is "spanned" if any position falls in its range
        upper = boundaries[i + 1] if i + 1 < len(boundaries) else max_val + 1
        # Binary-search-like check: any position in [boundary, upper)?
        # Simpler: count chromosomes whose offset range overlaps [min_val, max_val]
        if boundary <= max_val and upper > min_val:
            n_chroms_spanned += 1

    if n_chroms_spanned < 5:
        errors.append(
            f"Positions span only {n_chroms_spanned} chromosome ranges "
            f"(expected >= 5)"
        )

    # Check 3: sorted ascending
    is_sorted = all(positions[i] <= positions[i + 1] for i in range(len(positions) - 1))
    if not is_sorted:
        errors.append("Positions are NOT sorted ascending")

    if errors:
        msg = f"VALIDATION FAILED [{region_name}]: " + "; ".join(errors)
        return False, msg

    summary = (
        f"VALIDATED: {len(positions):,} positions spanning "
        f"{n_chroms_spanned} chromosomes, max={max_val:,}"
    )
    return True, summary


# ---------------------------------------------------------------------------
# BG file discovery
# ---------------------------------------------------------------------------

def discover_bg_files(input_dir):
    """Find BG_*.bed files and extract region names.

    Recognizes patterns:
        BG_AltEX.bed       -> region "AltEX"
        BG_3UTR_hg19.bed   -> region "3UTR"
        BG_merged.bed       -> region "merged"

    Ignores:
        BG_*_unmapped.bed  (liftOver rejects, not real BG files)

    Args:
        input_dir: Directory to search.

    Returns:
        list[tuple]: [(region_name, bed_file_path), ...] sorted by region.
    """
    pattern = re.compile(r"^BG_(.+?)(?:_hg19)?\.bed$")
    results = []
    for fname in os.listdir(input_dir):
        if "_unmapped" in fname:
            continue
        m = pattern.match(fname)
        if m:
            region = m.group(1)
            results.append((region, os.path.join(input_dir, fname)))
    results.sort(key=lambda x: x[0])
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Build per-base expanded null models with cumulative genome positions.",
        epilog="Builds cumulative-position null models from BG BED files and a genome build file.",
    )
    p.add_argument(
        "--input-dir", required=True,
        help="Directory containing BG_*.bed files (hg19 coordinates)",
    )
    p.add_argument(
        "--genome-build", required=True,
        help="Path to genome build file (e.g., hg19.txt) with chrom sizes",
    )
    p.add_argument(
        "--output-dir", default=None,
        help="Output directory for null models (default: same as --input-dir)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir or args.input_dir

    # Load chromosome offsets
    print(f"Loading genome build: {args.genome_build}")
    chrom_offsets = load_chrom_offsets(args.genome_build)
    print(f"  {len(chrom_offsets)} chromosomes loaded")

    # Discover BG BED files
    bg_files = discover_bg_files(args.input_dir)
    if not bg_files:
        print(f"ERROR: No BG_*.bed files found in {args.input_dir}")
        sys.exit(1)
    print(f"\nFound {len(bg_files)} BG files:")
    for region, path in bg_files:
        print(f"  {os.path.basename(path)} -> region '{region}'")

    # Process each BG file
    os.makedirs(output_dir, exist_ok=True)
    results = []
    any_failed = False

    print(f"\n{'='*60}")
    print("Building per-base null models (cumulative positions)")
    print(f"{'='*60}")

    for region, bed_path in bg_files:
        null_model_name = f"Null_Model_BG_{region}"
        null_model_path = os.path.join(output_dir, null_model_name)

        print(f"\n--- {region} ---")

        # Count input intervals
        with open(bed_path) as fh:
            n_intervals = sum(1 for line in fh if line.strip())
        print(f"  Input: {os.path.basename(bed_path)} ({n_intervals:,} intervals)")

        # Expand to per-base cumulative positions
        positions = expand_bed_to_cumulative_positions(bed_path, chrom_offsets)
        print(f"  Expanded to {len(positions):,} per-base positions")

        # Write null model
        write_null_model(positions, null_model_path)
        file_size = os.path.getsize(null_model_path)

        # Verify file starts with 0 on line 1
        with open(null_model_path) as fh:
            first_line = fh.readline().strip()
        if first_line != "0":
            print(f"  ERROR: First line is '{first_line}', expected '0'")
            any_failed = True
            results.append((region, n_intervals, len(positions), file_size, 0, False))
            continue

        # Validate cumulative positions
        is_valid, summary = validate_null_model(positions, region, chrom_offsets)
        print(f"  {summary}")

        if not is_valid:
            any_failed = True

        max_val = positions[-1] if positions else 0
        results.append((region, n_intervals, len(positions), file_size, max_val, is_valid))

    # Create dummy_dbsnp
    dummy_path = os.path.join(output_dir, "dummy_dbsnp")
    with open(dummy_path, "w") as fh:
        pass
    print(f"\nCreated {dummy_path}")

    # Summary table
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    header = f"{'Region':<12} {'BG Events':>10} {'Positions':>12} {'File Size':>12} {'Max Value':>16} {'Status':<10}"
    print(header)
    print("-" * len(header))
    for region, n_events, n_pos, fsize, maxv, valid in results:
        size_str = _format_size(fsize)
        status = "OK" if valid else "FAILED"
        print(f"{region:<12} {n_events:>10,} {n_pos:>12,} {size_str:>12} {maxv:>16,} {status:<10}")

    if any_failed:
        print("\nERROR: One or more null models failed validation. See above.")
        sys.exit(1)
    else:
        print(f"\nAll {len(results)} null models validated successfully.")


def _format_size(nbytes):
    """Format byte count as human-readable string."""
    if nbytes >= 1_073_741_824:
        return f"{nbytes / 1_073_741_824:.1f} GB"
    if nbytes >= 1_048_576:
        return f"{nbytes / 1_048_576:.1f} MB"
    if nbytes >= 1024:
        return f"{nbytes / 1024:.1f} KB"
    return f"{nbytes} B"


if __name__ == "__main__":
    main()
