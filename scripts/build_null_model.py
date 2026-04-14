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
import sys

from reli_utils import (
    load_chrom_offsets, expand_bed_to_cumulative_positions,
    write_null_model, validate_null_model, discover_bg_files,
)


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
