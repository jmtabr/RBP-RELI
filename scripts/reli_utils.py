"""Shared utility module for RBP-RELI pipeline scripts.

Provides common constants (chromosome sets, region/direction lists) and core
functions for building per-base cumulative-position null models from BED files.
Used by build_null_model.py, extract_inputs_deseq2.py, extract_inputs_database.py,
and the unified pipeline.
"""

import os
import re


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STANDARD_CHROMS = {f"chr{i}" for i in range(1, 23)} | {"chrX", "chrY"}

SPLICING_REGIONS = ["AltEX", "DNintr", "UPintr", "merged"]
SPLICING_DIRECTIONS = ["SKIP", "INCL"]

DESEQ2_REGIONS = ["5UTR", "CDS", "intron", "3UTR"]
DESEQ2_DIRECTIONS = ["UP", "DOWN"]


# ---------------------------------------------------------------------------
# Core functions (null model construction)
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
# Convenience helpers
# ---------------------------------------------------------------------------

def write_dummy_dbsnp(output_dir):
    """Write empty dummy_dbsnp file required by C++ RELI."""
    path = os.path.join(output_dir, "dummy_dbsnp")
    with open(path, "w") as fh:
        pass  # empty file
    return path


def build_null_model_map(regions):
    """Build region_suffix -> Null_Model_BG_suffix mapping."""
    return {r: f"Null_Model_BG_{r}" for r in regions}
