#!/usr/bin/env python3
"""Extract RELI input files from user-provided query and background BED files.

Database mode: test CLIP-seq enrichment at arbitrary genomic regions without
requiring differential splicing or expression analysis. Provide your own
query regions (e.g., from a database, custom annotations, or any BED file)
and matched background regions.

Input BED format (3 or 4 columns):
    chr  start  end  [name]

Output:
    {prefix}.snp          RELI query file (one locus per line or tiled)
    BG_{prefix}.bed       Background BED for null model construction
    {prefix}_summary.txt  Run summary with event counts

The background BED is passed to liftover_and_build_null.sh, which expands
it to per-base positions for RELI's permutation null model.

Examples:
    # Simple: test custom peaks against CLIP library
    python extract_inputs_database.py \\
        --query-bed my_peaks.bed --bg-bed control_peaks.bed \\
        --output-dir inputs/ --prefix MY_QUERY

    # Tile large regions (e.g., 3'UTRs) into 100bp-spaced loci
    python extract_inputs_database.py \\
        --query-bed are_3utrs.bed --bg-bed non_are_3utrs.bed \\
        --output-dir inputs/ --prefix ARE_3UTR --tile 100
"""

import argparse
import os
import random
import sys

STANDARD_CHROMS = {f"chr{i}" for i in range(1, 23)} | {"chrX", "chrY"}


def read_bed(path):
    """Read a BED file (3 or 4 columns). Returns list of (chrom, start, end, name)."""
    regions = []
    with open(path) as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("track") or line.startswith("browser"):
                continue
            fields = line.split("\t")
            if len(fields) < 3:
                print(f"WARNING: skipping line {lineno} ({len(fields)} fields): {line[:80]}", file=sys.stderr)
                continue
            chrom = fields[0]
            try:
                start = int(fields[1])
                end = int(fields[2])
            except ValueError:
                print(f"WARNING: skipping line {lineno} (non-integer coords): {line[:80]}", file=sys.stderr)
                continue
            name = fields[3] if len(fields) > 3 else f"locus_{lineno}"
            if chrom in STANDARD_CHROMS and end > start:
                regions.append((chrom, start, end, name))
    return regions


def tile_regions(regions, spacing):
    """Tile each region into multiple loci spaced `spacing` bp apart.

    For regions smaller than `spacing`, a single midpoint locus is emitted.
    This ensures CLIP peaks have representative coverage across large regions
    (e.g., 3'UTRs or gene bodies) rather than testing a single point.
    """
    tiled = []
    for chrom, start, end, name in regions:
        length = end - start
        if length <= spacing:
            mid = (start + end) // 2
            tiled.append((chrom, mid, mid + 1, name))
        else:
            for pos in range(start, end, spacing):
                tiled.append((chrom, pos, pos + 1, name))
    return tiled


def write_snp(regions, path):
    """Write RELI .snp file. Format: chr<TAB>start<TAB>end<TAB>name"""
    with open(path, "w", newline="\n") as fh:
        for chrom, start, end, name in regions:
            fh.write(f"{chrom}\t{start}\t{end}\t{name}\n")
    return len(regions)


def write_bed(regions, path):
    """Write BED3 file for null model. Format: chr<TAB>start<TAB>end"""
    with open(path, "w", newline="\n") as fh:
        for chrom, start, end, _name in regions:
            fh.write(f"{chrom}\t{start}\t{end}\n")
    return len(regions)


def parse_args():
    p = argparse.ArgumentParser(
        description="Extract RELI inputs from query and background BED files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--query-bed", required=True,
                   help="BED file with query regions (regions of interest)")
    p.add_argument("--bg-bed", required=True,
                   help="BED file with background regions (matched null set)")
    p.add_argument("--output-dir", required=True,
                   help="Output directory")
    p.add_argument("--prefix", default="DB",
                   help="Output file prefix (default: DB)")
    p.add_argument("--tile", type=int, default=None,
                   help="Tile query regions at this spacing in bp "
                        "(e.g., 100 for large regions like 3'UTRs). "
                        "If omitted, each region becomes one query locus.")
    p.add_argument("--max-bg", type=int, default=None,
                   help="Max background regions to keep (subsample if exceeded)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for background subsampling (default: 42)")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load ---
    print(f"Loading query: {args.query_bed}")
    query = read_bed(args.query_bed)
    print(f"  {len(query)} regions on standard chromosomes")

    print(f"Loading background: {args.bg_bed}")
    bg = read_bed(args.bg_bed)
    print(f"  {len(bg)} regions on standard chromosomes")

    if not query:
        print("ERROR: no valid query regions found.", file=sys.stderr)
        sys.exit(1)
    if not bg:
        print("ERROR: no valid background regions found.", file=sys.stderr)
        sys.exit(1)

    # --- Tile query (optional) ---
    if args.tile:
        print(f"Tiling query regions at {args.tile}bp spacing")
        query_loci = tile_regions(query, args.tile)
        print(f"  {len(query)} regions -> {len(query_loci)} tiled loci")
    else:
        query_loci = query

    # --- Subsample background (optional) ---
    if args.max_bg and len(bg) > args.max_bg:
        print(f"Subsampling background: {len(bg)} -> {args.max_bg}")
        bg = random.sample(bg, args.max_bg)

    # --- Write outputs ---
    snp_path = os.path.join(args.output_dir, f"{args.prefix}.snp")
    bg_path = os.path.join(args.output_dir, f"BG_{args.prefix}.bed")
    summary_path = os.path.join(args.output_dir, f"{args.prefix}_summary.txt")

    n_snp = write_snp(query_loci, snp_path)
    n_bg = write_bed(bg, bg_path)

    # --- Query size stats ---
    query_sizes = [end - start for _, start, end, _ in query]
    bg_sizes = [end - start for _, start, end, _ in bg]

    # --- Summary ---
    with open(summary_path, "w", newline="\n") as fh:
        lines = [
            f"Query BED:      {args.query_bed}",
            f"Background BED: {args.bg_bed}",
            f"Prefix:         {args.prefix}",
            f"Tiling:         {args.tile or 'off'}",
            f"",
            f"Query regions:  {len(query)}",
            f"Query loci:     {n_snp}" + (f" (tiled at {args.tile}bp)" if args.tile else ""),
            f"Query size:     median={sorted(query_sizes)[len(query_sizes)//2]}, "
            f"range=[{min(query_sizes)}, {max(query_sizes)}]",
            f"",
            f"BG regions:     {n_bg}",
            f"BG size:        median={sorted(bg_sizes)[len(bg_sizes)//2]}, "
            f"range=[{min(bg_sizes)}, {max(bg_sizes)}]",
            f"",
            f"Output:",
            f"  {snp_path}",
            f"  {bg_path}",
        ]
        for line in lines:
            fh.write(line + "\n")
            print(line)

    print(f"\nDone. Next: liftover_and_build_null.sh {args.output_dir}")


if __name__ == "__main__":
    main()
