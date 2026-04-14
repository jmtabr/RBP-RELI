"""
reli_turbo.cli -- Command-line interface for GPU-accelerated RELI.

Provides the ``reli-turbo`` console entry point.  Accepts flags that
mirror the C++ RELI binary where applicable, plus GPU-specific options.

Usage examples:

    # Full pipeline (all 8 queries)
    reli-turbo \\
        --input-dir ./inputs \\
        --output-dir ./output \\
        --index ./CLIP-seq/CLIPseq.index \\
        --data ./CLIP-seq \\
        --build ./GenomeBuild/hg19.txt

    # Single query mode (C++ RELI flag compatibility)
    reli-turbo \\
        -snp SKIP_UPintr.snp \\
        -null Null_Model_BG_UPintr \\
        -index CLIPseq.index \\
        -data ./CLIP-seq \\
        -build hg19.txt \\
        -out ./output/SKIP_UPintr \\
        -rep 2000 \\
        -corr 473 \\
        -phenotype SKIP_UPintr
"""

from __future__ import annotations

import argparse
import sys

from .reli import (
    N_PERMS_DEFAULT,
    run_full_pipeline,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="reli-turbo",
        description=(
            "GPU-accelerated RELI permutation testing for RBP enrichment "
            "analysis.  Reimplements RELI v0.90 (Harley et al. 2018) using "
            "CuPy/CUDA for 20-40x speedup."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Full pipeline (all 8 RBP-RELI queries)\n"
            "  reli-turbo --input-dir inputs/ --output-dir output/ \\\n"
            "    --index CLIP-seq/CLIPseq.index --data CLIP-seq/ \\\n"
            "    --build GenomeBuild/hg19.txt\n"
            "\n"
            "  # Single-query mode (C++ RELI compatible flags)\n"
            "  reli-turbo -snp SKIP_UPintr.snp -null Null_Model_BG_UPintr \\\n"
            "    -index CLIPseq.index -data CLIP-seq/ -build hg19.txt \\\n"
            "    -out output/SKIP_UPintr -rep 2000 -corr 473 \\\n"
            "    -phenotype SKIP_UPintr\n"
        ),
    )

    # ---- Full pipeline mode ----
    pipeline = parser.add_argument_group("Full pipeline mode")
    pipeline.add_argument(
        "--input-dir",
        help="Directory containing .snp query files and null model files.",
    )
    pipeline.add_argument(
        "--output-dir",
        help="Output directory for results.",
    )
    pipeline.add_argument(
        "--queries",
        nargs="+",
        help=(
            "Specific query names to run (default: all 8). "
            "E.g., SKIP_UPintr SKIP_AltEX"
        ),
    )

    # ---- C++ RELI compatible flags (single-query mode) ----
    compat = parser.add_argument_group(
        "C++ RELI compatible flags (single-query mode)"
    )
    compat.add_argument(
        "-snp",
        help="Path to query .snp file (BED4 format).",
    )
    compat.add_argument(
        "-null",
        help="Path to null model file.",
    )
    compat.add_argument(
        "-out",
        help="Output directory for single-query mode.",
    )
    compat.add_argument(
        "-phenotype",
        default=".",
        help="Phenotype label for output (default: '.').",
    )
    compat.add_argument(
        "-ancestry",
        default=".",
        help="Ancestry label for output (default: '.').",
    )

    # ---- Shared flags ----
    shared = parser.add_argument_group("Shared flags")
    shared.add_argument(
        "-index", "--index",
        dest="index",
        help="Path to CLIPseq.index file.",
    )
    shared.add_argument(
        "-data", "--data",
        dest="data",
        help="Directory containing target peak BED files.",
    )
    shared.add_argument(
        "-build", "--build",
        dest="build",
        help="Path to genome build file (e.g., hg19.txt).",
    )
    shared.add_argument(
        "-rep", "--reps",
        dest="reps",
        type=int,
        default=N_PERMS_DEFAULT,
        help=f"Number of permutations (default: {N_PERMS_DEFAULT}).",
    )
    shared.add_argument(
        "-corr", "--corr",
        dest="corr",
        type=int,
        default=0,
        help="Bonferroni correction multiplier (0 = auto from index).",
    )

    # ---- GPU-specific flags ----
    gpu = parser.add_argument_group("GPU options")
    gpu.add_argument(
        "--device",
        type=int,
        default=0,
        help="CUDA device ID (default: 0).",
    )
    gpu.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    gpu.add_argument(
        "--per-target-files",
        action="store_true",
        default=False,
        help="Write per-target .stats/.overlaps/.rsids files (legacy C++ format). "
             "Default: write 3 consolidated TSVs per query for faster I/O.",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    """Main entry point for the reli-turbo CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Validate: need at least index + data + build
    if not args.index or not args.data or not args.build:
        parser.error(
            "Required flags: -index/--index, -data/--data, -build/--build"
        )

    # Determine mode: single-query or full pipeline
    if args.snp and args.null:
        # ---- Single-query mode ----
        _run_single_query_mode(args)
    elif args.input_dir:
        # ---- Full pipeline mode ----
        run_full_pipeline(
            input_dir=args.input_dir,
            output_dir=args.output_dir or "output",
            clip_index=args.index,
            peaks_dir=args.data,
            genome_build=args.build,
            n_reps=args.reps,
            corr=args.corr,
            seed=args.seed,
            device=args.device,
            queries=args.queries,
        )
    else:
        parser.error(
            "Either provide --input-dir (full pipeline mode) or "
            "-snp + -null (single-query mode)."
        )


def _run_single_query_mode(args: argparse.Namespace) -> None:
    """Run RELI in single-query mode (C++ RELI compatible)."""
    from . import io as reli_io
    from .reli import (
        collect_observed_rsids,
        collect_observed_rsids_gpu_from_flags,
        run_batch,
        write_results,
        write_results_consolidated,
    )

    print("=" * 60)
    print("GPU-RELI: Single-query mode")
    print("=" * 60)

    # Load genome build
    chrom_cumsum, chrom_names = reli_io.load_genome_build(args.build)
    chrom_to_idx = reli_io.make_chrom_to_idx(chrom_names)
    n_chroms = len(chrom_names)
    print(f"  {n_chroms} chromosomes")

    # Load query
    query = reli_io.load_snp_file(args.snp, chrom_to_idx)
    n_loci = len(query["chr_idx"])
    print(f"  Query: {n_loci} loci from {args.snp}")

    # Load null model
    null_model = reli_io.load_null_model(args.null)
    print(f"  Null model: {len(null_model):,} positions from {args.null}")

    # Load all targets
    targets = reli_io.load_all_targets(
        args.data, args.index, chrom_to_idx, n_chroms
    )
    n_targets = len(targets["target_labels"])
    print(f"  Targets: {n_targets}")

    corr_multiplier = args.corr if args.corr > 0 else n_targets

    # Run
    results = run_batch(
        query, null_model, chrom_cumsum, targets,
        n_reps=args.reps, n_targets_corr=corr_multiplier,
        seed=args.seed, device=args.device,
    )

    # Collect RSIDs (use GPU flags if available, fall back to CPU)
    if "rsid_flags" in results:
        rsids = collect_observed_rsids_gpu_from_flags(query, results["rsid_flags"])
    else:
        rsids = collect_observed_rsids(query, targets, results["observed"])

    # Write
    out_dir = args.out or "output"
    null_name = args.null.split("/")[-1].split("\\")[-1]
    if getattr(args, "per_target_files", False):
        write_results(
            out_dir, args.phenotype, targets, results, rsids,
            corr_multiplier, n_loci, null_name,
        )
        print(f"\n  Results written to {out_dir}/ (per-target files)")
    else:
        write_results_consolidated(
            out_dir, args.phenotype, targets, results, rsids,
            corr_multiplier, n_loci, null_name,
        )
        print(f"\n  Results written to {out_dir}/")

    n_sig = int((results["corr_pval"] < 0.05).sum())
    print(f"  Bonferroni-significant targets: {n_sig}")


# Allow running as: python -m reli_turbo
if __name__ == "__main__":
    main()
