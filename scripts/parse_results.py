#!/usr/bin/env python3
"""Parse reli-turbo output: collect per-target .RELI.stats files, sort by Z-score, write combined TSVs.

Supports two output layouts:
  - Turbo layout (default): output_dir/{query}/{target}.RELI.stats
  - Legacy layout (fallback): output_dir/{query}_results.tsv  (requires --clip-index)

RBP functional categories are appended
automatically when --categories is provided or when the default category
file is found at data/rbp_categories_complete.tsv relative to this script.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Default location for the RBP category lookup (relative to this script)
_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_CATEGORIES = _SCRIPT_DIR.parent / "data" / "rbp_categories_complete.tsv"


def parse_args():
    p = argparse.ArgumentParser(
        description="Parse reli-turbo (or legacy RELI) results into combined TSVs."
    )
    p.add_argument("output_dir", help="Directory containing query subdirectories (turbo) or *_results.tsv files (legacy)")
    p.add_argument("--queries", nargs="+", default=None,
                   help="Limit to these query names (default: auto-discover all)")
    p.add_argument("--clip-index", default=None,
                   help="Path to CLIPseq.index (required only for legacy _results.tsv format)")
    p.add_argument("--categories", default=None,
                   help="Path to RBP category TSV (default: data/rbp_categories_complete.tsv)")
    p.add_argument("--no-categories", action="store_true",
                   help="Disable category annotation even if the file exists")
    p.add_argument("--min-ratio", type=float, default=None,
                   help="Minimum overlap ratio filter (e.g., 0.1). "
                        "When set, an additional *_filtered.tsv is written alongside the full results.")
    return p.parse_args()


def load_categories(path: Path | str | None) -> dict[str, str] | None:
    """Load RBP -> Category mapping from a TSV file.

    Returns None if no category file is found or --no-categories is set.
    """
    if path is not None:
        cat_path = Path(path)
    elif _DEFAULT_CATEGORIES.exists():
        cat_path = _DEFAULT_CATEGORIES
    else:
        return None

    if not cat_path.exists():
        print(f"WARNING: Category file not found: {cat_path}", file=sys.stderr)
        return None

    df = pd.read_csv(cat_path, sep="\t", usecols=["RBP", "Category"])
    return dict(zip(df["RBP"], df["Category"]))


def discover_queries(output_dir: Path) -> list[str]:
    """Return sorted list of subdirectory names that contain .RELI.stats files."""
    queries = []
    for d in sorted(output_dir.iterdir()):
        if d.is_dir() and any(d.glob("*.RELI.stats")):
            queries.append(d.name)
    return queries


def load_turbo_query(query_dir: Path) -> pd.DataFrame:
    """Read all .RELI.stats files in a query directory and concatenate."""
    frames = []
    for stats_file in sorted(query_dir.glob("*.RELI.stats")):
        try:
            df = pd.read_csv(stats_file, sep="\t")
            if len(df) > 0:
                frames.append(df)
        except Exception as e:
            print(f"  WARNING: Could not read {stats_file.name}: {e}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def annotate_categories(df: pd.DataFrame, rbp_to_cat: dict[str, str] | None) -> pd.DataFrame:
    """Add a Category column to the results DataFrame.

    If rbp_to_cat is None, returns df unchanged.
    """
    if rbp_to_cat is None or "RBP" not in df.columns:
        return df
    df["Category"] = df["RBP"].map(rbp_to_cat).fillna("Unknown")
    return df


def summarize_ratio_filter(df: pd.DataFrame, min_ratio: float | None) -> None:
    """Print a summary of how many targets pass a ratio filter (informational only).

    No files are written — filtering is applied at the visualization/plot layer,
    not the data layer.  The full unfiltered TSV is always preserved so edge
    cases remain visible to the researcher.
    """
    if min_ratio is None or "Ratio" not in df.columns:
        return
    n_pass = (df["Ratio"] >= min_ratio).sum()
    print(f"  Ratio >= {min_ratio}: {n_pass} / {len(df)} targets would pass (plot filter)")


def process_turbo(output_dir: Path, queries: list[str],
                  rbp_to_cat: dict[str, str] | None = None,
                  min_ratio: float | None = None):
    """Process turbo-format output (per-target .RELI.stats in query subdirs)."""
    for query in queries:
        query_dir = output_dir / query
        if not query_dir.is_dir():
            print(f"MISSING: {query_dir}")
            continue

        df = load_turbo_query(query_dir)
        if df.empty:
            print(f"\n{query}: no .RELI.stats files found")
            continue

        # Rename TF -> RBP for downstream compatibility (plot_figure.py)
        if "TF" in df.columns:
            df = df.rename(columns={"TF": "RBP"})

        # Annotate with functional category
        df = annotate_categories(df, rbp_to_cat)

        df = df.sort_values("Z-score", ascending=False).reset_index(drop=True)

        out = output_dir / f"{query}_all_results.tsv"
        df.to_csv(out, sep="\t", index=False)

        # Write ratio-filtered version if requested
        summarize_ratio_filter(df, min_ratio)

        # Bonferroni significance check (handle both column name variants)
        pval_col = "Corrected P-value" if "Corrected P-value" in df.columns else "Corrected P-val"
        if pval_col in df.columns:
            bonf = df[df[pval_col] < 0.05]
        else:
            bonf = pd.DataFrame()

        show_cols = [c for c in ["RBP", "Cell", "Category", "Z-score", pval_col] if c in df.columns]
        top5 = df.head(5)[show_cols].to_string(index=False)
        print(f"\n{query}: {len(df)} targets, {len(bonf)} Bonferroni-significant")
        print(top5)

    print("\nDone.")


def process_legacy(output_dir: Path, queries: list[str], clip_index: str, rbp_to_cat: dict[str, str] | None = None):
    """Process legacy format (_results.tsv files, requires CLIPseq.index)."""
    idx = pd.read_csv(clip_index, sep="\t")
    target_to_rbp = dict(zip(idx.iloc[:, 0], idx.iloc[:, 1]))

    for query in queries:
        tsv = output_dir / f"{query}_results.tsv"
        if not tsv.exists():
            print(f"MISSING: {tsv}")
            continue

        df = pd.read_csv(tsv, sep="\t")
        df["RBP"] = df["Label"].map(target_to_rbp).fillna(df["Label"])

        # Annotate with functional category
        df = annotate_categories(df, rbp_to_cat)

        df = df.sort_values("Z-score", ascending=False).reset_index(drop=True)

        out = output_dir / f"{query}_all_results.tsv"
        df.to_csv(out, sep="\t", index=False)

        pval_col = "Corrected P-value" if "Corrected P-value" in df.columns else "Corrected P-val"
        if pval_col in df.columns:
            bonf = df[df[pval_col] < 0.05]
        else:
            bonf = pd.DataFrame()

        show_cols = [c for c in ["RBP", "Cell", "Category", "Z-score", pval_col] if c in df.columns]
        top5 = df.head(5)[show_cols].to_string(index=False)
        print(f"\n{query}: {len(df)} targets, {len(bonf)} Bonferroni-significant")
        print(top5)

    print("\nDone.")


# Default queries for legacy fallback when --queries is not specified
STANDARD_QUERIES = [
    "SKIP_AltEX", "SKIP_DNintr", "SKIP_UPintr", "SKIP_merged",
    "INCL_AltEX", "INCL_DNintr", "INCL_UPintr", "INCL_merged",
]


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)

    if not output_dir.is_dir():
        print(f"ERROR: {output_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    # Load RBP category annotations
    rbp_to_cat = None
    if not args.no_categories:
        rbp_to_cat = load_categories(args.categories)
        if rbp_to_cat is not None:
            print(f"Loaded {len(rbp_to_cat)} RBP category annotations.")
        else:
            print("No RBP category file found; Category column will not be added.")

    # Detect layout: turbo (query subdirs with .RELI.stats) vs legacy vs pre-parsed
    discovered = discover_queries(output_dir)

    # Check for pre-parsed _all_results.tsv files (re-annotation mode)
    all_results_files = sorted(output_dir.glob("*_all_results.tsv"))
    if not discovered and all_results_files and (rbp_to_cat is not None or args.min_ratio is not None):
        mode_parts = []
        if rbp_to_cat is not None:
            mode_parts.append("category annotation")
        if args.min_ratio is not None:
            mode_parts.append(f"ratio filter >= {args.min_ratio}")
        print(f"Re-annotation mode ({', '.join(mode_parts)}): found {len(all_results_files)} _all_results.tsv files.")
        for tsv in all_results_files:
            df = pd.read_csv(tsv, sep="\t")
            if "TF" in df.columns:
                df = df.rename(columns={"TF": "RBP"})
            df = annotate_categories(df, rbp_to_cat)
            df.to_csv(tsv, sep="\t", index=False)
            summarize_ratio_filter(df, args.min_ratio)
            query_name = tsv.stem.replace("_all_results", "")
            n_cat = df["Category"].notna().sum() if "Category" in df.columns else 0
            print(f"  {query_name}: {len(df)} targets, {n_cat} categorized")
        print("\nDone (re-annotation).")
        sys.exit(0)

    if discovered:
        # Turbo layout
        queries = args.queries if args.queries else discovered
        missing = [q for q in queries if q not in discovered]
        if missing:
            print(f"WARNING: requested queries not found as subdirectories: {missing}")
            queries = [q for q in queries if q in discovered]
        print(f"Turbo layout detected. Queries: {queries}")
        process_turbo(output_dir, queries, rbp_to_cat, args.min_ratio)
    else:
        # Legacy fallback
        queries = args.queries if args.queries else STANDARD_QUERIES
        legacy_files = list(output_dir.glob("*_results.tsv"))
        if not legacy_files:
            print(f"ERROR: No query subdirectories or *_results.tsv files found in {output_dir}", file=sys.stderr)
            sys.exit(1)
        if not args.clip_index:
            print("ERROR: Legacy layout detected but --clip-index not provided.", file=sys.stderr)
            print("  For legacy _results.tsv files, pass: --clip-index /path/to/CLIPseq.index", file=sys.stderr)
            sys.exit(1)
        print(f"Legacy layout detected. Queries: {queries}")
        process_legacy(output_dir, queries, args.clip_index, rbp_to_cat)


if __name__ == "__main__":
    main()
