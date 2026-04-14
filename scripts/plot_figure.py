#!/usr/bin/env python3
"""
RBP-RELI enrichment bar plot: 3 columns x 2 rows with exon schematic.

General-purpose CLI tool for visualizing RELI enrichment results.
Reads *_all_results.tsv files produced by parse_results.py and generates
a publication-quality figure with gold INCL bars (top) and blue SKIP bars
(bottom) across Upstream Intron, Exon Body, and Downstream Intron panels.

Outputs both PNG (300 dpi) and PDF.
"""

import argparse
import sys

import matplotlib
matplotlib.use("Agg")  # headless backend (Docker-safe)

# ---------------------------------------------------------------------------
# rcParams: fonts and line weights (set once at import)
# ---------------------------------------------------------------------------
matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.default": "regular",
    "axes.linewidth": 0.8,
    "legend.frameon": False,
})

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Print-scale font sizes
# ---------------------------------------------------------------------------
FONT = {
    "title": 10,
    "subtitle": 9,
    "axis_label": 8,
    "tick": 7,
    "annotation": 7,
    "small_label": 6,
}

# White halo for all text over data
TEXT_HALO = [pe.withStroke(linewidth=3, foreground="white")]

# Journal dimensions (inches)
DOUBLE_COL = 7.2  # 183 mm double column
DPI = 300


INCL_COLOR = "#FFD700"   # gold
SKIP_COLOR = "#6495ED"   # cornflower blue
INCL_EDGE  = "#B8860B"   # dark goldenrod
SKIP_EDGE  = "#4169E1"   # royal blue

REGIONS = ["UPintr", "AltEX", "DNintr"]
REGION_LABELS = ["Upstream Intron", "Exon Body", "Downstream Intron"]


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate RBP-RELI enrichment barplot.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python plot_figure.py output/ --title 'KCNQ1OT1 RBP-RELI' --top-n 15\n"
        ),
    )
    p.add_argument(
        "output_dir",
        help="Directory containing *_all_results.tsv files (from parse_results.py)",
    )
    p.add_argument(
        "--out-dir", default=None,
        help="Figure output directory (default: {output_dir}/figures)",
    )
    p.add_argument("--title", default="RBP-RELI Enrichment", help="Figure title")
    p.add_argument("--top-n", type=int, default=10, help="Number of top RBPs per panel (default: 10)")
    p.add_argument("--y-max", type=float, default=None, help="Manual Y-axis limit (default: auto)")
    p.add_argument("--min-ratio", type=float, default=None,
                   help="Minimum overlap ratio to include in plot (e.g., 0.1)")
    p.add_argument("--exclude-category", nargs="+", default=None,
                   help="Exclude RBPs in these categories (e.g., 'Spliceosome')")
    p.add_argument("--include-category", nargs="+", default=None,
                   help="Include ONLY RBPs in these categories (overrides --exclude-category)")
    p.add_argument("--subtitle", default=None,
                   help="Experiment context line (e.g., 'DM1 (n=46) vs Control (n=11), tibialis anterior')")
    p.add_argument("--stem", default="reli_barplot",
                   help="Output filename stem (default: reli_barplot)")
    return p.parse_args()


def load_top_rbps(path, n, min_ratio=None, exclude_category=None, include_category=None):
    """Load TSV, apply filters, collapse to best Z-score per unique RBP, return top n.

    Filters (applied before top-N selection):
      - min_ratio: exclude rows with Ratio < threshold
      - exclude_category: exclude rows whose Category matches any of these strings
      - include_category: keep ONLY rows whose Category matches any of these strings
                          (takes precedence over exclude_category)
    """
    df = pd.read_csv(path, sep="\t")

    # Apply ratio filter
    if min_ratio is not None and "Ratio" in df.columns:
        df = df[df["Ratio"] >= min_ratio]

    # Apply category filter
    if "Category" in df.columns:
        if include_category is not None:
            df = df[df["Category"].isin(include_category)]
        elif exclude_category is not None:
            df = df[~df["Category"].isin(exclude_category)]

    if df.empty:
        return pd.DataFrame(columns=["RBP", "Z-score"])

    best = df.groupby("RBP", as_index=False)["Z-score"].max()
    best = best.sort_values("Z-score", ascending=False).head(n).reset_index(drop=True)
    return best


def draw_empty_panel(ax, message="No data"):
    """Configure an empty panel with a centered message."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.5, 0.5, message, ha="center", va="center",
            fontsize=FONT["annotation"], color="#999999", fontstyle="italic")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def main():
    args = parse_args()
    data_dir = Path(args.output_dir)
    fig_dir = Path(args.out_dir) if args.out_dir else data_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    top_n = args.top_n

    # --- Load data -----------------------------------------------------------
    files = {}
    for direction in ["INCL", "SKIP"]:
        for region in REGIONS:
            files[(direction, region)] = data_dir / f"{direction}_{region}_all_results.tsv"

    # Print active filters
    active_filters = []
    if args.min_ratio is not None:
        active_filters.append(f"ratio >= {args.min_ratio}")
    if args.include_category:
        active_filters.append(f"include: {args.include_category}")
    elif args.exclude_category:
        active_filters.append(f"exclude: {args.exclude_category}")
    if active_filters:
        print(f"Plot filters: {', '.join(active_filters)}")

    data = {}
    any_data = False
    for key, fpath in files.items():
        if not fpath.exists():
            print(f"WARNING: {fpath} not found — panel will be empty")
            data[key] = pd.DataFrame(columns=["RBP", "Z-score"])
            continue
        data[key] = load_top_rbps(
            fpath, top_n,
            min_ratio=args.min_ratio,
            exclude_category=args.exclude_category,
            include_category=args.include_category,
        )
        if len(data[key]) > 0:
            any_data = True

    if not any_data:
        print("ERROR: No data found in any panel. Nothing to plot.")
        sys.exit(1)

    # --- Y-axis limit --------------------------------------------------------
    if args.y_max:
        y_max = args.y_max
    else:
        all_z = [d["Z-score"].max() for d in data.values() if len(d) > 0]
        y_max = max(all_z) * 1.15 if all_z else 22

    # --- Figure layout -------------------------------------------------------
    # Height 11 (was 9) gives room above plots for tick labels + titles + text.
    # gs.top=0.80 (was 0.88) reserves 20% of figure height above the axes for
    # the rotated x-tick labels, column titles, "included" label, and suptitle.
    fig = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL * 0.86), facecolor="white")
    gs = fig.add_gridspec(
        nrows=3, ncols=3,
        height_ratios=[1, 0.08, 1],
        hspace=0.35, wspace=0.35,
        left=0.07, right=0.97, top=0.76, bottom=0.10,
    )

    # --- Draw panels ---------------------------------------------------------
    for col_idx, region in enumerate(REGIONS):
        # INCL (top row) -----------------------------------------------------
        ax_top = fig.add_subplot(gs[0, col_idx])
        d = data[("INCL", region)]
        if len(d) == 0:
            draw_empty_panel(ax_top)
        else:
            ax_top.bar(
                range(len(d)), d["Z-score"],
                color=INCL_COLOR, edgecolor=INCL_EDGE, linewidth=0.5, width=0.7,
            )
            ax_top.set_xticks(range(len(d)))
            ax_top.set_xticklabels(
                d["RBP"], rotation=55, ha="left", fontsize=FONT["tick"],
                rotation_mode="anchor",
            )
            ax_top.tick_params(
                axis="x", bottom=False, top=False, pad=2, length=0,
                labeltop=True, labelbottom=False,
            )
            ax_top.xaxis.set_ticks_position("top")
            ax_top.set_xlim(-0.6, top_n - 0.4)
            ax_top.set_ylabel("Z-score" if col_idx == 0 else "", fontsize=FONT["axis_label"])
            ax_top.spines["top"].set_visible(False)
            ax_top.spines["right"].set_visible(False)
            ax_top.spines["bottom"].set_visible(False)
            ax_top.tick_params(axis="y", labelsize=FONT["tick"])
            ax_top.set_ylim(0, y_max)
            ax_top.yaxis.grid(True, alpha=0.3, linewidth=0.5)
            ax_top.set_axisbelow(True)

        # Column titles are placed with fig.text() below (not set_title)
        # so they align horizontally across all columns regardless of
        # tick label height differences.

        # SKIP (bottom row) --------------------------------------------------
        ax_bot = fig.add_subplot(gs[2, col_idx])
        d2 = data[("SKIP", region)]
        if len(d2) == 0:
            draw_empty_panel(ax_bot)
        else:
            ax_bot.bar(
                range(len(d2)), -d2["Z-score"],
                color=SKIP_COLOR, edgecolor=SKIP_EDGE, linewidth=0.5, width=0.7,
            )
            ax_bot.set_xticks(range(len(d2)))
            ax_bot.set_xticklabels(d2["RBP"], rotation=55, ha="right", fontsize=FONT["tick"])
            ax_bot.set_xlim(-0.6, top_n - 0.4)
            ax_bot.set_ylabel("Z-score" if col_idx == 0 else "", fontsize=FONT["axis_label"])
            ax_bot.spines["top"].set_visible(False)
            ax_bot.spines["right"].set_visible(False)
            ax_bot.tick_params(axis="y", labelsize=FONT["tick"])
            ax_bot.set_ylim(-y_max, 0)
            ax_bot.yaxis.grid(True, alpha=0.3, linewidth=0.5)
            ax_bot.set_axisbelow(True)

    # --- Exon schematic (middle row, spans all columns) ----------------------
    ax_s = fig.add_subplot(gs[1, :])
    ax_s.set_xlim(0, 1)
    ax_s.set_ylim(0, 1)
    ax_s.axis("off")
    y_mid = 0.5
    ax_s.plot([0.15, 0.40], [y_mid, y_mid], color="black", linewidth=2,
              solid_capstyle="butt")
    ax_s.plot([0.60, 0.85], [y_mid, y_mid], color="black", linewidth=2,
              solid_capstyle="butt")
    exon_rect = mpatches.FancyBboxPatch(
        (0.40, y_mid - 0.3), 0.20, 0.6,
        boxstyle="round,pad=0.02", facecolor="#888888",
        edgecolor="black", linewidth=1.5,
    )
    ax_s.add_patch(exon_rect)
    ax_s.text(0.12, y_mid, "5'", fontsize=FONT["axis_label"], fontweight="bold",
              ha="right", va="center", path_effects=TEXT_HALO)
    ax_s.text(0.88, y_mid, "3'", fontsize=FONT["axis_label"], fontweight="bold",
              ha="left", va="center", path_effects=TEXT_HALO)

    # --- Figure-level text ---------------------------------------------------
    # Column titles at fixed y so they align across all three columns.
    # Compute x-centers from gridspec: left=0.07, right=0.97, wspace=0.35
    gs_left, gs_right, gs_wspace = 0.07, 0.97, 0.35
    # Column titles placed just above gridspec top (0.76)
    ax_w = (gs_right - gs_left) / (3 + 2 * gs_wspace)  # axes width
    gap = gs_wspace * ax_w
    col_centers = [gs_left + ax_w / 2 + i * (ax_w + gap) for i in range(3)]
    for i, label in enumerate(REGION_LABELS):
        fig.text(col_centers[i], 0.875, label, fontsize=FONT["axis_label"],
                 fontweight="bold", ha="center", va="bottom",
                 path_effects=TEXT_HALO)

    # --- Figure-level annotations ---
    # Stack from top: title -> subtitle -> filters -> "included"
    y_cursor = 0.98
    fig.suptitle(args.title, fontsize=FONT["title"], fontweight="bold",
                 y=y_cursor, path_effects=TEXT_HALO)
    y_cursor -= 0.025

    if args.subtitle:
        fig.text(0.52, y_cursor, args.subtitle, fontsize=FONT["annotation"],
                 ha="center", va="top", color="#555555", fontstyle="italic",
                 path_effects=TEXT_HALO)
        y_cursor -= 0.02

    filter_parts = []
    if args.min_ratio is not None:
        filter_parts.append(f"ratio \u2265 {args.min_ratio}")
    if args.include_category:
        filter_parts.append(f"only: {', '.join(args.include_category)}")
    elif args.exclude_category:
        filter_parts.append(f"excl. {', '.join(args.exclude_category)}")
    if filter_parts:
        filter_text = "Filters: " + "; ".join(filter_parts)
        fig.text(0.52, y_cursor, filter_text, fontsize=FONT["small_label"],
                 ha="center", va="top", color="#999999",
                 path_effects=TEXT_HALO)
        y_cursor -= 0.015

    fig.text(0.52, y_cursor - 0.005, "included", fontsize=FONT["subtitle"],
             fontweight="bold", ha="center", va="top", color=INCL_EDGE,
             path_effects=TEXT_HALO)
    fig.text(0.52, 0.005, "skipped", fontsize=FONT["subtitle"],
             fontweight="bold", ha="center", va="bottom", color=SKIP_EDGE,
             path_effects=TEXT_HALO)

    # --- Save ----------------------------------------------------------------
    stem = args.stem
    png_path = fig_dir / f"{stem}.png"
    pdf_path = fig_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=DPI, bbox_inches="tight", facecolor="white",
                edgecolor="none", pad_inches=0.05)
    fig.savefig(pdf_path, dpi=DPI, bbox_inches="tight", facecolor="white",
                edgecolor="none", pad_inches=0.05)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
