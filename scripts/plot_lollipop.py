#!/usr/bin/env python3
"""
RBP-RELI lollipop plot: panel layout with horizontal exon schematic.

Layout: 3 columns (Upstream Intron, Exon Body, Downstream Intron)
        INCL (gold) on top pointing UP, SKIP (blue) on bottom pointing DOWN.
        Horizontal exon schematic in the middle row.

Each lollipop encodes three variables:
  - Stalk length  = Z-score (vertical)
  - Circle size   = Ratio (overlap / total)
  - Circle color  = Corrected P-value (light -> saturated)

Reads *_all_results.tsv files produced by parse_results.py.
"""

import argparse
import sys

import matplotlib
matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# rcParams: figure defaults (Arial/Helvetica, no legend frame)
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
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Print-scale font sizes (designed at final column width)
# ---------------------------------------------------------------------------
FS_TITLE = 10
FS_SUBTITLE = 9
FS_AXIS_LABEL = 8
FS_TICK = 7
FS_ANNOTATION = 7
FS_SMALL_LABEL = 6

TEXT_HALO = [pe.withStroke(linewidth=3, foreground="white")]


INCL_CMAP = mcolors.LinearSegmentedColormap.from_list("incl", ["#FFF8DC", "#DAA520"])
SKIP_CMAP = mcolors.LinearSegmentedColormap.from_list("skip", ["#E8EEF8", "#2B4C99"])
INCL_EDGE = "#B8860B"
SKIP_EDGE = "#4169E1"

# Splicing mode (rMATS): INCL/SKIP × UPintr/AltEX/DNintr
SPLICE_REGIONS = ["UPintr", "AltEX", "DNintr"]
SPLICE_REGION_LABELS = ["Upstream Intron", "Exon Body", "Downstream Intron"]
SPLICE_DIRECTIONS = ["INCL", "SKIP"]

# DESeq2 mode: UP/DOWN × regions ordered as pre-mRNA (5'→3')
DESEQ2_REGIONS = ["5UTR", "CDS", "intron", "3UTR"]
DESEQ2_REGION_LABELS = ["5' UTR", "CDS", "Intron", "3' UTR"]
DESEQ2_DIRECTIONS = ["UP", "DOWN"]

# Direction display labels and colors for DESeq2 mode
UP_EDGE = "#B8860B"      # same gold family
DOWN_EDGE = "#4169E1"     # same blue family
UP_CMAP = INCL_CMAP       # reuse gold for "up"
DOWN_CMAP = SKIP_CMAP     # reuse blue for "down"


def parse_args():
    p = argparse.ArgumentParser(
        description="RBP-RELI lollipop plot with horizontal exon schematic.",
    )
    p.add_argument("output_dir", help="Directory containing *_all_results.tsv files")
    p.add_argument("--out-dir", default=None, help="Figure output directory")
    p.add_argument("--title", default="RBP-RELI Enrichment", help="Figure title")
    p.add_argument("--subtitle", default=None, help="Line 2: rMATS parameters and event counts")
    p.add_argument("--subtitle2", default=None, help="Line 3: RBP-RELI filtering criteria")
    p.add_argument("--top-n", type=int, default=10, help="Top RBPs per panel (default: 10)")
    p.add_argument("--min-ratio", type=float, default=None, help="Min overlap ratio filter")
    p.add_argument("--max-pval", type=float, default=0.05, help="Max corrected P-value (default: 0.05)")
    p.add_argument("--exclude-category", nargs="+", default=None, help="Exclude these categories")
    p.add_argument("--include-category", nargs="+", default=None, help="Include ONLY these categories")
    p.add_argument("--stem", default="reli_lollipop", help="Output filename stem")
    p.add_argument("--y-max", type=float, default=None, help="Manual Y-axis limit (default: auto)")
    p.add_argument("--all-clips", action="store_true",
                   help="Show all CLIP datasets (RBP can appear multiple times with cell line label)")
    return p.parse_args()


def load_panel_data(path, n, min_ratio=None, max_pval=None,
                    exclude_category=None, include_category=None,
                    all_clips=False):
    df = pd.read_csv(path, sep="\t")
    pval_col = "Corrected P-value" if "Corrected P-value" in df.columns else "P-value"
    if max_pval is not None and pval_col in df.columns:
        df = df[df[pval_col] <= max_pval]
    if min_ratio is not None and "Ratio" in df.columns:
        df = df[df["Ratio"] >= min_ratio]
    if "Category" in df.columns:
        if include_category is not None:
            df = df[df["Category"].isin(include_category)]
        elif exclude_category is not None:
            df = df[~df["Category"].isin(exclude_category)]
    if df.empty:
        return pd.DataFrame()
    if all_clips:
        # Show all CLIP datasets — RBP can appear multiple times
        best = df.sort_values("Z-score", ascending=False).head(n).reset_index(drop=True)
        # Build display label: "RBP\n(Cell)" for two-line x-axis labels
        if "Cell" in df.columns:
            cell = best["Cell"].fillna("").astype(str)
            # Strip replicate suffixes like @1, @2
            cell = cell.str.replace(r"@\d+$", "", regex=True)
            best["_display_label"] = best["RBP"] + "\n(" + cell + ")"
        else:
            best["_display_label"] = best["RBP"]
    else:
        idx = df.groupby("RBP")["Z-score"].idxmax()
        best = df.loc[idx].sort_values("Z-score", ascending=False).head(n).reset_index(drop=True)
    return best


def draw_vertical_lollipop(ax, df, cmap, edge_color, direction, y_max, top_n):
    """Draw vertical lollipops. direction='INCL' goes up, 'SKIP' goes down."""
    if df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes,
                fontsize=FS_ANNOTATION, color="#999999", fontstyle="italic")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        return

    n = len(df)
    x_pos = np.arange(n)

    z_scores = df["Z-score"].values
    # Circle size = Enrichment (fold over null expectation)
    enrich = df["Enrichment"].values if "Enrichment" in df.columns else np.full(n, 1.0)
    # Scale: enrichment 1x → small, 3x → medium, 8x → large
    sizes = np.clip(enrich, 1.0, 10.0) * 40

    pval_col = "Corrected P-value" if "Corrected P-value" in df.columns else "P-value"
    if pval_col in df.columns:
        pvals = np.clip(df[pval_col].values.astype(float), 1e-300, 1.0)
        log_pvals = -np.log10(pvals)
        # Scale across the actual data range so colors differentiate the plotted points
        vmin = max(log_pvals.min() - 5, 0)
        vmax = log_pvals.max() + 5
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        colors = cmap(norm(log_pvals))
    else:
        colors = [cmap(0.7)] * n

    # Compute uniform ticks based on y_max
    tick_step = max(1, int(np.ceil(y_max / 5)))  # ~5 ticks
    uniform_ticks = np.arange(0, y_max + tick_step, tick_step)

    if direction == "SKIP":
        y_vals = -z_scores
        ax.set_ylim(-y_max, 0)
        ax.set_yticks(-uniform_ticks)
    else:
        y_vals = z_scores
        ax.set_ylim(0, y_max)
        ax.set_yticks(uniform_ticks)

    # Draw stalks (vertical segments from 0 to Z)
    for i in range(n):
        ax.plot([x_pos[i], x_pos[i]], [0, y_vals[i]], color="#CCCCCC",
                linewidth=1.2, zorder=1)

    # Draw circles (clip_on=False so circles at axis edge aren't truncated)
    ax.scatter(x_pos, y_vals, s=sizes, c=colors, edgecolors=edge_color,
               linewidths=0.6, zorder=2, clip_on=False)

    # X-axis: RBP names (or RBP + cell line in all-clips mode)
    ax.set_xticks(x_pos)
    ax.set_xlim(-0.6, top_n - 0.4)

    has_two_line = "_display_label" in df.columns

    if has_two_line:
        # Two-size labels: RBP as tick label, cell line as separate smaller text
        rbp_names = df["RBP"].values
        cell_names = []
        for i in range(n):
            cell_raw = df["Cell"].values[i] if "Cell" in df.columns else ""
            cell_names.append(str(cell_raw).split("@")[0] if cell_raw else "")

        if direction == "INCL":
            ax.set_xticklabels(rbp_names, rotation=55, ha="right",
                               fontsize=FS_TICK - 0.5, fontweight="bold")
            ax.tick_params(axis="x", bottom=True, top=False, pad=2, length=0,
                           labeltop=False, labelbottom=True)
        else:
            ax.set_xticklabels(rbp_names, rotation=55, ha="left",
                               fontsize=FS_TICK - 0.5, fontweight="bold",
                               rotation_mode="anchor")
            ax.tick_params(axis="x", bottom=False, top=True, pad=2, length=0,
                           labeltop=True, labelbottom=False)
            ax.xaxis.set_ticks_position("top")

        # Draw cell line labels at the lollipop stalk base (y=0 in data).
        for i in range(n):
            if not cell_names[i]:
                continue
            if direction == "INCL":
                ax.text(x_pos[i] + 0.55, 0, f"({cell_names[i]})",
                        fontsize=FS_TICK - 2.5, color="#666666",
                        rotation=55, ha="right", va="top",
                        clip_on=False)
            else:
                ax.text(x_pos[i] + 0.30, 0, f"({cell_names[i]})",
                        fontsize=FS_TICK - 2.5, color="#666666",
                        rotation=55, ha="left", va="bottom",
                        rotation_mode="anchor",
                        clip_on=False)

        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
    elif direction == "INCL":
        labels = df["RBP"].values
        # Names at BOTTOM of INCL panels (adjacent to exon model)
        ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=FS_TICK)
        ax.tick_params(axis="x", bottom=True, top=False, pad=2, length=0,
                       labeltop=False, labelbottom=True)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
    else:
        labels = df["RBP"].values
        # Names at TOP of SKIP panels (adjacent to exon model)
        ax.set_xticklabels(labels, rotation=55, ha="left", fontsize=FS_TICK,
                           rotation_mode="anchor")
        ax.tick_params(axis="x", bottom=False, top=True, pad=2, length=0,
                       labeltop=True, labelbottom=False)
        ax.xaxis.set_ticks_position("top")
        ax.spines["top"].set_visible(False)

    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=FS_TICK)
    ax.yaxis.grid(True, alpha=0.15, linewidth=0.5)
    ax.set_axisbelow(True)

    # Y-axis label
    if direction == "SKIP":
        ticks = ax.get_yticks()
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{abs(t):.0f}" for t in ticks], fontsize=FS_TICK)


def main():
    args = parse_args()
    data_dir = Path(args.output_dir)
    fig_dir = Path(args.out_dir) if args.out_dir else data_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    top_n = args.top_n

    active_filters = []
    if args.max_pval is not None:
        active_filters.append(f"P(adj) <= {args.max_pval}")
    if args.min_ratio is not None:
        active_filters.append(f"ratio >= {args.min_ratio}")
    if args.include_category:
        active_filters.append(f"only: {', '.join(args.include_category)}")
    elif args.exclude_category:
        active_filters.append(f"excl. {', '.join(args.exclude_category)}")
    if active_filters:
        print(f"Plot filters: {', '.join(active_filters)}")

    # --- Auto-detect mode: splicing (INCL/SKIP) vs DESeq2 (UP/DOWN) ----------
    splice_files = list(data_dir.glob("INCL_*_all_results.tsv")) + \
                   list(data_dir.glob("SKIP_*_all_results.tsv"))
    deseq2_files = list(data_dir.glob("UP_*_all_results.tsv")) + \
                   list(data_dir.glob("DOWN_*_all_results.tsv"))

    # Also check for single-query files (database mode, e.g., ARE_3UTR)
    all_result_files = list(data_dir.glob("*_all_results.tsv"))
    single_query_files = [f for f in all_result_files
                          if not f.name.startswith(("INCL_", "SKIP_", "UP_", "DOWN_"))]

    if splice_files and not deseq2_files:
        mode = "splice"
    elif deseq2_files and not splice_files:
        mode = "deseq2"
    elif splice_files and deseq2_files:
        mode = "splice"
    elif single_query_files:
        mode = "database"
    else:
        print("ERROR: No *_all_results.tsv files found.")
        sys.exit(1)

    if mode == "splice":
        directions = SPLICE_DIRECTIONS
        regions = SPLICE_REGIONS
        region_labels = SPLICE_REGION_LABELS
        up_cmap, down_cmap = INCL_CMAP, SKIP_CMAP
        up_edge, down_edge = INCL_EDGE, SKIP_EDGE
        up_label, down_label = "included", "skipped"
    else:
        directions = DESEQ2_DIRECTIONS
        regions = DESEQ2_REGIONS
        region_labels = DESEQ2_REGION_LABELS
        up_cmap, down_cmap = UP_CMAP, DOWN_CMAP
        up_edge, down_edge = UP_EDGE, DOWN_EDGE
        up_label, down_label = "upregulated", "downregulated"

    # --- Database mode: single-query horizontal lollipop ----------------------
    if mode == "database":
        query_file = single_query_files[0]
        query_name = query_file.stem.replace("_all_results", "")
        print(f"Mode: database (single query: {query_name})")
        df = load_panel_data(
            query_file, top_n,
            min_ratio=args.min_ratio, max_pval=args.max_pval,
            exclude_category=args.exclude_category,
            include_category=args.include_category,
            all_clips=args.all_clips,
        )
        if df.empty:
            print("ERROR: No data after filtering.")
            sys.exit(1)

        n = len(df)
        z_scores = df["Z-score"].values
        enrich = df["Enrichment"].values if "Enrichment" in df.columns else np.ones(n)
        sizes = np.clip(enrich, 1.0, 10.0) * 40

        pval_col = "Corrected P-value" if "Corrected P-value" in df.columns else "P-value"
        if pval_col in df.columns:
            pvals = np.clip(df[pval_col].values.astype(float), 1e-300, 1.0)
            log_pvals = -np.log10(pvals)
            vmin = max(log_pvals.min() - 5, 0)
            vmax = log_pvals.max() + 5
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            colors = INCL_CMAP(norm(log_pvals))
        else:
            colors = [INCL_CMAP(0.7)] * n

        # Build labels
        rbp_labels = df["RBP"].values
        cell_labels = []
        if args.all_clips and "Cell" in df.columns:
            for i in range(n):
                cell_labels.append(str(df["Cell"].values[i]).split("@")[0])
        else:
            cell_labels = [""] * n

        # Horizontal figure — wider to leave room for legend on right
        fig_h = max(4.0, 0.38 * n + 1.8)
        fig, ax = plt.subplots(figsize=(8.5, fig_h), facecolor="white")
        fig.subplots_adjust(right=0.75)  # leave space for legend
        y_pos = np.arange(n)[::-1]  # top to bottom = highest Z at top

        # Stalks
        for i in range(n):
            ax.plot([0, z_scores[i]], [y_pos[i], y_pos[i]], color="#CCCCCC",
                    linewidth=1.2, zorder=1)
        # Circles
        ax.scatter(z_scores, y_pos, s=sizes, c=colors, edgecolors=INCL_EDGE,
                   linewidths=0.6, zorder=2, clip_on=False)

        # Y-axis: RBP names as tick labels
        ax.set_yticks(y_pos)
        ax.set_yticklabels(rbp_labels, fontsize=FS_TICK, fontweight="bold")
        ax.set_ylim(-0.8, n - 0.2)

        # Cell line labels: below each RBP (between this RBP and next)
        if any(cell_labels):
            for i in range(n):
                if cell_labels[i]:
                    ax.text(-0.01, y_pos[i] - 0.35, f"({cell_labels[i]})",
                            transform=ax.get_yaxis_transform(),
                            fontsize=FS_TICK - 2.5, color="#666666",
                            ha="right", va="top", clip_on=False)

        ax.set_xlabel("Z-score", fontsize=FS_AXIS_LABEL)
        ax.set_xlim(0, None)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.xaxis.grid(True, alpha=0.15, linewidth=0.5)
        ax.set_axisbelow(True)

        # Title block
        title_y = 0.98
        if args.title:
            fig.text(0.5, title_y, args.title, ha="center", va="top",
                     fontsize=FS_TITLE, fontweight="bold")
            title_y -= 0.04
        if args.subtitle:
            fig.text(0.5, title_y, args.subtitle, ha="center", va="top",
                     fontsize=FS_SUBTITLE, color="#444444")
            title_y -= 0.035
        if args.subtitle2:
            fig.text(0.5, title_y, args.subtitle2, ha="center", va="top",
                     fontsize=FS_SMALL_LABEL, color="#888888")

        # Enrichment + P-value legend — outside plot on the right
        ax_leg = fig.add_axes([0.78, 0.15, 0.18, 0.7])
        ax_leg.set_xlim(0, 1)
        ax_leg.set_ylim(0, 1)
        ax_leg.axis("off")

        # Enrichment section
        ax_leg.text(0.4, 0.97, "Enrichment", fontsize=FS_ANNOTATION, fontweight="bold", ha="center")
        for i, e in enumerate([1, 2, 4, 8]):
            y = 0.90 - i * 0.08
            ax_leg.scatter([0.2], [y], s=[e * 40], facecolor="#CCCCCC", edgecolor="black",
                           linewidths=0.5, clip_on=False, zorder=2)
            ax_leg.text(0.55, y, f"{e}x", fontsize=FS_ANNOTATION, ha="left", va="center")

        # P-value section
        all_pvals_db = []
        pcol_db = "Corrected P-value" if "Corrected P-value" in df.columns else "P-value"
        if pcol_db in df.columns:
            all_pvals_db = [p for p in df[pcol_db].values.astype(float) if 0 < p <= 0.05]
        if all_pvals_db:
            log_min_db = -np.log10(min(all_pvals_db))
            log_max_db = -np.log10(max(all_pvals_db))
        else:
            log_min_db, log_max_db = 100, 2

        leg_log_db = np.linspace(log_max_db, log_min_db, 4)
        leg_pvals_db = 10 ** (-leg_log_db)
        pnorm_db = plt.Normalize(vmin=max(log_max_db - 5, 0), vmax=log_min_db + 5)

        ax_leg.text(0.4, 0.53, "P (adj.)", fontsize=FS_ANNOTATION, fontweight="bold", ha="center")
        for i, (lv, pv) in enumerate(zip(leg_log_db, leg_pvals_db)):
            y = 0.46 - i * 0.07
            if pv >= 0.001:
                label = f"< {pv:.3f}"
            else:
                exp = int(-np.log10(pv))
                label = f"< 1e-{exp}"
            color_db = INCL_CMAP(pnorm_db(lv))
            ax_leg.scatter([0.2], [y], s=[70], facecolor=color_db, edgecolor=INCL_EDGE,
                           linewidths=0.5, clip_on=False, zorder=2)
            ax_leg.text(0.45, y, label, fontsize=FS_SMALL_LABEL, ha="left", va="center")

        png_path = fig_dir / f"{args.stem}.png"
        pdf_path = fig_dir / f"{args.stem}.pdf"
        fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white",
                    edgecolor="none", pad_inches=0.05)
        fig.savefig(pdf_path, dpi=300, bbox_inches="tight", facecolor="white",
                    edgecolor="none", pad_inches=0.05)
        print(f"Saved: {png_path}")
        print(f"Saved: {pdf_path}")
        plt.close(fig)
        sys.exit(0)

    print(f"Mode: {mode} ({len(regions)} regions: {', '.join(regions)})")

    ncols = len(regions)

    data = {}
    any_data = False
    for direction in directions:
        for region in regions:
            fpath = data_dir / f"{direction}_{region}_all_results.tsv"
            if not fpath.exists():
                data[(direction, region)] = pd.DataFrame()
                continue
            df = load_panel_data(
                fpath, top_n,
                min_ratio=args.min_ratio, max_pval=args.max_pval,
                exclude_category=args.exclude_category,
                include_category=args.include_category,
                all_clips=args.all_clips,
            )
            data[(direction, region)] = df
            if not df.empty:
                any_data = True

    if not any_data:
        print("ERROR: No data. Nothing to plot.")
        sys.exit(1)

    if args.y_max:
        y_max = args.y_max
    else:
        all_z = [d["Z-score"].max() for d in data.values() if not d.empty]
        # Round up to nearest 5 for clean axis, same range for both directions
        raw_max = max(all_z) * 1.05 if all_z else 22
        y_max = int(np.ceil(raw_max / 5) * 5)

    # --- Figure layout -------------------------------------------------------
    # Splice mode: 3 cols with exon schematic in middle row
    # DESeq2 mode: 4 cols, no exon schematic (replaced by thin separator)
    if mode == "splice":
        fig_w = 7.2
        nrows = 3
        height_ratios = [1, 0.12, 1]
    else:
        fig_w = 9.0
        nrows = 3
        height_ratios = [1, 0.04, 1]

    fig = plt.figure(figsize=(fig_w, 6.5), facecolor="white")
    gs = fig.add_gridspec(
        nrows=nrows, ncols=ncols,
        height_ratios=height_ratios,
        hspace=0.35, wspace=0.12,
        left=0.06, right=0.84, top=0.82, bottom=0.08,
    )

    dir_up, dir_down = directions[0], directions[1]

    for col_idx, region in enumerate(regions):
        # Top row (UP / INCL — lollipops pointing UP)
        ax_top = fig.add_subplot(gs[0, col_idx])
        draw_vertical_lollipop(ax_top, data[(dir_up, region)], up_cmap, up_edge,
                               "INCL", y_max, top_n)
        if col_idx == 0:
            ax_top.set_ylabel("Z-score", fontsize=FS_AXIS_LABEL)
        elif col_idx == ncols - 1:
            ax_top.yaxis.tick_right()
            ax_top.yaxis.set_label_position("right")
            ax_top.set_ylabel("Z-score", fontsize=FS_AXIS_LABEL)
        else:
            ax_top.set_yticklabels([])
            ax_top.set_ylabel("")

        # Bottom row (DOWN / SKIP — lollipops pointing DOWN)
        ax_bot = fig.add_subplot(gs[2, col_idx])
        draw_vertical_lollipop(ax_bot, data[(dir_down, region)], down_cmap, down_edge,
                               "SKIP", y_max, top_n)
        if col_idx == 0:
            ax_bot.set_ylabel("Z-score", fontsize=FS_AXIS_LABEL)
        elif col_idx == ncols - 1:
            ax_bot.yaxis.tick_right()
            ax_bot.yaxis.set_label_position("right")
            ax_bot.set_ylabel("Z-score", fontsize=FS_AXIS_LABEL)
        else:
            ax_bot.set_yticklabels([])
            ax_bot.set_ylabel("")

    # --- Middle row: exon schematic (splice) or separator line (DESeq2) -----
    gs_left, gs_right_val, gs_wspace_val = 0.06, 0.84, 0.12

    if mode == "splice":
        gs_total = gs_right_val - gs_left
        col_w = gs_total / (ncols + (ncols - 1) * gs_wspace_val)
        col_gap = gs_wspace_val * col_w

        c0_l = gs_left
        c0_r = c0_l + col_w
        c1_l = c0_r + col_gap
        c1_r = c1_l + col_w
        c2_l = c1_r + col_gap
        c2_r = c2_l + col_w

        ax_s = fig.add_subplot(gs[1, :])
        ax_s.set_xlim(gs_left, gs_right_val)
        ax_s.set_ylim(0, 1)
        ax_s.axis("off")
        y_mid = 0.5

        ax_s.plot([c0_l + 0.005, c0_r], [y_mid, y_mid], color="black", linewidth=3,
                  solid_capstyle="butt", zorder=1)
        ax_s.plot([c2_l, c2_r - 0.005], [y_mid, y_mid], color="black", linewidth=3,
                  solid_capstyle="butt", zorder=1)

        exon_inset = col_gap * 0.6
        exon_rect = mpatches.FancyBboxPatch(
            (c1_l + exon_inset, y_mid - 0.38),
            (c1_r - c1_l - 2 * exon_inset), 0.76,
            boxstyle="round,pad=0.005", facecolor="#888888",
            edgecolor="black", linewidth=2, zorder=2,
        )
        ax_s.add_patch(exon_rect)

        ax_s.text(c0_l - 0.005, y_mid, "5'", fontsize=FS_AXIS_LABEL, fontweight="bold",
                  ha="right", va="center", path_effects=TEXT_HALO)
        ax_s.text(c2_r + 0.005, y_mid, "3'", fontsize=FS_AXIS_LABEL, fontweight="bold",
                  ha="left", va="center", path_effects=TEXT_HALO)
    else:
        # DESeq2 mode: thin horizontal separator
        ax_s = fig.add_subplot(gs[1, :])
        ax_s.set_xlim(0, 1)
        ax_s.set_ylim(0, 1)
        ax_s.axis("off")
        ax_s.axhline(0.5, color="#CCCCCC", linewidth=0.8, linestyle="--")

    # --- Column titles (above top panels) ------------------------------------
    gs_left_t, gs_right_t, gs_wspace_t = 0.06, 0.84, 0.12
    ax_w = (gs_right_t - gs_left_t) / (ncols + (ncols - 1) * gs_wspace_t)
    gap = gs_wspace_t * ax_w
    col_centers = [gs_left_t + ax_w / 2 + i * (ax_w + gap) for i in range(ncols)]
    for i, label in enumerate(region_labels):
        fig.text(col_centers[i], 0.86, label, fontsize=FS_AXIS_LABEL, fontweight="bold",
                 ha="center", va="bottom", color="#444444")

    # --- Title / subtitle / subtitle2 / direction labels ---------------------
    y_cursor = 0.98
    fig.suptitle(args.title, fontsize=FS_TITLE, fontweight="bold", y=y_cursor)
    y_cursor -= 0.025
    if args.subtitle:
        fig.text(0.48, y_cursor, args.subtitle, fontsize=FS_SUBTITLE, ha="center",
                 va="top", color="#555555", fontstyle="italic")
        y_cursor -= 0.02
    if args.subtitle2:
        fig.text(0.48, y_cursor, args.subtitle2, fontsize=FS_SMALL_LABEL, ha="center",
                 va="top", color="#777777")
        y_cursor -= 0.015

    fig.text(0.48, y_cursor - 0.003, up_label, fontsize=FS_AXIS_LABEL, fontweight="bold",
             ha="center", va="top", color=up_edge)
    fig.text(0.48, 0.01, down_label, fontsize=FS_AXIS_LABEL, fontweight="bold",
             ha="center", va="bottom", color=down_edge)

    # --- Legend (right side) -------------------------------------------------
    ax_leg = fig.add_axes([0.88, 0.25, 0.11, 0.50])
    ax_leg.set_xlim(0, 1)
    ax_leg.set_ylim(0, 1)
    ax_leg.axis("off")

    ax_leg.text(0.35, 0.97, "Enrichment", fontsize=FS_ANNOTATION, fontweight="bold", ha="center")
    for i, e in enumerate([1, 2, 4, 8]):
        y = 0.90 - i * 0.08
        ax_leg.scatter([0.25], [y], s=[e * 40], facecolor="#CCCCCC", edgecolor="black",
                       linewidths=0.5, clip_on=False, zorder=2)
        ax_leg.text(0.58, y, f"{e}x", fontsize=FS_ANNOTATION, ha="left", va="center")

    # Compute actual P-value range from data for representative legend
    all_pvals = []
    for d in data.values():
        if not d.empty:
            pcol = "Corrected P-value" if "Corrected P-value" in d.columns else "P-value"
            if pcol in d.columns:
                all_pvals.extend(d[pcol].values.astype(float))
    if all_pvals:
        all_pvals = np.array([p for p in all_pvals if p > 0 and p <= 0.05])
        if len(all_pvals) > 0:
            log_min = -np.log10(all_pvals.min())
            log_max = -np.log10(all_pvals.max())
        else:
            log_min, log_max = 2, 100
    else:
        log_min, log_max = 2, 100

    # Build 4 legend ticks spanning the actual data range
    leg_log_vals = np.linspace(log_max, log_min, 4)
    leg_pvals = 10 ** (-leg_log_vals)
    pnorm_leg = plt.Normalize(vmin=max(log_max - 5, 0), vmax=log_min + 5)

    ax_leg.text(0.40, 0.53, "P (adj.)", fontsize=FS_ANNOTATION, fontweight="bold", ha="center")
    for i, (lv, pv) in enumerate(zip(leg_log_vals, leg_pvals)):
        y = 0.46 - i * 0.065
        # Human-friendly label
        if pv >= 0.001:
            label = f"< {pv:.3f}"
        elif pv >= 1e-6:
            exp = int(-np.log10(pv))
            label = f"< 1e-{exp}"
        else:
            exp = int(-np.log10(pv))
            label = f"< 1e-{exp}"
        color_incl = INCL_CMAP(pnorm_leg(lv))
        ax_leg.scatter([0.15], [y], s=[70], facecolor=color_incl, edgecolor=INCL_EDGE,
                       linewidths=0.5, clip_on=False, zorder=2)
        color_skip = SKIP_CMAP(pnorm_leg(lv))
        ax_leg.scatter([0.30], [y], s=[70], facecolor=color_skip, edgecolor=SKIP_EDGE,
                       linewidths=0.5, clip_on=False, zorder=2)
        ax_leg.text(0.45, y, label, fontsize=FS_SMALL_LABEL, ha="left", va="center")

    # --- Save ----------------------------------------------------------------
    png_path = fig_dir / f"{args.stem}.png"
    pdf_path = fig_dir / f"{args.stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white",
                edgecolor="none", pad_inches=0.05)
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight", facecolor="white",
                edgecolor="none", pad_inches=0.05)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
