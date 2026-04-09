#!/usr/bin/env python3
"""Extract RELI input files from DESeq2 differential expression results.

Parses GENCODE GTF to extract per-gene sub-region coordinates (5'UTR, CDS,
intron, 3'UTR), splits DE genes into UP and DOWN by log2FC direction, and
produces RELI-compatible .snp query files and region-matched null models
from non-DE genes.

Output files (in --output-dir):
  Query (8):
    UP_5UTR.snp, UP_CDS.snp, UP_intron.snp, UP_3UTR.snp
    DOWN_5UTR.snp, DOWN_CDS.snp, DOWN_intron.snp, DOWN_3UTR.snp

  Null model BED (4):
    BG_5UTR.bed, BG_CDS.bed, BG_intron.bed, BG_3UTR.bed

  Per-base null model (4):
    Null_Model_BG_5UTR, Null_Model_BG_CDS, Null_Model_BG_intron, Null_Model_BG_3UTR

  Gene lists:
    UP_genes.txt, DOWN_genes.txt, BG_genes.txt

  Misc:
    dummy_dbsnp
"""

import argparse
import collections
import csv
import gzip
import os
import random
import re
import sys


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STANDARD_CHROMS = {f"chr{i}" for i in range(1, 23)} | {"chrX", "chrY"}

REGIONS = ["5UTR", "CDS", "intron", "3UTR"]


# ---------------------------------------------------------------------------
# GTF Parsing
# ---------------------------------------------------------------------------

def parse_gtf_attributes(attr_string):
    """Parse GTF attribute string into a dict."""
    attrs = {}
    for match in re.finditer(r'(\w+)\s+"([^"]*)"', attr_string):
        attrs[match.group(1)] = match.group(2)
    return attrs


def parse_gencode_gtf(gtf_path):
    """Parse GENCODE GTF and return per-transcript feature intervals.

    Returns:
        transcripts: dict mapping transcript_id -> {
            'gene_name': str,
            'gene_id': str,
            'chrom': str,
            'strand': str,
            'tx_start': int,
            'tx_end': int,
            'exons': [(start, end), ...],
            'cds': [(start, end), ...],
            'utr': [(start, end), ...],
            'tags': set of str,
        }
    """
    opener = gzip.open if gtf_path.endswith(".gz") else open
    transcripts = {}
    n_lines = 0

    with opener(gtf_path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            n_lines += 1
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9:
                continue

            chrom = fields[0]
            feature = fields[2]
            start = int(fields[3]) - 1  # GTF 1-based -> 0-based
            end = int(fields[4])        # GTF inclusive end -> half-open
            strand = fields[6]
            attrs = parse_gtf_attributes(fields[8])

            if chrom not in STANDARD_CHROMS:
                continue

            tid = attrs.get("transcript_id", "")
            if not tid:
                continue

            if feature == "transcript":
                gene_name = attrs.get("gene_name", "")
                gene_id = attrs.get("gene_id", "")
                gene_type = attrs.get("gene_type", attrs.get("gene_biotype", ""))
                transcript_type = attrs.get("transcript_type",
                                            attrs.get("transcript_biotype", ""))
                # Only protein-coding transcripts have meaningful sub-regions
                if gene_type != "protein_coding" or transcript_type != "protein_coding":
                    continue
                tags = set()
                # Collect all tag values from the attribute string
                for t in re.findall(r'tag\s+"([^"]*)"', fields[8]):
                    tags.add(t)
                transcripts[tid] = {
                    "gene_name": gene_name,
                    "gene_id": gene_id,
                    "chrom": chrom,
                    "strand": strand,
                    "tx_start": start,
                    "tx_end": end,
                    "exons": [],
                    "cds": [],
                    "utr": [],
                    "tags": tags,
                }
            elif feature == "exon" and tid in transcripts:
                transcripts[tid]["exons"].append((start, end))
            elif feature == "CDS" and tid in transcripts:
                transcripts[tid]["cds"].append((start, end))
            elif feature == "UTR" and tid in transcripts:
                transcripts[tid]["utr"].append((start, end))

    print(f"Parsed {n_lines:,} GTF lines, {len(transcripts):,} protein-coding transcripts")
    return transcripts


# ---------------------------------------------------------------------------
# Sub-region Extraction
# ---------------------------------------------------------------------------

def merge_intervals(intervals):
    """Merge overlapping/adjacent intervals. Input: list of (start, end)."""
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        if s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def subtract_intervals(base, subtract):
    """Subtract a set of intervals from base intervals.

    Both inputs should be sorted, non-overlapping lists of (start, end).
    Returns sorted list of remaining intervals.
    """
    result = []
    si = 0
    for bs, be in base:
        pos = bs
        while si < len(subtract) and subtract[si][1] <= pos:
            si += 1
        j = si
        while j < len(subtract) and subtract[j][0] < be:
            ss, se = subtract[j]
            if ss > pos:
                result.append((pos, min(ss, be)))
            pos = max(pos, se)
            j += 1
        if pos < be:
            result.append((pos, be))
    return result


def extract_subregions(tx):
    """Extract 5'UTR, CDS, intron, and 3'UTR intervals from a transcript.

    Returns dict with keys '5UTR', 'CDS', 'intron', '3UTR', each a list
    of (start, end) intervals (0-based half-open, merged and sorted).
    """
    cds_intervals = sorted(tx["cds"])
    utr_intervals = sorted(tx["utr"])
    exon_intervals = merge_intervals(sorted(tx["exons"]))
    strand = tx["strand"]

    regions = {"5UTR": [], "CDS": [], "intron": [], "3UTR": []}

    # CDS
    regions["CDS"] = merge_intervals(cds_intervals) if cds_intervals else []

    # Split UTR into 5' and 3' based on strand and CDS position
    if cds_intervals and utr_intervals:
        if strand == "+":
            cds_min = min(s for s, _ in cds_intervals)
            cds_max = max(e for _, e in cds_intervals)
            regions["5UTR"] = merge_intervals(
                [(s, e) for s, e in utr_intervals if e <= cds_min])
            regions["3UTR"] = merge_intervals(
                [(s, e) for s, e in utr_intervals if s >= cds_max])
        else:  # minus strand
            cds_min = min(s for s, _ in cds_intervals)
            cds_max = max(e for _, e in cds_intervals)
            # On minus strand, 3'UTR is upstream (lower coords), 5'UTR is downstream
            regions["3UTR"] = merge_intervals(
                [(s, e) for s, e in utr_intervals if e <= cds_min])
            regions["5UTR"] = merge_intervals(
                [(s, e) for s, e in utr_intervals if s >= cds_max])

    # Introns = gene body (tx_start to tx_end) minus exons
    gene_body = [(tx["tx_start"], tx["tx_end"])]
    regions["intron"] = subtract_intervals(gene_body, exon_intervals)

    return regions


def select_best_transcript(transcript_group):
    """Select the best transcript per gene.

    Priority:
    1. appris_principal_1 tagged
    2. Longest total exonic length (proxy for most complete annotation)
    """
    best = None
    best_score = (-1, -1)

    for tid, tx in transcript_group:
        exon_len = sum(e - s for s, e in tx["exons"]) if tx["exons"] else 0
        has_appris = 1 if "appris_principal_1" in tx["tags"] else 0
        score = (has_appris, exon_len)
        if score > best_score:
            best_score = score
            best = (tid, tx)

    return best


def get_gene_subregions(transcripts):
    """For each gene, pick the best transcript and extract sub-region intervals.

    Returns:
        dict: gene_name -> {
            'chrom': str,
            'strand': str,
            '5UTR': [(start, end), ...],
            'CDS': [(start, end), ...],
            'intron': [(start, end), ...],
            '3UTR': [(start, end), ...],
        }
    """
    # Group transcripts by gene_name
    gene_txs = collections.defaultdict(list)
    for tid, tx in transcripts.items():
        if tx["cds"]:  # must have CDS for meaningful sub-regions
            gene_txs[tx["gene_name"]].append((tid, tx))

    gene_regions = {}
    for gene_name, group in gene_txs.items():
        best = select_best_transcript(group)
        if best is None:
            continue
        tid, tx = best
        regions = extract_subregions(tx)
        gene_regions[gene_name] = {
            "chrom": tx["chrom"],
            "strand": tx["strand"],
        }
        for r in REGIONS:
            gene_regions[gene_name][r] = regions[r]

    # Report stats
    stats = {}
    for r in REGIONS:
        genes_with_r = sum(1 for g in gene_regions.values() if g[r])
        total_bp = sum(sum(e - s for s, e in g[r]) for g in gene_regions.values())
        stats[r] = (genes_with_r, total_bp)
        print(f"  {r}: {genes_with_r:,} genes, {total_bp:,} bp total")

    print(f"Total genes with CDS annotation: {len(gene_regions):,}")
    return gene_regions


# ---------------------------------------------------------------------------
# DESeq2 Parsing
# ---------------------------------------------------------------------------

def load_deseq2(filepath, gene_col, log2fc_col, padj_col, basemean_col):
    """Load DESeq2 results from TSV/CSV.

    Returns list of dicts with keys: gene, log2fc, padj, basemean.
    """
    # Detect delimiter
    with open(filepath, "r") as fh:
        first_line = fh.readline()

    delimiter = "\t" if "\t" in first_line else ","

    records = []
    skipped_na = 0
    skipped_parse = 0

    with open(filepath, "r") as fh:
        reader = csv.DictReader(fh, delimiter=delimiter)

        # Validate columns exist
        if gene_col not in reader.fieldnames:
            # Try common alternatives
            alternatives = ["gene_symbol", "gene_name", "Gene", "Symbol",
                            "GeneSymbol", "SYMBOL", "external_gene_name"]
            found = None
            for alt in alternatives:
                if alt in reader.fieldnames:
                    found = alt
                    break
            if found:
                print(f"  Warning: '{gene_col}' not found, using '{found}'")
                gene_col = found
            else:
                print(f"ERROR: Gene column '{gene_col}' not found in header: "
                      f"{reader.fieldnames}", file=sys.stderr)
                sys.exit(1)

        for col_name in [log2fc_col, padj_col, basemean_col]:
            if col_name not in reader.fieldnames:
                print(f"ERROR: Column '{col_name}' not found in header: "
                      f"{reader.fieldnames}", file=sys.stderr)
                sys.exit(1)

        for row in reader:
            gene = row[gene_col].strip()
            if not gene or gene == "NA":
                skipped_na += 1
                continue

            try:
                l2fc_str = row[log2fc_col].strip()
                padj_str = row[padj_col].strip()
                bm_str = row[basemean_col].strip()

                if any(v in ("NA", "", "nan", "NaN") for v in [l2fc_str, padj_str, bm_str]):
                    skipped_na += 1
                    continue

                records.append({
                    "gene": gene,
                    "log2fc": float(l2fc_str),
                    "padj": float(padj_str),
                    "basemean": float(bm_str),
                })
            except (ValueError, KeyError):
                skipped_parse += 1
                continue

    print(f"Loaded {len(records):,} genes from DESeq2 results "
          f"(skipped {skipped_na} NA, {skipped_parse} parse errors)")
    return records


# ---------------------------------------------------------------------------
# Output Writers
# ---------------------------------------------------------------------------

def tile_region_positions(intervals, spacing=100):
    """Tile intervals with evenly spaced positions.

    Returns list of genomic positions, one every `spacing` bases.
    """
    positions = []
    for start, end in intervals:
        for pos in range(start, end, spacing):
            positions.append(pos)
    # Always include at least one position if intervals exist
    if not positions and intervals:
        s, e = intervals[0]
        positions.append((s + e) // 2)
    return positions


def write_snp_file(gene_regions, gene_list, region, output_path, spacing=100):
    """Write .snp file with tiled positions for the given genes and region.

    Format: chr<TAB>position<TAB>position<TAB>gene_symbol
    """
    records = []
    genes_with_loci = 0
    for gene in sorted(gene_list):
        if gene not in gene_regions:
            continue
        info = gene_regions[gene]
        intervals = info[region]
        if not intervals:
            continue
        positions = tile_region_positions(intervals, spacing=spacing)
        if positions:
            genes_with_loci += 1
        for pos in positions:
            records.append((info["chrom"], pos, gene))

    with open(output_path, "w", newline="\n") as fh:
        for chrom, pos, sym in records:
            fh.write(f"{chrom}\t{pos}\t{pos}\t{sym}\n")

    per_gene = len(records) // max(genes_with_loci, 1)
    print(f"  {os.path.basename(output_path)}: {len(records):,} loci from "
          f"{genes_with_loci} genes (~{per_gene} per gene)")
    return len(records)


def write_null_bed(gene_regions, gene_list, region, output_path):
    """Write null model BED file for the given genes and region.

    Format: chr<TAB>start<TAB>end (BED3)
    """
    n_intervals = 0
    total_bases = 0

    with open(output_path, "w", newline="\n") as fh:
        for gene in sorted(gene_list):
            if gene not in gene_regions:
                continue
            info = gene_regions[gene]
            for start, end in info[region]:
                fh.write(f"{info['chrom']}\t{start}\t{end}\n")
                n_intervals += 1
                total_bases += end - start

    print(f"  {os.path.basename(output_path)}: {n_intervals:,} intervals, "
          f"{total_bases:,} bases from {len(gene_list)} genes")
    return n_intervals, total_bases


def load_chrom_cumsum(genome_build_path):
    """Load chromosome cumulative offsets from a genome build file (e.g., hg19.txt)."""
    offsets = {}
    cumsum = 0
    with open(genome_build_path) as fh:
        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                offsets[parts[0]] = cumsum
                cumsum += int(parts[1])
    return offsets


def expand_bed_to_null_model(bed_path, null_model_path, chrom_offsets):
    """Expand BED intervals to per-base cumulative genome positions, sorted.

    Each position = chrom_cumulative_offset + genomic_coordinate.
    First line is 0 (header — both C++ and GPU backends skip line 1).
    """
    positions = []
    with open(bed_path, "r") as fh:
        for line in fh:
            fields = line.strip().split("\t")
            if len(fields) < 3:
                continue
            chrom = fields[0]
            start = int(fields[1])
            end = int(fields[2])
            if chrom not in chrom_offsets:
                continue
            offset = chrom_offsets[chrom]
            for pos in range(start, end):
                positions.append(offset + pos)

    positions.sort()

    with open(null_model_path, "w", newline="\n") as fh:
        fh.write("0\n")  # header line (skipped by RELI)
        for pos in positions:
            fh.write(f"{pos}\n")

    print(f"  {os.path.basename(null_model_path)}: {len(positions):,} cumulative positions")
    return len(positions)


def write_gene_list(gene_list, output_path):
    """Write gene symbols to a text file, one per line."""
    with open(output_path, "w", newline="\n") as fh:
        for gene in sorted(gene_list):
            fh.write(f"{gene}\n")


def write_dummy_dbsnp(output_path):
    """Write dummy dbSNP file required by RELI."""
    with open(output_path, "w", newline="\n") as fh:
        fh.write("dummy\t0\tdummy\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Extract RELI inputs from DESeq2 results, split by gene sub-region.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required inputs
    p.add_argument("--deseq2", required=True,
                   help="DESeq2 results file (TSV or CSV)")
    p.add_argument("--gtf", required=True,
                   help="GENCODE GTF (v19 for hg19, v44 for hg38; gzipped OK)")
    p.add_argument("--output-dir", required=True,
                   help="Output directory")

    # Column names
    p.add_argument("--gene-col", default="gene_symbol",
                   help="Gene symbol column name (default: gene_symbol)")
    p.add_argument("--log2fc-col", default="log2FoldChange",
                   help="log2FC column name (default: log2FoldChange)")
    p.add_argument("--padj-col", default="padj",
                   help="Adjusted p-value column name (default: padj)")
    p.add_argument("--basemean-col", default="baseMean",
                   help="Base mean column name (default: baseMean)")

    # Filtering thresholds
    p.add_argument("--min-basemean", type=float, default=100,
                   help="Minimum baseMean for all genes (default: 100)")
    p.add_argument("--padj-threshold", type=float, default=0.05,
                   help="Adjusted p-value threshold for DE (default: 0.05)")
    p.add_argument("--min-log2fc", type=float, default=0.4,
                   help="Minimum |log2FC| for DE genes (default: 0.4)")

    # Tiling and null model
    p.add_argument("--spacing", type=int, default=100,
                   help="Spacing between tiled query positions (default: 100bp)")
    p.add_argument("--max-null-genes", type=int, default=5000,
                   help="Max non-DE genes for null model (default: 5000)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for null gene sampling (default: 42)")
    p.add_argument("--genome-build", required=True,
                   help="Genome build file (e.g., hg19.txt) with chrom sizes for cumulative position conversion")

    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    # === Step 1: Parse GTF ===
    print("=" * 60)
    print("Step 1: Parsing GENCODE GTF")
    print("=" * 60)
    transcripts = parse_gencode_gtf(args.gtf)

    # === Step 2: Extract sub-regions per gene ===
    print("\nStep 2: Extracting sub-regions per gene")
    print("-" * 40)
    gene_regions = get_gene_subregions(transcripts)
    del transcripts  # free memory

    all_annotated_genes = set(gene_regions.keys())

    # === Step 3: Load and filter DESeq2 results ===
    print("\nStep 3: Loading DESeq2 results")
    print("-" * 40)
    deseq2_records = load_deseq2(
        args.deseq2, args.gene_col, args.log2fc_col,
        args.padj_col, args.basemean_col)

    # Expression filter
    expressed = [r for r in deseq2_records if r["basemean"] >= args.min_basemean]
    print(f"  After baseMean >= {args.min_basemean}: {len(expressed):,} genes")

    # Match to GENCODE
    matched = [r for r in expressed if r["gene"] in all_annotated_genes]
    unmatched = [r for r in expressed if r["gene"] not in all_annotated_genes]
    print(f"  Matched to GENCODE: {len(matched):,} / {len(expressed):,} "
          f"({len(unmatched):,} unmatched)")
    if unmatched:
        sample = sorted(set(r["gene"] for r in unmatched))[:10]
        print(f"  Unmatched examples: {', '.join(sample)}")

    # DE genes
    de_genes = [r for r in matched
                if r["padj"] <= args.padj_threshold
                and abs(r["log2fc"]) >= args.min_log2fc]
    print(f"  DE genes (padj <= {args.padj_threshold}, |log2FC| >= {args.min_log2fc}): "
          f"{len(de_genes):,}")

    up_genes = set(r["gene"] for r in de_genes if r["log2fc"] > 0)
    down_genes = set(r["gene"] for r in de_genes if r["log2fc"] < 0)
    print(f"    UP:   {len(up_genes):,}")
    print(f"    DOWN: {len(down_genes):,}")

    # BG (null) genes: not DE, very stable expression
    all_matched_genes = set(r["gene"] for r in matched)
    de_gene_set = up_genes | down_genes
    bg_candidates = [r for r in matched
                     if r["gene"] not in de_gene_set
                     and r["padj"] > 0.5
                     and abs(r["log2fc"]) < 0.1]
    bg_gene_set = set(r["gene"] for r in bg_candidates)
    print(f"\n  BG candidates (padj > 0.5, |log2FC| < 0.1): {len(bg_gene_set):,}")

    if len(bg_gene_set) > args.max_null_genes:
        bg_gene_set = set(random.sample(sorted(bg_gene_set), args.max_null_genes))
        print(f"  Sampled {args.max_null_genes:,} for null model")
    else:
        print(f"  Using all {len(bg_gene_set):,} for null model")

    if len(bg_gene_set) < 100:
        print(f"  WARNING: Only {len(bg_gene_set)} BG genes. Consider relaxing "
              f"--min-basemean or null model thresholds.", file=sys.stderr)

    # === Step 4: Write output files ===
    print("\n" + "=" * 60)
    print("Step 4: Writing output files")
    print("=" * 60)
    os.makedirs(args.output_dir, exist_ok=True)

    # Gene lists
    write_gene_list(up_genes, os.path.join(args.output_dir, "UP_genes.txt"))
    write_gene_list(down_genes, os.path.join(args.output_dir, "DOWN_genes.txt"))
    write_gene_list(bg_gene_set, os.path.join(args.output_dir, "BG_genes.txt"))

    # Query .snp files (8: UP/DOWN x 4 regions)
    print("\nQuery .snp files:")
    snp_counts = {}
    for direction, gene_set in [("UP", up_genes), ("DOWN", down_genes)]:
        for region in REGIONS:
            label = f"{direction}_{region}"
            path = os.path.join(args.output_dir, f"{label}.snp")
            n = write_snp_file(gene_regions, gene_set, region, path,
                               spacing=args.spacing)
            snp_counts[label] = n

    # Null model BED + per-base expansion (4 regions)
    print("\nNull model files:")
    print(f"  Loading genome build: {args.genome_build}")
    chrom_offsets = load_chrom_cumsum(args.genome_build)
    null_counts = {}
    for region in REGIONS:
        bed_path = os.path.join(args.output_dir, f"BG_{region}.bed")
        null_path = os.path.join(args.output_dir, f"Null_Model_BG_{region}")
        n_int, n_bases = write_null_bed(gene_regions, bg_gene_set, region, bed_path)

        print(f"  Expanding {os.path.basename(bed_path)} to per-base null model...")
        n_positions = expand_bed_to_null_model(bed_path, null_path, chrom_offsets)
        null_counts[region] = (n_int, n_bases, n_positions)

    # Dummy dbSNP
    write_dummy_dbsnp(os.path.join(args.output_dir, "dummy_dbsnp"))
    print("\n  dummy_dbsnp: written")

    # === Summary ===
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  DESeq2 input:         {os.path.basename(args.deseq2)}")
    print(f"  GENCODE GTF:          {os.path.basename(args.gtf)}")
    print(f"  Annotated genes:      {len(all_annotated_genes):,}")
    print(f"  Matched to DESeq2:    {len(all_matched_genes):,}")
    print(f"  UP genes:             {len(up_genes):,}")
    print(f"  DOWN genes:           {len(down_genes):,}")
    print(f"  BG genes:             {len(bg_gene_set):,}")
    print()

    print(f"  {'Query':<20} {'Loci':>10}")
    print(f"  {'-'*20} {'-'*10}")
    for direction in ["UP", "DOWN"]:
        for region in REGIONS:
            label = f"{direction}_{region}"
            print(f"  {label:<20} {snp_counts[label]:>10,}")

    print()
    print(f"  {'Null Model':<20} {'Intervals':>10} {'Bases':>15} {'Positions':>15}")
    print(f"  {'-'*20} {'-'*10} {'-'*15} {'-'*15}")
    for region in REGIONS:
        n_int, n_bases, n_pos = null_counts[region]
        print(f"  BG_{region:<16} {n_int:>10,} {n_bases:>15,} {n_pos:>15,}")

    print(f"\nOutput directory: {args.output_dir}")


if __name__ == "__main__":
    main()
