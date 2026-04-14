#!/usr/bin/env python3
"""Unified RELI input extractor: splicing, DESeq2, and database modes.

Modes:
  splicing  — Extract from rMATS SE.MATS.JCEC.txt (SKIP/INCL x AltEX/DNintr/UPintr/merged)
  deseq2   — Extract from DESeq2 + GENCODE GTF (UP/DOWN x 5UTR/CDS/intron/3UTR)
  database — Extract from user-provided query/background BED files

All modes produce:
  - .snp query files (BED4: chr, start, end, name)
  - BG .bed files for null model construction
  - Per-base cumulative-position null models (Null_Model_BG_*)
  - dummy_dbsnp
  - manifest.json

Usage:
  python extract_inputs.py --mode splicing --se-file SE.MATS.JCEC.txt \\
      --output-dir inputs/ --genome-build hg19.txt

  python extract_inputs.py --mode deseq2 --deseq2 results.csv --gtf gencode.gtf.gz \\
      --output-dir inputs/ --genome-build hg19.txt

  python extract_inputs.py --mode database --query-bed peaks.bed --bg-bed bg.bed \\
      --output-dir inputs/ --genome-build hg19.txt
"""

import argparse
import collections
import csv
import gzip
import json
import os
import random
import re
import sys
from datetime import datetime, timezone

from reli_utils import (
    STANDARD_CHROMS,
    SPLICING_REGIONS,
    SPLICING_DIRECTIONS,
    DESEQ2_REGIONS,
    DESEQ2_DIRECTIONS,
    load_chrom_offsets,
    expand_bed_to_cumulative_positions,
    write_null_model,
    validate_null_model,
    write_dummy_dbsnp,
)


# ============================================================================
# Splicing mode helpers
# ============================================================================

def parse_rmats_se(filepath, has_pvalue):
    """Parse rMATS SE.MATS.JCEC.txt into a list of event dicts."""
    events = []
    with open(filepath, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            chrom = row["chr"]
            if chrom not in STANDARD_CHROMS:
                continue
            event = {
                "chr": chrom,
                "strand": row["strand"],
                "exonStart_0base": int(row["exonStart_0base"]),
                "exonEnd": int(row["exonEnd"]),
                "FDR": float(row["FDR"]),
                "IncLevelDifference": float(row["IncLevelDifference"]),
            }
            if has_pvalue and "PValue" in row:
                event["PValue"] = float(row["PValue"])
            events.append(event)
    return events


def get_splicing_regions(event):
    """Compute AltEX, UPintr, DNintr, merged regions for a splicing event.

    Strand-aware: UPintr is upstream of the exon (5' side), DNintr is
    downstream (3' side). For minus-strand genes, genomic coordinates
    are flipped relative to transcription direction.
    """
    chrom = event["chr"]
    strand = event["strand"]
    exon_start = event["exonStart_0base"]
    exon_end = event["exonEnd"]

    altex = (chrom, exon_start, exon_end)

    if strand == "+":
        upintr = (chrom, max(0, exon_start - 250), exon_start)
        dnintr = (chrom, exon_end, exon_end + 250)
    else:
        upintr = (chrom, exon_end, exon_end + 250)
        dnintr = (chrom, max(0, exon_start - 250), exon_start)

    all_starts = [altex[1], upintr[1], dnintr[1]]
    all_ends = [altex[2], upintr[2], dnintr[2]]
    merged = (chrom, min(all_starts), max(all_ends))

    return {"AltEX": altex, "UPintr": upintr, "DNintr": dnintr, "merged": merged}


def extract_splicing(args):
    """Extract RELI inputs from rMATS SE output.

    Returns dict:
        queries: list of {name, direction, region, records: [(chrom, start, end, name), ...]}
        bg_groups: list of {region, records: [(chrom, start, end, name), ...]}
        metadata: dict with event counts
    """
    use_pvalue = args.pvalue < 1.0
    events = parse_rmats_se(args.se_file, use_pvalue)
    print(f"Total SE events (standard chroms): {len(events)}")

    sign = -1.0 if args.swap else 1.0

    skip_events, incl_events, bg_candidates = [], [], []
    for e in events:
        fdr = e["FDR"]
        ild = e["IncLevelDifference"] * sign
        pval = e.get("PValue", 0.0)

        passes_sig = fdr <= args.fdr and abs(ild) >= args.dpsi
        if use_pvalue:
            passes_sig = passes_sig and pval <= args.pvalue

        if passes_sig and ild < 0:
            skip_events.append(e)
        elif passes_sig and ild > 0:
            incl_events.append(e)
        elif fdr >= args.bg_fdr and abs(ild) <= args.bg_dpsi:
            bg_candidates.append(e)

    if len(bg_candidates) > args.max_bg:
        bg_events = random.sample(bg_candidates, args.max_bg)
    else:
        bg_events = bg_candidates

    print(f"SKIP: {len(skip_events)}  INCL: {len(incl_events)}  "
          f"BG: {len(bg_events)} (of {len(bg_candidates)} candidates)")

    # Build region records for each event set
    skip_wr = [(e, get_splicing_regions(e)) for e in skip_events]
    incl_wr = [(e, get_splicing_regions(e)) for e in incl_events]
    bg_wr = [(e, get_splicing_regions(e)) for e in bg_events]

    # Build query list
    queries = []
    for direction, records in [("SKIP", skip_wr), ("INCL", incl_wr)]:
        for region in SPLICING_REGIONS:
            loci = []
            for i, (_, regions) in enumerate(records, 1):
                chrom, start, end = regions[region]
                loci.append((chrom, start, end, f"E{i}"))
            queries.append({
                "name": f"{direction}_{region}",
                "direction": direction,
                "region": region,
                "records": loci,
            })

    # Build BG groups
    bg_groups = []
    for region in SPLICING_REGIONS:
        loci = []
        for i, (_, regions) in enumerate(bg_wr, 1):
            chrom, start, end = regions[region]
            loci.append((chrom, start, end, f"BG{i}"))
        bg_groups.append({
            "region": region,
            "records": loci,
        })

    metadata = {
        "se_file": os.path.basename(args.se_file),
        "total_events": len(events),
        "skip_events": len(skip_events),
        "incl_events": len(incl_events),
        "bg_events": len(bg_events),
        "bg_candidates": len(bg_candidates),
        "dpsi": args.dpsi,
        "fdr": args.fdr,
        "pvalue": args.pvalue,
        "bg_fdr": args.bg_fdr,
        "bg_dpsi": args.bg_dpsi,
        "swap": args.swap,
    }

    return {"queries": queries, "bg_groups": bg_groups, "metadata": metadata}


# ============================================================================
# DESeq2 mode helpers
# ============================================================================

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
        for r in DESEQ2_REGIONS:
            gene_regions[gene_name][r] = regions[r]

    # Report stats
    for r in DESEQ2_REGIONS:
        genes_with_r = sum(1 for g in gene_regions.values() if g[r])
        total_bp = sum(sum(e - s for s, e in g[r]) for g in gene_regions.values())
        print(f"  {r}: {genes_with_r:,} genes, {total_bp:,} bp total")

    print(f"Total genes with CDS annotation: {len(gene_regions):,}")
    return gene_regions


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


def tile_deseq2_positions(intervals, spacing=100):
    """Tile intervals with evenly spaced positions for DESeq2 queries.

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


def extract_deseq2(args):
    """Extract RELI inputs from DESeq2 results + GENCODE GTF.

    Returns dict:
        queries: list of {name, direction, region, records: [(chrom, start, end, name), ...]}
        bg_groups: list of {region, records: [(chrom, start, end)], raw_intervals: [...]}
        gene_lists: {up: set, down: set, bg: set}
        metadata: dict
    """
    # Step 1: Parse GTF
    print("=" * 60)
    print("Step 1: Parsing GENCODE GTF")
    print("=" * 60)
    transcripts = parse_gencode_gtf(args.gtf)

    # Step 2: Extract sub-regions per gene
    print("\nStep 2: Extracting sub-regions per gene")
    print("-" * 40)
    gene_regions = get_gene_subregions(transcripts)
    del transcripts  # free memory

    all_annotated_genes = set(gene_regions.keys())

    # Step 3: Load and filter DESeq2 results
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
    de_gene_set = up_genes | down_genes
    bg_candidates = [r for r in matched
                     if r["gene"] not in de_gene_set
                     and r["padj"] > 0.5
                     and abs(r["log2fc"]) < 0.1]
    bg_gene_set = set(r["gene"] for r in bg_candidates)
    print(f"\n  BG candidates (padj > 0.5, |log2FC| < 0.1): {len(bg_gene_set):,}")

    if len(bg_gene_set) > args.max_bg:
        bg_gene_set = set(random.sample(sorted(bg_gene_set), args.max_bg))
        print(f"  Sampled {args.max_bg:,} for null model")
    else:
        print(f"  Using all {len(bg_gene_set):,} for null model")

    if len(bg_gene_set) < 100:
        print(f"  WARNING: Only {len(bg_gene_set)} BG genes. Consider relaxing "
              f"--min-basemean or null model thresholds.", file=sys.stderr)

    # Build query records
    queries = []
    for direction, gene_set in [("UP", up_genes), ("DOWN", down_genes)]:
        for region in DESEQ2_REGIONS:
            loci = []
            genes_with_loci = 0
            for gene in sorted(gene_set):
                if gene not in gene_regions:
                    continue
                info = gene_regions[gene]
                intervals = info[region]
                if not intervals:
                    continue
                positions = tile_deseq2_positions(intervals, spacing=args.spacing)
                if positions:
                    genes_with_loci += 1
                for pos in positions:
                    loci.append((info["chrom"], pos, pos, gene))

            per_gene = len(loci) // max(genes_with_loci, 1)
            label = f"{direction}_{region}"
            print(f"  {label}: {len(loci):,} loci from "
                  f"{genes_with_loci} genes (~{per_gene} per gene)")

            queries.append({
                "name": label,
                "direction": direction,
                "region": region,
                "records": loci,
            })

    # Build BG groups (full intervals for null model expansion)
    bg_groups = []
    for region in DESEQ2_REGIONS:
        intervals = []
        for gene in sorted(bg_gene_set):
            if gene not in gene_regions:
                continue
            info = gene_regions[gene]
            for start, end in info[region]:
                intervals.append((info["chrom"], start, end))
        bg_groups.append({
            "region": region,
            "records": [(c, s, e, f"BG{i}") for i, (c, s, e) in enumerate(intervals, 1)],
            "raw_intervals": intervals,
        })

    all_matched_genes = set(r["gene"] for r in matched)

    metadata = {
        "deseq2_file": os.path.basename(args.deseq2),
        "gtf_file": os.path.basename(args.gtf),
        "annotated_genes": len(all_annotated_genes),
        "matched_genes": len(all_matched_genes),
        "up_genes": len(up_genes),
        "down_genes": len(down_genes),
        "bg_genes": len(bg_gene_set),
        "min_basemean": args.min_basemean,
        "padj_threshold": args.padj_threshold,
        "min_log2fc": args.min_log2fc,
        "spacing": args.spacing,
    }

    return {
        "queries": queries,
        "bg_groups": bg_groups,
        "metadata": metadata,
        "gene_lists": {"up": up_genes, "down": down_genes, "bg": bg_gene_set},
    }


# ============================================================================
# Database mode helpers
# ============================================================================

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
                print(f"WARNING: skipping line {lineno} ({len(fields)} fields): {line[:80]}",
                      file=sys.stderr)
                continue
            chrom = fields[0]
            try:
                start = int(fields[1])
                end = int(fields[2])
            except ValueError:
                print(f"WARNING: skipping line {lineno} (non-integer coords): {line[:80]}",
                      file=sys.stderr)
                continue
            name = fields[3] if len(fields) > 3 else f"locus_{lineno}"
            if chrom in STANDARD_CHROMS and end > start:
                regions.append((chrom, start, end, name))
    return regions


def tile_database_regions(regions, spacing):
    """Tile each region into multiple loci spaced `spacing` bp apart.

    For regions smaller than `spacing`, a single midpoint locus is emitted.
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


def extract_database(args):
    """Extract RELI inputs from user-provided query/background BED files.

    Returns dict:
        queries: list of {name, direction, region, records: [(chrom, start, end, name), ...]}
        bg_groups: list of {region, records: [(chrom, start, end, name), ...]}
        metadata: dict
    """
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

    # Tile query (optional)
    if args.tile:
        print(f"Tiling query regions at {args.tile}bp spacing")
        query_loci = tile_database_regions(query, args.tile)
        print(f"  {len(query)} regions -> {len(query_loci)} tiled loci")
    else:
        query_loci = query

    # Subsample background (optional)
    if args.max_bg and len(bg) > args.max_bg:
        print(f"Subsampling background: {len(bg)} -> {args.max_bg}")
        bg = random.sample(bg, args.max_bg)

    prefix = args.prefix

    # Query size stats
    query_sizes = [end - start for _, start, end, _ in query]
    bg_sizes = [end - start for _, start, end, _ in bg]

    queries = [{
        "name": prefix,
        "direction": prefix,
        "region": prefix,
        "records": query_loci,
    }]

    bg_groups = [{
        "region": prefix,
        "records": bg,
    }]

    metadata = {
        "query_bed": os.path.basename(args.query_bed),
        "bg_bed": os.path.basename(args.bg_bed),
        "prefix": prefix,
        "tile_spacing": args.tile,
        "query_regions": len(query),
        "query_loci": len(query_loci),
        "query_size_median": sorted(query_sizes)[len(query_sizes) // 2],
        "query_size_min": min(query_sizes),
        "query_size_max": max(query_sizes),
        "bg_regions": len(bg),
        "bg_size_median": sorted(bg_sizes)[len(bg_sizes) // 2],
        "bg_size_min": min(bg_sizes),
        "bg_size_max": max(bg_sizes),
    }

    return {"queries": queries, "bg_groups": bg_groups, "metadata": metadata}


# ============================================================================
# Common output functions
# ============================================================================

def write_snp_file(records, output_path):
    """Write .snp query file in BED4 format: chr, start, end, name."""
    with open(output_path, "w", newline="\n") as fh:
        for chrom, start, end, name in records:
            fh.write(f"{chrom}\t{start}\t{end}\t{name}\n")
    return len(records)


def write_bg_bed(records, output_path):
    """Write background BED file (BED3 or BED4 depending on input)."""
    with open(output_path, "w", newline="\n") as fh:
        for rec in records:
            if len(rec) >= 4:
                chrom, start, end = rec[0], rec[1], rec[2]
            else:
                chrom, start, end = rec
            fh.write(f"{chrom}\t{start}\t{end}\n")
    return len(records)


def write_deseq2_bg_bed(intervals, output_path):
    """Write background BED3 from raw (chrom, start, end) intervals."""
    n_intervals = 0
    total_bases = 0
    with open(output_path, "w", newline="\n") as fh:
        for chrom, start, end in intervals:
            fh.write(f"{chrom}\t{start}\t{end}\n")
            n_intervals += 1
            total_bases += end - start
    return n_intervals, total_bases


def write_gene_list(gene_set, output_path):
    """Write gene symbols to a text file, one per line."""
    with open(output_path, "w", newline="\n") as fh:
        for gene in sorted(gene_set):
            fh.write(f"{gene}\n")


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Unified RELI input extractor: splicing, DESeq2, and database modes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # --- Shared args (ALL modes) ---
    p.add_argument("--mode", required=True, choices=["splicing", "deseq2", "database"],
                   help="Extraction mode: splicing, deseq2, or database")
    p.add_argument("--output-dir", required=True,
                   help="Output directory for all generated files")
    p.add_argument("--genome-build", required=True,
                   help="Genome build file (e.g., hg19.txt) with chrom sizes")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (default: 42)")
    p.add_argument("--max-bg", type=int, default=5000,
                   help="Max background events/genes to sample (default: 5000)")

    # --- Splicing args ---
    p.add_argument("--se-file",
                   help="[splicing] Path to SE.MATS.JCEC.txt")
    p.add_argument("--dpsi", type=float, default=0.1,
                   help="[splicing] Min |dPSI| for sig events (default: 0.1)")
    p.add_argument("--fdr", type=float, default=0.05,
                   help="[splicing] Max FDR for sig events (default: 0.05)")
    p.add_argument("--pvalue", type=float, default=1.0,
                   help="[splicing] Max PValue for sig events (default: 1.0, no filter)")
    p.add_argument("--bg-fdr", type=float, default=0.5,
                   help="[splicing] Min FDR for BG events (default: 0.5)")
    p.add_argument("--bg-dpsi", type=float, default=0.05,
                   help="[splicing] Max |dPSI| for BG events (default: 0.05)")
    p.add_argument("--swap", action="store_true",
                   help="[splicing] Flip sign convention (Sample1=Control)")

    # --- DESeq2 args ---
    p.add_argument("--deseq2",
                   help="[deseq2] DESeq2 results file (TSV or CSV)")
    p.add_argument("--gtf",
                   help="[deseq2] GENCODE GTF (v19 for hg19, v44 for hg38; gzipped OK)")
    p.add_argument("--gene-col", default="gene_symbol",
                   help="[deseq2] Gene symbol column name (default: gene_symbol)")
    p.add_argument("--log2fc-col", default="log2FoldChange",
                   help="[deseq2] log2FC column name (default: log2FoldChange)")
    p.add_argument("--padj-col", default="padj",
                   help="[deseq2] Adjusted p-value column name (default: padj)")
    p.add_argument("--basemean-col", default="baseMean",
                   help="[deseq2] Base mean column name (default: baseMean)")
    p.add_argument("--min-basemean", type=float, default=100,
                   help="[deseq2] Minimum baseMean for all genes (default: 100)")
    p.add_argument("--padj-threshold", type=float, default=0.05,
                   help="[deseq2] Adjusted p-value threshold for DE (default: 0.05)")
    p.add_argument("--min-log2fc", type=float, default=0.4,
                   help="[deseq2] Minimum |log2FC| for DE genes (default: 0.4)")
    p.add_argument("--spacing", type=int, default=100,
                   help="[deseq2] Spacing between tiled query positions (default: 100bp)")

    # --- Database args ---
    p.add_argument("--query-bed",
                   help="[database] BED file with query regions")
    p.add_argument("--bg-bed",
                   help="[database] BED file with background regions")
    p.add_argument("--prefix", default="DB",
                   help="[database] Output file prefix (default: DB)")
    p.add_argument("--tile", type=int, default=None,
                   help="[database] Tile query regions at this spacing in bp")

    return p.parse_args()


def validate_args(args):
    """Validate that mode-specific required args are present and files exist."""
    errors = []

    if args.mode == "splicing":
        if not args.se_file:
            errors.append("--se-file is required for --mode splicing")
        elif not os.path.isfile(args.se_file):
            errors.append(f"--se-file not found: {args.se_file}")

    elif args.mode == "deseq2":
        if not args.deseq2:
            errors.append("--deseq2 is required for --mode deseq2")
        elif not os.path.isfile(args.deseq2):
            errors.append(f"--deseq2 not found: {args.deseq2}")
        if not args.gtf:
            errors.append("--gtf is required for --mode deseq2")
        elif not os.path.isfile(args.gtf):
            errors.append(f"--gtf not found: {args.gtf}")

    elif args.mode == "database":
        if not args.query_bed:
            errors.append("--query-bed is required for --mode database")
        elif not os.path.isfile(args.query_bed):
            errors.append(f"--query-bed not found: {args.query_bed}")
        if not args.bg_bed:
            errors.append("--bg-bed is required for --mode database")
        elif not os.path.isfile(args.bg_bed):
            errors.append(f"--bg-bed not found: {args.bg_bed}")

    if not os.path.isfile(args.genome_build):
        errors.append(f"--genome-build not found: {args.genome_build}")

    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()
    validate_args(args)
    random.seed(args.seed)

    print("=" * 60)
    print(f"RBP-RELI Input Extractor  --  mode: {args.mode}")
    print("=" * 60)

    # --- Mode-specific extraction ---
    if args.mode == "splicing":
        result = extract_splicing(args)
    elif args.mode == "deseq2":
        result = extract_deseq2(args)
    elif args.mode == "database":
        result = extract_database(args)

    queries = result["queries"]
    bg_groups = result["bg_groups"]
    metadata = result["metadata"]

    # Guard: no queries
    total_loci = sum(len(q["records"]) for q in queries)
    if total_loci == 0:
        print("ERROR: No query loci generated. Check input data and thresholds.",
              file=sys.stderr)
        sys.exit(1)

    # --- Write outputs ---
    print("\n" + "=" * 60)
    print("Writing output files")
    print("=" * 60)
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Write .snp query files
    print("\nQuery .snp files:")
    snp_counts = {}
    for q in queries:
        snp_path = os.path.join(args.output_dir, f"{q['name']}.snp")
        n = write_snp_file(q["records"], snp_path)
        snp_counts[q["name"]] = n
        print(f"  {q['name']}.snp: {n:,} loci")

    # 2. Write BG .bed files
    print("\nBackground .bed files:")
    for bg in bg_groups:
        region = bg["region"]
        bed_path = os.path.join(args.output_dir, f"BG_{region}.bed")
        if "raw_intervals" in bg:
            # DESeq2 mode: write from raw intervals (chrom, start, end)
            n_int, n_bases = write_deseq2_bg_bed(bg["raw_intervals"], bed_path)
            print(f"  BG_{region}.bed: {n_int:,} intervals, {n_bases:,} bases")
        else:
            n = write_bg_bed(bg["records"], bed_path)
            print(f"  BG_{region}.bed: {n:,} regions")

    # 3. Build null models
    print("\nBuilding null models:")
    print(f"  Loading genome build: {args.genome_build}")
    chrom_offsets = load_chrom_offsets(args.genome_build)

    null_stats = {}
    for bg in bg_groups:
        region = bg["region"]
        bed_path = os.path.join(args.output_dir, f"BG_{region}.bed")
        null_path = os.path.join(args.output_dir, f"Null_Model_BG_{region}")

        print(f"  Expanding BG_{region}.bed to per-base null model...")
        positions = expand_bed_to_cumulative_positions(bed_path, chrom_offsets)
        write_null_model(positions, null_path)

        is_valid, summary = validate_null_model(positions, region, chrom_offsets)
        status = "OK" if is_valid else "FAIL"
        print(f"    Null_Model_BG_{region}: {len(positions):,} positions [{status}]")
        if not is_valid:
            print(f"    {summary}", file=sys.stderr)

        null_stats[region] = len(positions)

    # 4. Write dummy_dbsnp
    dbsnp_path = write_dummy_dbsnp(args.output_dir)
    print(f"\n  dummy_dbsnp: written")

    # 5. Write gene lists (DESeq2 mode only)
    if args.mode == "deseq2" and "gene_lists" in result:
        gene_lists = result["gene_lists"]
        write_gene_list(gene_lists["up"], os.path.join(args.output_dir, "UP_genes.txt"))
        write_gene_list(gene_lists["down"], os.path.join(args.output_dir, "DOWN_genes.txt"))
        write_gene_list(gene_lists["bg"], os.path.join(args.output_dir, "BG_genes.txt"))
        print("  Gene lists: UP_genes.txt, DOWN_genes.txt, BG_genes.txt")

    # 6. Write manifest.json
    manifest = {
        "version": 1,
        "mode": args.mode,
        "genome_build": os.path.basename(args.genome_build),
        "created": datetime.now(timezone.utc).isoformat(),
        "queries": [],
        "dummy_dbsnp": "dummy_dbsnp",
        "metadata": metadata,
    }
    for q in queries:
        region = q["region"]
        manifest["queries"].append({
            "name": q["name"],
            "snp_file": f"{q['name']}.snp",
            "null_model": f"Null_Model_BG_{region}",
            "region": region,
            "direction": q["direction"],
            "n_loci": len(q["records"]),
        })

    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w", newline="\n") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")
    print(f"  manifest.json: written ({len(manifest['queries'])} queries)")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    if args.mode == "splicing":
        print(f"  Input:     {metadata['se_file']}")
        print(f"  Events:    {metadata['total_events']:,} total")
        print(f"  SKIP:      {metadata['skip_events']:,}")
        print(f"  INCL:      {metadata['incl_events']:,}")
        print(f"  BG:        {metadata['bg_events']:,} (of {metadata['bg_candidates']:,})")
    elif args.mode == "deseq2":
        print(f"  DESeq2:    {metadata['deseq2_file']}")
        print(f"  GTF:       {metadata['gtf_file']}")
        print(f"  Annotated: {metadata['annotated_genes']:,} genes")
        print(f"  Matched:   {metadata['matched_genes']:,} genes")
        print(f"  UP:        {metadata['up_genes']:,} genes")
        print(f"  DOWN:      {metadata['down_genes']:,} genes")
        print(f"  BG:        {metadata['bg_genes']:,} genes")
    elif args.mode == "database":
        print(f"  Query:     {metadata['query_bed']}")
        print(f"  BG:        {metadata['bg_bed']}")
        print(f"  Prefix:    {metadata['prefix']}")
        print(f"  Query:     {metadata['query_regions']:,} regions -> {metadata['query_loci']:,} loci")
        print(f"  BG:        {metadata['bg_regions']:,} regions")

    print()
    print(f"  {'Query':<25} {'Loci':>10}")
    print(f"  {'-'*25} {'-'*10}")
    for q in queries:
        print(f"  {q['name']:<25} {snp_counts[q['name']]:>10,}")

    print()
    print(f"  {'Null Model':<25} {'Positions':>15}")
    print(f"  {'-'*25} {'-'*15}")
    for bg in bg_groups:
        region = bg["region"]
        print(f"  Null_Model_BG_{region:<10} {null_stats[region]:>15,}")

    print(f"\nOutput directory: {args.output_dir}")
    print(f"\nNext steps:")
    print(f"  Run GPU-RELI:")
    print(f"    python -m reli_turbo --input-dir {args.output_dir} --output-dir <output_dir> \\")
    print(f"      --index <CLIPseq.index> --data <peaks_dir> --build <hg19.txt> --reps 2000")
    print(f"  Note: Null models are already built (per-base cumulative positions).")


if __name__ == "__main__":
    main()
