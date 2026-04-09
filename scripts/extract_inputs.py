#!/usr/bin/env python3
"""Extract RELI input files from rMATS SE.MATS.JCEC.txt output.

Creates 8 .snp query files (SKIP/INCL x AltEX/DNintr/UPintr/merged)
and 4 BG region files for null model building.

Convention: IncLevelDifference (ILD) = Sample1 - Sample2.
Default assumes Sample1=Treatment, Sample2=Control:
  ILD > 0 => INCL (more included in treatment)
  ILD < 0 => SKIP (more skipped in treatment)
Use --swap if Sample1=Control, Sample2=Treatment.
"""

import argparse
import csv
import os
import random


def parse_args():
    p = argparse.ArgumentParser(description="Extract RELI inputs from rMATS SE output.")
    p.add_argument("se_file", help="Path to SE.MATS.JCEC.txt")
    p.add_argument("output_dir", help="Output directory for .snp and .bed files")
    p.add_argument("--dpsi", type=float, default=0.1, help="Min |dPSI| for sig events (default: 0.1)")
    p.add_argument("--fdr", type=float, default=0.05, help="Max FDR for sig events (default: 0.05)")
    p.add_argument("--pvalue", type=float, default=1.0, help="Max PValue for sig events (default: 1.0, no filter)")
    p.add_argument("--bg-fdr", type=float, default=0.5, help="Min FDR for BG events (default: 0.5)")
    p.add_argument("--bg-dpsi", type=float, default=0.05, help="Max |dPSI| for BG events (default: 0.05)")
    p.add_argument("--max-bg", type=int, default=5000, help="Max BG events to sample (default: 5000)")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument("--swap", action="store_true", help="Flip sign convention (Sample1=Control)")
    return p.parse_args()


def parse_rmats_se(filepath, has_pvalue):
    events = []
    with open(filepath, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            event = {
                "chr": row["chr"],
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


def get_regions(event):
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


def write_bed4(records, region_type, output_path, prefix="E"):
    with open(output_path, "w", newline="\n") as f:
        for i, (_, regions) in enumerate(records, 1):
            chrom, start, end = regions[region_type]
            f.write(f"{chrom}\t{start}\t{end}\t{prefix}{i}\n")
    print(f"  {os.path.basename(output_path)}: {len(records)} entries")


def main():
    args = parse_args()
    random.seed(args.seed)

    use_pvalue = args.pvalue < 1.0
    events = parse_rmats_se(args.se_file, use_pvalue)
    print(f"Total SE events: {len(events)}")

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

    print(f"SKIP: {len(skip_events)}  INCL: {len(incl_events)}  BG: {len(bg_events)} (of {len(bg_candidates)} candidates)")

    skip_wr = [(e, get_regions(e)) for e in skip_events]
    incl_wr = [(e, get_regions(e)) for e in incl_events]
    bg_wr = [(e, get_regions(e)) for e in bg_events]

    os.makedirs(args.output_dir, exist_ok=True)

    for direction, records in [("SKIP", skip_wr), ("INCL", incl_wr)]:
        for region in ["AltEX", "DNintr", "UPintr", "merged"]:
            write_bed4(records, region, os.path.join(args.output_dir, f"{direction}_{region}.snp"))

    for region in ["AltEX", "DNintr", "UPintr", "merged"]:
        write_bed4(bg_wr, region, os.path.join(args.output_dir, f"BG_{region}.bed"), prefix="BG")

    print(f"Done. Output: {args.output_dir}")


if __name__ == "__main__":
    main()
