#!/bin/bash
# Run RELI for all 8 queries against CLIP targets. Resume-safe.

set -e

usage() {
    echo "Usage: $0 <input_dir> <output_dir> <clip_index> <peaks_dir> <genome_build> [--threads N] [--reps N] [--corr N]"
    exit 1
}

if [ $# -lt 5 ]; then usage; fi

INPUT_DIR="$1"
OUTPUT_DIR="$2"
CLIP_INDEX="$3"
PEAKS_DIR="$4"
GENOME_BUILD="$5"
shift 5

THREADS=16
REPS=2000
CORR=0  # auto-detect from index if not overridden

while [ $# -gt 0 ]; do
    case "$1" in
        --threads) THREADS="$2"; shift 2 ;;
        --reps)    REPS="$2";    shift 2 ;;
        --corr)    CORR="$2";    shift 2 ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

mkdir -p "$OUTPUT_DIR"
tail -n +2 "$CLIP_INDEX" | cut -f1 > "$OUTPUT_DIR/.targets.txt"
NTARGETS=$(wc -l < "$OUTPUT_DIR/.targets.txt")
echo "Found $NTARGETS CLIP targets"

# Auto-detect correction factor from library size if not overridden
if [ "$CORR" -eq 0 ]; then
    CORR=$NTARGETS
    echo "Bonferroni correction set to $CORR (auto from index)"
fi

QUERIES="SKIP_AltEX SKIP_DNintr SKIP_UPintr SKIP_merged INCL_AltEX INCL_DNintr INCL_UPintr INCL_merged"

for QUERY in $QUERIES; do
    echo ""
    echo "=========================================="
    echo "Running RELI for query: $QUERY ($(date))"
    echo "=========================================="

    case "$QUERY" in
        *_AltEX)  NULL="Null_Model_BG_AltEX" ;;
        *_DNintr) NULL="Null_Model_BG_DNintr" ;;
        *_UPintr) NULL="Null_Model_BG_UPintr" ;;
        *_merged) NULL="Null_Model_BG_merged" ;;
    esac

    SNP_FILE="$INPUT_DIR/${QUERY}.snp"
    NULL_FILE="$INPUT_DIR/${NULL}"
    OUT_DIR="$OUTPUT_DIR/${QUERY}"

    if [ ! -f "$SNP_FILE" ]; then
        echo "ERROR: SNP file not found: $SNP_FILE"
        continue
    fi
    if [ ! -f "$NULL_FILE" ]; then
        echo "ERROR: Null model not found: $NULL_FILE"
        continue
    fi

    mkdir -p "$OUT_DIR"

    > $OUTPUT_DIR/.remaining.txt
    while IFS= read -r TARGET; do
        if [ ! -f "${OUT_DIR}/${TARGET}.RELI.stats" ]; then
            echo "$TARGET" >> $OUTPUT_DIR/.remaining.txt
        fi
    done < $OUTPUT_DIR/.targets.txt
    NREMAINING=$(wc -l < $OUTPUT_DIR/.remaining.txt)
    echo "Remaining targets: $NREMAINING of $NTARGETS"

    DBSNP_FILE="$INPUT_DIR/dummy_dbsnp"

    run_one_target() {
        local TARGET="$1"
        local SNP="$2"
        local NULL="$3"
        local OUTD="$4"
        local PHENO="$5"
        local IDX="$6"
        local PEAKS="$7"
        local BUILD="$8"
        local REPS="$9"
        local CORR="${10}"
        local DBSNP="${11}"
        /reli/RELI \
            -snp "$SNP" \
            -null "$NULL" \
            -index "$IDX" \
            -data "$PEAKS" \
            -target "$TARGET" \
            -build "$BUILD" \
            -dbsnp "$DBSNP" \
            -out "$OUTD" \
            -rep "$REPS" \
            -corr "$CORR" \
            -phenotype "$PHENO" 2>>"$OUTD/reli_stderr.log" || true
    }
    export -f run_one_target

    cat "$OUTPUT_DIR/.remaining.txt" | xargs -P "$THREADS" -I {} bash -c \
        "run_one_target '{}' '$SNP_FILE' '$NULL_FILE' '$OUT_DIR' '$QUERY' '$CLIP_INDEX' '$PEAKS_DIR' '$GENOME_BUILD' '$REPS' '$CORR' '$DBSNP_FILE'" || true

    echo "Completed: $QUERY at $(date)"

    RESULT_FILE="$OUTPUT_DIR/${QUERY}_results.tsv"
    FIRST=1
    for STATS_FILE in "$OUT_DIR"/*.RELI.stats; do
        [ -f "$STATS_FILE" ] || { echo "WARNING: No stats files for $QUERY"; break; }
        if [ "$FIRST" -eq 1 ]; then
            head -1 "$STATS_FILE" > "$RESULT_FILE"
            FIRST=0
        fi
        tail -n +2 "$STATS_FILE" >> "$RESULT_FILE"
    done

    if [ -f "$RESULT_FILE" ]; then
        NRESULTS=$(tail -n +2 "$RESULT_FILE" | wc -l)
        echo "Collected $NRESULTS results into $RESULT_FILE"
    fi
done

# Clean up temp files
rm -f "$OUTPUT_DIR/.targets.txt" "$OUTPUT_DIR/.remaining.txt"

echo ""
echo "=========================================="
echo "All RELI queries complete at $(date)"
echo "=========================================="
