# RBP-RELI

Permutation-based enrichment testing of RNA-binding protein CLIP-seq peaks at alternatively spliced exons and other genomic regions of interest.

## Overview

RBP-RELI adapts the RELI permutation framework ([Harley et al., *Nature Genetics* 2018](https://doi.org/10.1038/s41588-018-0102-3)) from its original GWAS SNP x ChIP-seq application to test whether CLIP-seq peaks from RNA-binding proteins are enriched at loci identified by differential splicing or expression analyses. The method is described in [Fagg et al., *Nucleic Acids Research* 2022](https://doi.org/10.1093/nar/gkac327).

The pipeline supports three analysis modes:

```
                        +-----------------+
  rMATS SE output ----->| extract_inputs  |----+
                        +-----------------+    |
                                               |    +-------------------+    +-----------+
  DESeq2 results  ----->| extract_inputs  |----+--->| build_null_model  |--->| run_reli  |
  + GENCODE GTF         |    _deseq2     |    |    |                   |    |           |
                        +-----------------+    |    +-------------------+    +-----+-----+
                                               |                                  |
  Custom BED files ---->| extract_inputs  |----+                                  v
                        |    _database   |         +----------------+    +----------------+
                        +-----------------+        | plot_figure /  |<---| parse_results  |
                                                   | plot_lollipop  |    +----------------+
                                                   +----------------+
```

Each mode extracts query and background BED files from its respective input, lifts coordinates from hg38 to hg19, builds a per-base null model, then runs RELI against a CLIP-seq peak library.

## Requirements

- **Docker** -- for either the C++ or GPU backend
- **Python 3.8+** with `pandas`, `numpy`, `matplotlib`
- **UCSC liftOver** binary and `hg38ToHg19.over.chain.gz` chain file
- **CLIP-seq peak library** -- a `CLIPseq.index` file and corresponding peak BED files (not included; see below)

## Backends

RBP-RELI provides two backends for the permutation test:

| | C++ (original) | GPU (CuPy/CUDA) |
|---|---|---|
| Dockerfile | `Dockerfile.cpp` | `Dockerfile.gpu` |
| Image | `reli:patched` | `reli-turbo:latest` |
| Requires | CPU only | NVIDIA GPU with CUDA 12+ |
| Speed | ~25 min / run | ~15 sec / run (with caching) |
| Execution | `run_reli.sh` (parallel xargs) | `python -m reli_turbo` |

Both backends produce identical overlap counts. Z-scores differ by <0.1% due to random seed differences.

## Quick Start

### 1. Build the Docker image

```bash
# C++ backend
docker build -f Dockerfile.cpp -t reli:patched .

# GPU backend (requires NVIDIA Docker runtime)
docker build -f Dockerfile.gpu -t reli-turbo:latest .
```

### 2. Prepare inputs (Splicing mode example)

```bash
python scripts/extract_inputs.py \
    SE.MATS.JCEC.txt \
    inputs_hg38/ \
    --dpsi 0.1 --fdr 0.05
```

### 3. Build null model

```bash
python scripts/build_null_model.py \
    --input-dir inputs_hg19/ \
    --genome-build data/GenomeBuild/hg19.txt
```

Note: If your coordinates are hg38, run UCSC `liftOver` on the .snp and BG .bed files first to convert to hg19.

### 4. Run RELI

**C++ backend:**

```bash
bash scripts/run_reli.sh \
    inputs_hg19/ \
    reli_output/ \
    /path/to/CLIPseq.index \
    /path/to/clip_peaks/ \
    /path/to/hg19.txt \
    --threads 16
```

**GPU backend:**

```bash
docker run --rm --gpus all \
    -v $(pwd)/inputs_hg19:/work/inputs \
    -v $(pwd)/reli_output:/work/output \
    -v /path/to/clip_library:/clip \
    reli-turbo:latest \
    --input-dir /work/inputs \
    --output-dir /work/output \
    --index /clip/CLIPseq.index \
    --data /clip/peaks \
    --build /work/inputs/../data/GenomeBuild/hg19.txt \
    --reps 2000
```

### 5. Parse and plot results

```bash
python scripts/parse_results.py reli_output/

python scripts/plot_figure.py reli_output/ --title "My Analysis" --top-n 15

python scripts/plot_lollipop.py reli_output/ --title "My Analysis" --top-n 15
```

## Analysis Modes

### Splicing mode (rMATS)

The core use case. Requires `SE.MATS.JCEC.txt` from rMATS (use `--novelSS` to detect unannotated splice sites). `extract_inputs.py` processes rMATS skipped-exon output and produces query files for six region/direction combinations (SKIP and INCL x upstream intron, alternative exon, downstream intron) plus matched background events. Strand-aware: upstream and downstream intronic regions are flipped for minus-strand genes to maintain biological directionality.

### DESeq2 mode

`extract_inputs_deseq2.py` takes DESeq2 differential expression results and a GENCODE GTF annotation. It tiles gene sub-regions (5'UTR, CDS, intron, 3'UTR) into fixed-width windows, producing separate query sets for upregulated and downregulated genes, with non-significant genes as background.

### Database mode

`extract_inputs_database.py` takes any pair of query and background BED files and converts them to RELI input format. Use this when you have pre-defined regions of interest from a database, custom annotations, or any other source. For large regions (e.g., 3'UTRs, gene bodies), use `--tile` to space query loci across each region rather than testing a single point. The background BED should be biologically matched to the query (e.g., non-target 3'UTRs for a 3'UTR query, random promoters for a promoter query).

## Parameters

### extract_inputs.py

| Parameter | Default | Description |
|-----------|---------|-------------|
| `se_file` | required | Path to `SE.MATS.JCEC.txt` from rMATS |
| `output_dir` | required | Output directory for query and background files |
| `--dpsi` | 0.1 | Minimum absolute delta-PSI for significant events |
| `--fdr` | 0.05 | Maximum FDR for significant events |
| `--pvalue` | 1.0 | Maximum p-value filter (disabled by default) |
| `--bg-fdr` | 0.5 | Minimum FDR for background events |
| `--bg-dpsi` | 0.05 | Maximum absolute delta-PSI for background events |
| `--max-bg` | 5000 | Maximum background events to sample |
| `--seed` | 42 | Random seed for background sampling |
| `--swap` | off | Flip sign convention when Sample1 is the control |

### run_reli.sh

| Parameter | Default | Description |
|-----------|---------|-------------|
| `input_dir` | required | Directory of hg19-lifted query and background files |
| `output_dir` | required | RELI output directory |
| `clip_index` | required | Path to `CLIPseq.index` |
| `peaks_dir` | required | Directory containing CLIP peak BED files |
| `genome_build` | required | Chromosome sizes file (e.g., `hg19.txt`) |
| `--threads` | 16 | Parallel RELI instances |
| `--reps` | 2000 | Permutation replicates |
| `--corr` | auto | Bonferroni correction factor (auto-detected from index) |

## Output

`parse_results.py` produces a combined TSV with columns including:

- **RBP** -- RNA-binding protein name
- **Query** -- region and direction (e.g., `SKIP_AltEX`, `INCL_UPintr`)
- **Overlap** -- observed overlap count between query loci and CLIP peaks
- **Mean_null** -- mean overlap across permuted null distributions
- **Z_score** -- standard score of observed vs. null
- **P_value** -- empirical p-value
- **Enrichment** -- observed / mean null (fold enrichment)
- **Ratio** -- overlap / total query loci

## CLIP Library

The CLIP-seq peak library is not included in this repository due to size. Users must provide:

1. `CLIPseq.index` -- tab-separated file listing dataset ID, BED file name, RBP name, cell type, and source.
2. A directory of peak BED files (hg19 coordinates) referenced by the index.

Peak files should contain original (unpadded) peak calls. RELI handles overlap testing internally.

## Acknowledgments

### Original RELI Framework

RBP-RELI builds directly on the **RELI** (Regulatory Element Locus Intersection) framework developed by the [Weirauch Lab](https://github.com/WeirauchLab) at Cincinnati Children's Hospital Medical Center. RELI was designed and implemented by **Xiaoting Chen**, who served as primary author of the tool and led the computational analysis demonstrating its application to autoimmune disease genetics (Harley et al., *Nature Genetics* 2018). The full author list includes John B. Harley, Xiaoting Chen, Mario Pujato, Daniel Miller, Avery Maddox, Carmy Forney, Albert F. Magnusen, Amber Lynch, Kenneth Kaufman, Leah C. Kottyan, and Matthew T. Weirauch. Kevin Ernst contributed substantially to the software engineering, build system, and cross-platform portability of the C++ codebase. **Phillip Dexheimer, PhD** (Weirauch Lab) has also been instrumental in the continued development and maintenance of the RELI codebase. Their foundational work on permutation-based enrichment testing -- including the null model construction, overlap counting, and statistical framework -- forms the algorithmic core that RBP-RELI extends. Xiaoting Chen also performed the original RBP-RELI analysis for the QKI cardiac mesoderm study (Fagg et al., *Nucleic Acids Research* 2022).

The original RELI source code is available at [https://github.com/WeirauchLab/RELI](https://github.com/WeirauchLab/RELI).

### RBP-RELI Methodology

The adaptation of RELI from GWAS SNP x ChIP-seq to RNA-binding protein CLIP-seq x alternative splicing was conceived and developed by **W. Sam Fagg, PhD**, and described in Fagg et al., *Nucleic Acids Research* 2022, where it was applied to identify Quaking (QKI) as a critical regulator of cardiac cell fate through alternative splicing programs. Dr. Fagg provided the scientific vision for repurposing RELI to the RNA-binding protein domain and has been central to ensuring the biological rigor of this implementation -- from the design of the CLIP-seq enrichment test and the interpretation of spliceosomal dominance patterns, to the validation strategy across disease datasets and the directional specificity analysis that connects RBP binding to splicing outcomes. His deep expertise in RNA biology and post-transcriptional gene regulation has shaped every aspect of how RBP-RELI interprets its results.

### GPU Backend

The GPU-accelerated backend (`reli_turbo/`) was developed with the help of [**James Weatherhead**](https://github.com/JamesWeatherhead), who contributed to the design and implementation of the CuPy/CUDA permutation engine that faithfully reproduces the original C++ algorithm with ~100x speedup through GPU target caching and parallelized overlap counting.

### C++ Bug Fixes

The `RELI/` directory in this repository contains a patched fork of the original C++ implementation with two bug fixes:

- **Uninitialized `max_diff`** -- the original code left a variable uninitialized in `SNPfit()`, causing Z-score saturation at high overlap counts. The patch initializes the value correctly.
- **`atoi()` overflow** -- genomic coordinates exceeding 32-bit integer range caused silent truncation. The patch uses appropriate integer types for coordinate parsing.

`Dockerfile.cpp` builds the patched binary.

## References

1. Harley JB, Chen X, Pujato M, et al. Transcription factors operate across disease loci, with EBNA2 implicated in autoimmunity. *Nature Genetics*. 2018;50:699-707. [doi:10.1038/s41588-018-0102-3](https://doi.org/10.1038/s41588-018-0102-3)

2. Fagg WS, Liu N, Braunschweig U, et al. Definition of germ layer cell lineage alternative splicing programs reveals a critical role for Quaking in specifying cardiac cell fate. *Nucleic Acids Research*. 2022;50(9):5313-5334. [doi:10.1093/nar/gkac327](https://doi.org/10.1093/nar/gkac327)

3. Original RELI source code: [https://github.com/WeirauchLab/RELI](https://github.com/WeirauchLab/RELI)
