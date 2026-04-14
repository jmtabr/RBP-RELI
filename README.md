# RBP-RELI

Permutation-based enrichment testing of RNA-binding protein CLIP-seq peaks at alternatively spliced exons and other genomic regions of interest.

## Overview

RBP-RELI adapts the RELI permutation framework ([Harley et al., *Nature Genetics* 2018](https://doi.org/10.1038/s41588-018-0102-3)) from its original GWAS SNP x ChIP-seq application to test whether CLIP-seq peaks from RNA-binding proteins are enriched at loci identified by differential splicing or expression analyses. The method is described in [Fagg et al., *Nucleic Acids Research* 2022](https://doi.org/10.1093/nar/gkac327).

The pipeline supports three analysis modes through a unified interface:

```
                              +-----------------------+
  rMATS SE output ----------->|                       |
                              |  extract_inputs.py    |----+
  DESeq2 + GENCODE GTF -----> |  --mode splicing      |    |    +---------------+
                              |  --mode deseq2        |    +--->|  run_reli.py  |
  Custom BED files ---------->|  --mode database      |    |    |  --backend    |
                              +-----------------------+    |    |  gpu | cpp    |
                                                           |    +-------+-------+
                              Outputs:                     |            |
                                .snp query files           |            v
                                BG .bed files              |    +---------------+
                                Null models                |    | plot_lollipop |
                                manifest.json              |    | plot_figure   |
                                dummy_dbsnp                |    +---------------+
```

## Requirements

- **Docker** -- for running either the C++ or GPU backend
- **Python 3.8+** with `pandas`, `numpy`, `matplotlib`
- **CLIP-seq peak library** -- a `CLIPseq.index` file and corresponding peak BED files (not included; see below)

Optional:
- **UCSC liftOver** binary and `hg38ToHg19.over.chain.gz` -- only if your coordinates are hg38

## Installation

```bash
git clone https://github.com/jmtabr/RBP-RELI.git
cd RBP-RELI

# Build the Docker image for your preferred backend:

# GPU backend (recommended, requires NVIDIA Docker runtime)
docker build -f Dockerfile.gpu -t reli-turbo:latest .

# C++ backend (no GPU required)
docker build -f Dockerfile.cpp -t reli:patched .
```

No additional Python installation is needed beyond the standard library for the extraction step. The RELI computation runs entirely inside Docker.

## Quick Start

The entire pipeline is two commands:

### Step 1: Extract inputs

```bash
# Splicing mode (rMATS)
python scripts/extract_inputs.py --mode splicing \
    --se-file SE.MATS.JCEC.txt \
    --output-dir inputs/ \
    --genome-build data/GenomeBuild/hg19.txt

# DESeq2 mode (differential expression)
python scripts/extract_inputs.py --mode deseq2 \
    --deseq2 deseq2_results.csv \
    --gtf gencode.v19.annotation.gtf.gz \
    --output-dir inputs/ \
    --genome-build data/GenomeBuild/hg19.txt

# Database mode (custom BED regions)
python scripts/extract_inputs.py --mode database \
    --query-bed my_regions.bed \
    --bg-bed background.bed \
    --output-dir inputs/ \
    --genome-build data/GenomeBuild/hg19.txt
```

This produces query `.snp` files, background `.bed` files, validated null models, and a `manifest.json` describing the extraction. All coordinates must be hg19.

### Step 2: Run RELI

```bash
# GPU backend (recommended)
python scripts/run_reli.py \
    --input-dir inputs/ \
    --output-dir output/ \
    --backend gpu \
    --index /path/to/CLIPseq.index \
    --data /path/to/clip_peaks/ \
    --build data/GenomeBuild/hg19.txt

# C++ backend (no GPU required)
python scripts/run_reli.py \
    --input-dir inputs/ \
    --output-dir output/ \
    --backend cpp \
    --index /path/to/CLIPseq.index \
    --data /path/to/clip_peaks/ \
    --build data/GenomeBuild/hg19.txt \
    --threads 16
```

Both backends run inside Docker automatically. Results are parsed and optionally annotated with RBP functional categories (when a category file is provided via `--categories`).

### Step 3: Plot

```bash
# Lollipop plot (Z-score stalk, enrichment circle, p-value color)
python scripts/plot_lollipop.py output/ \
    --title "My Analysis" \
    --top-n 10 \
    --max-pval 0.05 --min-ratio 0.1

# Show all CLIP datasets (RBP can appear multiple times with cell line)
python scripts/plot_lollipop.py output/ \
    --title "My Analysis" \
    --top-n 10 \
    --all-clips
```

## Input Formats

### Splicing mode

The splicing mode expects a tab-separated file with the following columns from rMATS `SE.MATS.JCEC.txt`:

| Column | Description | Example |
|--------|-------------|---------|
| `chr` | Chromosome | `chr1` |
| `strand` | Strand | `+` or `-` |
| `exonStart_0base` | Exon start (0-based) | `12345` |
| `exonEnd` | Exon end | `12500` |
| `FDR` | False discovery rate | `0.001` |
| `IncLevelDifference` | Delta PSI (Sample1 - Sample2) | `-0.25` |
| `PValue` | Raw p-value (optional, for extra filtering) | `0.0005` |

**Adapting other tools:** If you have output from SUPPA2, VAST-Tools, or another splicing tool, you can create a TSV with these column names. The key requirements are: chromosome, strand, exon coordinates (0-based start, 1-based end), an FDR/adjusted p-value column, and a delta-PSI column (positive = more inclusion in Sample1, negative = more skipping). Rename your columns to match and the extractor will work.

### DESeq2 mode

The DESeq2 mode expects a tab-separated or comma-separated results file with:

| Column | Default name | Description | Configurable via |
|--------|-------------|-------------|-----------------|
| Gene symbol | `gene_symbol` | HGNC gene name | `--gene-col` |
| Log2 fold change | `log2FoldChange` | Effect size | `--log2fc-col` |
| Adjusted p-value | `padj` | Multiple-testing corrected | `--padj-col` |
| Base mean | `baseMean` | Mean expression level | `--basemean-col` |

Column names are configurable, so output from edgeR, limma, or other DE tools can be used by specifying the appropriate column names. The mode also requires a GENCODE GTF annotation file (v19 for hg19) for gene coordinate extraction.

### Database mode

The database mode expects two standard BED files (tab-separated, no header):

**Query BED** (regions of interest):
```
chr1    12345    12500    my_region_1
chr2    98000    98200    my_region_2
```

**Background BED** (matched null set):
```
chr1    50000    50200    bg_1
chr3    75000    75150    bg_2
```

Columns: chromosome, start (0-based), end, name (optional). The background should be biologically matched to the query (e.g., non-target 3'UTRs for a 3'UTR query).

## Analysis Modes

### Splicing mode (rMATS)

The core use case. Requires `SE.MATS.JCEC.txt` from rMATS (use `--novelSS` to detect unannotated splice sites). Produces query files for six region/direction combinations (SKIP and INCL x upstream intron, alternative exon, downstream intron) plus matched background events. Strand-aware: upstream and downstream intronic regions are flipped for minus-strand genes.

### DESeq2 mode

Takes DESeq2 differential expression results and a GENCODE GTF annotation. Tiles gene sub-regions (5'UTR, CDS, intron, 3'UTR) into fixed-width windows, producing separate query sets for upregulated and downregulated genes, with non-significant genes as background.

### Database mode

Takes any pair of query and background BED files. Use this when you have pre-defined regions of interest from a database, custom annotations, or any other source. For large regions (e.g., 3'UTRs, gene bodies), use `--tile` to space query loci across each region. The background BED should be biologically matched to the query.

## Backends

| | C++ (original) | GPU (CuPy/CUDA) |
|---|---|---|
| Docker image | `reli:patched` | `reli-turbo:latest` |
| Requires | CPU only | NVIDIA GPU with CUDA 12+ |
| Speed | ~25 min / run | ~15 sec / run (with caching) |
| `--backend` flag | `cpp` | `gpu` |

Both backends use the same null model format and produce identical overlap counts. Z-scores differ by <0.1% due to random seed differences in permutation sampling. The GPU backend is recommended for speed; the C++ backend is available for environments without GPU access.

## Parameters

### extract_inputs.py (shared)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--mode` | required | `splicing`, `deseq2`, or `database` |
| `--output-dir` | required | Output directory |
| `--genome-build` | required | Genome build file (e.g., `hg19.txt`) |
| `--seed` | 42 | Random seed for background sampling |
| `--max-bg` | 5000 | Maximum background events to sample |

### extract_inputs.py (splicing mode)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--se-file` | required | Path to `SE.MATS.JCEC.txt` from rMATS |
| `--dpsi` | 0.1 | Minimum absolute delta-PSI for significant events |
| `--fdr` | 0.05 | Maximum FDR for significant events |
| `--pvalue` | 1.0 | Maximum p-value filter (disabled by default) |
| `--bg-fdr` | 0.5 | Minimum FDR for background events |
| `--bg-dpsi` | 0.05 | Maximum absolute delta-PSI for background events |
| `--swap` | off | Flip sign convention when Sample1 is the control |

### extract_inputs.py (DESeq2 mode)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--deseq2` | required | DESeq2 results file (TSV or CSV) |
| `--gtf` | required | GENCODE GTF annotation (gzip OK) |
| `--padj-threshold` | 0.05 | Adjusted p-value threshold for DE genes |
| `--min-log2fc` | 0.4 | Minimum absolute log2 fold change |
| `--spacing` | 100 | Spacing between tiled query positions (bp) |

### extract_inputs.py (database mode)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--query-bed` | required | BED file with query regions |
| `--bg-bed` | required | BED file with background regions |
| `--prefix` | DB | Output file prefix |
| `--tile` | off | Tile large regions at this spacing (bp) |

### run_reli.py

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--input-dir` | required | Directory from extract_inputs.py |
| `--output-dir` | required | Results directory |
| `--index` | required | Path to `CLIPseq.index` |
| `--data` | required | Directory containing CLIP peak BED files |
| `--build` | required | Genome build file |
| `--backend` | gpu | `gpu` or `cpp` |
| `--reps` | 2000 | Permutation replicates |
| `--corr` | auto | Bonferroni correction (auto-detected from index) |
| `--threads` | 16 | C++ backend: parallel RELI instances |

## Output

Each query produces a consolidated TSV (`{query}_all_results.tsv`) with columns:

| Column | Description |
|--------|-------------|
| **RBP** | RNA-binding protein name |
| **Cell** | Cell type of the CLIP dataset |
| **Label** | CLIP dataset identifier |
| **Overlap** | Observed overlap count |
| **Total** | Total query loci |
| **Ratio** | Overlap / Total |
| **Mean** | Mean overlap across permuted null |
| **SD** | Standard deviation of null distribution |
| **Z-score** | (Observed - Mean) / SD |
| **Enrichment** | Observed / Mean (fold enrichment) |
| **P-value** | Empirical p-value from permutation |
| **Corrected P-value** | Bonferroni-corrected p-value |

## CLIP Library

The CLIP-seq peak library is not included in this repository due to size. Users must provide:

1. `CLIPseq.index` -- tab-separated file with columns: label, source, Cell, RBP, Cell_label, PMID, Group, Method, Species.
2. A directory of peak BED files (hg19 coordinates) referenced by the label column.

Peak files should contain original (unpadded) peak calls. RELI handles overlap testing internally.

## Acknowledgments

### Original RELI Framework

RBP-RELI builds directly on the **RELI** (Regulatory Element Locus Intersection) framework developed by the **[Weirauch Lab]**(https://github.com/WeirauchLab) at Cincinnati Children's Hospital Medical Center. RELI was designed and implemented by **Xiaoting Chen**, who served as primary author of the tool and led the computational analysis demonstrating its application to autoimmune disease genetics (Harley et al., *Nature Genetics* 2018). The full author list includes John B. Harley, Xiaoting Chen, Mario Pujato, Daniel Miller, Avery Maddox, Carmy Forney, Albert F. Magnusen, Amber Lynch, Kenneth Kaufman, Leah C. Kottyan, and **Matthew T. Weirauch**. Phillip Dexheimer, PhD, and Kevin Ernst have also been instrumental in the continued development and maintenance of the RELI codebase. Their foundational work on permutation-based enrichment testing -- including the null model construction, overlap counting, and statistical framework -- forms the algorithmic core that RBP-RELI extends. Xiaoting Chen also performed the original RBP-RELI analysis for the QKI cardiac mesoderm study (Fagg et al., *Nucleic Acids Research* 2022).

The original RELI source code is available at [https://github.com/WeirauchLab/RELI](https://github.com/WeirauchLab/RELI).

### RBP-RELI Methodology

The adaptation of RELI from GWAS SNP x ChIP-seq to RNA-binding protein CLIP-seq x alternative splicing was conceived and developed by **W. Sam Fagg, PhD**, and described in Fagg et al., *Nucleic Acids Research* 2022, where it was applied to identify Quaking (QKI) as a critical regulator of cardiac cell fate through alternative splicing programs. Dr. Fagg, along with the Weirauch Lab, provided the scientific vision for repurposing RELI to the RNA-binding protein domain and has been central to ensuring the biological rigor of this implementation -- from the design of the CLIP-seq enrichment test and the interpretation of spliceosomal dominance patterns, to the validation strategy across disease datasets and the directional specificity analysis that connects RBP binding to splicing outcomes. His deep expertise in RNA biology and post-transcriptional gene regulation has shaped every aspect of how RBP-RELI interprets its results.

### GPU Backend

The GPU-accelerated backend (`reli_turbo/`) was developed with the help of [**James Weatherhead**](https://github.com/JamesWeatherhead), who contributed to the design and implementation of the CuPy/CUDA permutation engine that faithfully reproduces the original C++ algorithm with ~100x speedup through GPU target caching and parallelized overlap counting.

### Patched C++

The `RELI/` directory in this repository contains a patched fork of the original C++ implementation to address bug fixes.
`Dockerfile.cpp` builds the patched binary.

## References

1. Harley JB, Chen X, Pujato M, et al. Transcription factors operate across disease loci, with EBNA2 implicated in autoimmunity. *Nature Genetics*. 2018;50:699-707. [doi:10.1038/s41588-018-0102-3](https://doi.org/10.1038/s41588-018-0102-3)

2. Fagg WS, Liu N, Braunschweig U, et al. Definition of germ layer cell lineage alternative splicing programs reveals a critical role for Quaking in specifying cardiac cell fate. *Nucleic Acids Research*. 2022;50(9):5313-5334. [doi:10.1093/nar/gkac327](https://doi.org/10.1093/nar/gkac327)

3. Original RELI source code: [https://github.com/WeirauchLab/RELI](https://github.com/WeirauchLab/RELI)
