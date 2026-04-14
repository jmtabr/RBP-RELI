"""
reli_turbo.io -- File I/O for loading RELI input data.

Handles the exact file formats used by the C++ RELI binary (v0.90):
  - .snp query files (BED4)
  - Null model files (header + cumulative uint32 positions)
  - Genome build files (chr\tsize)
  - Target peak BED files (BED3+)
  - CLIPseq.index (9-column TSV)

Reference: RELI_impl.cpp from the jmtabr/RELI patched fork.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


# ------------------------------------------------------------------ #
#  Genome Build                                                       #
# ------------------------------------------------------------------ #

def load_genome_build(build_path: str | Path) -> tuple[np.ndarray, list[str]]:
    """Parse genome build file (e.g., hg19.txt).

    The file is two-column tab-delimited: chromosome_name<TAB>size.
    Matches RELI_impl.cpp ``createSpeciesMap()`` (line 483).

    The function builds a cumulative prefix-sum array identical to
    ``chromosome_strucuture_val`` in the C++ code: element 0 is 0,
    element *k* is the sum of the first *k* chromosome sizes.

    Parameters
    ----------
    build_path : path-like
        Path to the genome build file.

    Returns
    -------
    chrom_cumsum : np.ndarray, dtype uint64, shape (N_chroms + 1,)
        Cumulative prefix sums.  ``chrom_cumsum[0] == 0``,
        ``chrom_cumsum[-1]`` equals the total genome size.
    chrom_names : list[str]
        Ordered chromosome names (e.g., ``["chr1", "chr2", ...]``).
    """
    chrom_names: list[str] = []
    chrom_sizes: list[int] = []

    with open(build_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            chrom_names.append(parts[0])
            chrom_sizes.append(int(parts[1]))

    # Build cumulative prefix sums (uint64 to avoid overflow for hg38)
    cumsum = np.zeros(len(chrom_sizes) + 1, dtype=np.uint64)
    for i, size in enumerate(chrom_sizes):
        cumsum[i + 1] = cumsum[i] + size

    return cumsum, chrom_names


def make_chrom_to_idx(chrom_names: list[str]) -> dict[str, int]:
    """Return a dict mapping chromosome name to its 0-based index."""
    return {name: idx for idx, name in enumerate(chrom_names)}


# ------------------------------------------------------------------ #
#  Null Model                                                         #
# ------------------------------------------------------------------ #

def load_null_model(null_path: str | Path) -> np.ndarray:
    """Load a null model file of cumulative genomic positions.

    Format (when ``-match`` is NOT used, the RBP-RELI case):
        Line 1: header (skipped -- see RELI_impl.cpp line 1058)
        Remaining lines: tab-delimited, column 0 is a cumulative
        genomic position as an unsigned integer.  Column 1 (bin) is
        ignored when MAF matching is off.

    Header format: ``cumulative_position\\tbin``

    Parameters
    ----------
    null_path : path-like
        Path to the null model file.

    Returns
    -------
    positions : np.ndarray, dtype uint64
        Sorted array of cumulative genomic positions.
    """
    positions: list[int] = []

    with open(null_path) as fh:
        fh.readline()  # skip header (RELI_impl.cpp line 1058: in.ignore)
        for line in fh:
            line = line.strip()
            if not line:
                continue
            # Column 0 is the cumulative position; column 1 (bin) is ignored.
            val = int(line.split("\t")[0])
            positions.append(val)

    return np.array(positions, dtype=np.uint64)


# ------------------------------------------------------------------ #
#  Query Loci (.snp)                                                  #
# ------------------------------------------------------------------ #

def load_snp_file(
    snp_path: str | Path,
    chrom_to_idx: dict[str, int],
) -> dict[str, np.ndarray | list[str]]:
    """Load a query .snp file in BED4 format.

    Matches RELI_impl.cpp ``loadSnpFile()`` (line 1074).  Each line is:
        chr<TAB>start<TAB>end<TAB>name

    ``length = end - start`` for each entry.  For traditional SNPs
    length is 0; for RBP-RELI splicing intervals it is > 0.

    Parameters
    ----------
    snp_path : path-like
        Path to the BED4 query file.
    chrom_to_idx : dict
        Mapping of chromosome name to integer index.

    Returns
    -------
    dict with keys:
        'chr_idx'  : np.ndarray, dtype uint32
        'start'    : np.ndarray, dtype uint32
        'end'      : np.ndarray, dtype uint32
        'length'   : np.ndarray, dtype uint32
        'names'    : list[str]
    """
    chr_indices: list[int] = []
    starts: list[int] = []
    ends: list[int] = []
    lengths: list[int] = []
    names: list[str] = []

    with open(snp_path) as fh:
        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue
            chrom = parts[0]
            if chrom not in chrom_to_idx:
                continue
            start = int(parts[1])
            end = int(parts[2])
            name = parts[3]

            chr_indices.append(chrom_to_idx[chrom])
            starts.append(start)
            ends.append(end)
            lengths.append(end - start)  # RELI_impl.cpp line 1087
            names.append(name)

    return {
        "chr_idx": np.array(chr_indices, dtype=np.uint32),
        "start": np.array(starts, dtype=np.uint32),
        "end": np.array(ends, dtype=np.uint32),
        "length": np.array(lengths, dtype=np.uint32),
        "names": names,
    }


def convert_snp_to_cumulative(
    snp_data: dict,
    chrom_cumsum: np.ndarray,
) -> np.ndarray:
    """Convert per-chromosome query coordinates to cumulative space.

    This is the inverse of ``SNPfit()`` -- useful for verification.
    The centre-point of each query locus is converted:
        cum_pos = chrom_cumsum[chr_idx] + (start + end) // 2

    Parameters
    ----------
    snp_data : dict
        Output of :func:`load_snp_file`.
    chrom_cumsum : np.ndarray
        Cumulative prefix sums from :func:`load_genome_build`.

    Returns
    -------
    np.ndarray, dtype uint64
        Cumulative positions for each query locus.
    """
    offsets = chrom_cumsum[snp_data["chr_idx"]]
    centres = ((snp_data["start"].astype(np.uint64)
                + snp_data["end"].astype(np.uint64)) // 2)
    return offsets + centres


# ------------------------------------------------------------------ #
#  CLIP Index                                                         #
# ------------------------------------------------------------------ #

def load_clip_index(
    index_path: str | Path,
) -> tuple[list[str], list[dict[str, str]]]:
    """Parse the CLIPseq.index file.

    The file is a 9-column TSV with a header line (skipped).  Columns:
        0: Dataset label (used as BED filename stem)
        1: Source
        2: Cell type
        3: TF / RBP name
        4-8: additional metadata (variable)

    Matches RELI_impl.cpp ``public_ver_read_data_index()`` (line ~920).

    Parameters
    ----------
    index_path : path-like
        Path to CLIPseq.index.

    Returns
    -------
    labels : list[str]
        Dataset labels (one per target).
    metadata : list[dict]
        Per-target metadata dicts with keys 'source', 'cell', 'tf'.
    """
    labels: list[str] = []
    metadata: list[dict[str, str]] = []

    with open(index_path) as fh:
        fh.readline()  # skip header
        for line in fh:
            parts = line.strip().split("\t")
            if not parts or not parts[0]:
                continue
            labels.append(parts[0])
            metadata.append({
                "source": parts[1] if len(parts) > 1 else ".",
                "cell": parts[2] if len(parts) > 2 else ".",
                "tf": parts[3] if len(parts) > 3 else ".",
            })

    return labels, metadata


# ------------------------------------------------------------------ #
#  Target Peaks                                                       #
# ------------------------------------------------------------------ #

def load_target_peaks(
    peaks_dir: str | Path,
    target_filename: str,
    chrom_to_idx: dict[str, int],
) -> np.ndarray:
    """Load a single target peak BED file.

    Matches RELI_impl.cpp ``target_bed_file::readingData()`` (line 785):
    reads BED3 (chr, start, end), sorts by (chr, start).

    Parameters
    ----------
    peaks_dir : path-like
        Directory containing peak BED files.
    target_filename : str
        Filename (without directory) of the target BED file.
    chrom_to_idx : dict
        Chromosome name to index mapping.

    Returns
    -------
    peaks : np.ndarray, dtype [('chr', uint32), ('start', uint32), ('end', uint32)]
        Structured array sorted by (chr, start).
    """
    bed_path = os.path.join(peaks_dir, target_filename)
    rows: list[tuple[int, int, int]] = []

    with open(bed_path) as fh:
        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            chrom = parts[0]
            if chrom not in chrom_to_idx:
                continue
            rows.append((
                chrom_to_idx[chrom],
                int(parts[1]),
                int(parts[2]),
            ))

    if not rows:
        return np.empty(0, dtype=[("chr", np.uint32),
                                   ("start", np.uint32),
                                   ("end", np.uint32)])

    arr = np.array(rows, dtype=[("chr", np.uint32),
                                 ("start", np.uint32),
                                 ("end", np.uint32)])
    # Sort by (chr, start) -- matches RELI's sort
    order = np.lexsort((arr["start"], arr["chr"]))
    return arr[order]


def load_all_targets(
    peaks_dir: str | Path,
    index_path: str | Path,
    chrom_to_idx: dict[str, int],
    n_chroms: int,
) -> dict:
    """Load all CLIP target peak files, concatenate into flat arrays.

    This is the batch loader used by the GPU pipeline.  All target
    BED files are concatenated into flat uint32 arrays with offset/size
    metadata so the GPU kernel can index into any target's slice.

    Additionally builds a per-target per-chromosome offset index so the
    kernel can jump directly to the peaks on a given chromosome within a
    given target (equivalent to ``targetbedfileindex_start`` in C++).

    Parameters
    ----------
    peaks_dir : path-like
        Directory containing target peak BED files.
    index_path : path-like
        Path to CLIPseq.index.
    chrom_to_idx : dict
        Chromosome name to index mapping.
    n_chroms : int
        Number of chromosomes in the genome build.

    Returns
    -------
    dict with keys:
        all_chr, all_start, all_end : np.ndarray uint32 -- flat peak arrays
        target_offsets : np.ndarray uint32 -- start index of each target
        target_sizes   : np.ndarray uint32 -- number of peaks per target
        target_chr_offsets : np.ndarray uint32 -- per-target chr offset index
                            shape conceptually (N_targets, n_chroms+1) flattened
        target_labels  : list[str]
        target_metadata: list[dict]
    """
    labels, metadata = load_clip_index(index_path)

    # --- Check for cached .npz file ---
    cache_path = str(index_path) + ".targets.npz"
    if os.path.exists(cache_path):
        try:
            cache_mtime = os.path.getmtime(cache_path)
            index_mtime = os.path.getmtime(index_path)
            if cache_mtime > index_mtime:
                data = np.load(cache_path, allow_pickle=False)
                if int(data["_n_chroms"]) == n_chroms:
                    print(f"  Loaded target cache: {cache_path}")
                    return {
                        "all_chr": data["all_chr"],
                        "all_start": data["all_start"],
                        "all_end": data["all_end"],
                        "target_offsets": data["target_offsets"],
                        "target_sizes": data["target_sizes"],
                        "target_chr_offsets": data["target_chr_offsets"],
                        "target_labels": data["target_labels"].tolist(),
                        "target_metadata": [
                            {"source": str(s), "cell": str(c), "tf": str(t)}
                            for s, c, t in zip(
                                data["meta_source"], data["meta_cell"], data["meta_tf"]
                            )
                        ],
                    }
        except Exception:
            pass  # Fall through to full load

    all_chr_list: list[int] = []
    all_start_list: list[int] = []
    all_end_list: list[int] = []
    offsets: list[int] = [0]
    sizes: list[int] = []
    chr_offsets: list[int] = []  # flat: [t0_c0, t0_c1, ..., t0_cN, t1_c0, ...]

    for label in labels:
        # Try with and without .bed extension
        bed_path = os.path.join(peaks_dir, label)
        if not os.path.exists(bed_path):
            bed_path = os.path.join(peaks_dir, label + ".bed")
        if not os.path.exists(bed_path):
            # Empty target -- no peaks
            sizes.append(0)
            offsets.append(offsets[-1])
            chr_offsets.extend([0] * (n_chroms + 1))
            continue

        # Read BED, convert chr to index, collect tuples
        peaks: list[tuple[int, int, int]] = []
        with open(bed_path) as fh:
            for line in fh:
                parts = line.strip().split("\t")
                if len(parts) >= 3 and parts[0] in chrom_to_idx:
                    peaks.append((
                        chrom_to_idx[parts[0]],
                        int(parts[1]),
                        int(parts[2]),
                    ))

        # Sort by (chr_idx, start) -- matches RELI readingData + sort
        peaks.sort(key=lambda x: (x[0], x[1]))

        n_peaks = len(peaks)
        sizes.append(n_peaks)

        # Build per-chromosome offset index within this target.
        # target_chr_off[c] = index of first peak on chromosome c.
        # target_chr_off[n_chroms] = n_peaks (sentinel).
        target_chr_off = [0] * (n_chroms + 1)
        if n_peaks > 0:
            current_chr = -1
            for j, (c, s, e) in enumerate(peaks):
                if c != current_chr:
                    for fill_c in range(current_chr + 1, c + 1):
                        target_chr_off[fill_c] = j
                    current_chr = c
            for fill_c in range(current_chr + 1, n_chroms + 1):
                target_chr_off[fill_c] = n_peaks

        chr_offsets.extend(target_chr_off)

        for c, s, e in peaks:
            all_chr_list.append(c)
            all_start_list.append(s)
            all_end_list.append(e)

        offsets.append(offsets[-1] + n_peaks)

    result = {
        "all_chr": np.array(all_chr_list, dtype=np.uint32),
        "all_start": np.array(all_start_list, dtype=np.uint32),
        "all_end": np.array(all_end_list, dtype=np.uint32),
        "target_offsets": np.array(offsets[:-1], dtype=np.uint32),
        "target_sizes": np.array(sizes, dtype=np.uint32),
        "target_chr_offsets": np.array(chr_offsets, dtype=np.uint32),
        "target_labels": labels,
        "target_metadata": metadata,
    }

    # Save cache for next run
    try:
        np.savez(
            cache_path,
            all_chr=result["all_chr"],
            all_start=result["all_start"],
            all_end=result["all_end"],
            target_offsets=result["target_offsets"],
            target_sizes=result["target_sizes"],
            target_chr_offsets=result["target_chr_offsets"],
            target_labels=np.array(result["target_labels"]),
            meta_source=np.array([m["source"] for m in metadata]),
            meta_cell=np.array([m["cell"] for m in metadata]),
            meta_tf=np.array([m["tf"] for m in metadata]),
            _n_chroms=np.array(n_chroms),
        )
        print(f"  Saved target cache: {cache_path}")
    except Exception as exc:
        print(f"  Warning: could not save target cache: {exc}")

    return result
