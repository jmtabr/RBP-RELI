"""
reli_turbo.reli -- Main RELI orchestration: GPU computation and output.

This module ties together I/O, GPU kernels, and statistics to provide
high-level functions for running RELI enrichment analysis.

Three levels of API:
  - run_single_query:  one query vs one target (for testing)
  - run_batch:         one query vs all targets (the GPU sweet spot)
  - run_full_pipeline: all 8 queries vs all targets
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

from . import io as reli_io
from . import stats as reli_stats
from .kernels import (
    BLOCK_SIZE,
    generate_random_indices,
    launch_reli_kernel,
    launch_rsid_kernel,
)

# Number of null permutations (matches RELI default)
N_PERMS_DEFAULT = 2000
# Max retries for rejection sampling on GPU
MAX_RETRIES = 10


def targets_to_gpu(targets: dict, device: int = 0) -> dict:
    """Transfer target peak data to GPU memory once for reuse across queries.

    Returns a dict of CuPy arrays that can be passed to ``run_batch()``
    via the ``gpu_targets`` parameter to skip redundant transfers.
    """
    cp.cuda.Device(device).use()
    t0 = time.time()
    gpu = {
        "d_all_start": cp.asarray(targets["all_start"]),
        "d_all_end": cp.asarray(targets["all_end"]),
        "d_offsets": cp.asarray(targets["target_offsets"]),
        "d_sizes": cp.asarray(targets["target_sizes"]),
        "d_chr_offsets": cp.asarray(targets["target_chr_offsets"]),
    }
    dt = time.time() - t0
    nbytes = sum(a.nbytes for a in gpu.values())
    print(f"  Targets cached on GPU: {nbytes / 1e6:.0f} MB in {dt:.3f}s")
    return gpu


# ================================================================== #
#  Observed RSID Collection (CPU)                                     #
# ================================================================== #

def collect_observed_rsids(
    query: dict,
    targets: dict,
    observed_counts: np.ndarray,
) -> dict[int, list[str]]:
    """Identify which query loci overlapped each target in iteration 0.

    This replicates the rsid_collector logic from overlapping3()
    when ``_iter_number == 0`` (RELI_impl.cpp line 335-336).
    Done on CPU since it requires string locus names and is only
    needed once (for the observed iteration).

    Parameters
    ----------
    query : dict
        Output of ``load_snp_file()``.
    targets : dict
        Output of ``load_all_targets()``.
    observed_counts : np.ndarray
        Observed overlap count per target (1-D).

    Returns
    -------
    rsids : dict[int, list[str]]
        Mapping from target index to sorted list of overlapping locus names.
    """
    n_targets = len(targets["target_labels"])
    rsids: dict[int, list[str]] = {}

    for t in range(n_targets):
        if int(observed_counts[t]) == 0:
            rsids[t] = []
            continue

        offset = int(targets["target_offsets"][t])
        size = int(targets["target_sizes"][t])
        if size == 0:
            rsids[t] = []
            continue

        peak_starts = targets["all_start"][offset:offset + size]
        peak_ends = targets["all_end"][offset:offset + size]
        peak_chrs = targets["all_chr"][offset:offset + size]

        overlapping_names: list[str] = []
        for i in range(len(query["chr_idx"])):
            q_chr = int(query["chr_idx"][i])
            q_start = int(query["start"][i])
            q_end = int(query["end"][i])

            # Find peaks on the same chromosome
            mask = peak_chrs == q_chr
            if not np.any(mask):
                continue

            p_starts = peak_starts[mask]
            p_ends = peak_ends[mask]

            # Check the 4 overlap conditions (RELI_impl.cpp lines 329-332)
            for j in range(len(p_starts)):
                bs = int(p_starts[j])
                be = int(p_ends[j])
                if ((bs <= q_start and be >= q_end)
                        or (bs >= q_start and bs + 1 < q_end)
                        or (be > q_start + 1 and be <= q_end)
                        or (bs >= q_start and be <= q_end)):
                    overlapping_names.append(query["names"][i])
                    break  # first overlap per locus is sufficient

        rsids[t] = sorted(set(overlapping_names))

    return rsids


def collect_observed_rsids_gpu_from_flags(
    query: dict,
    flags_2d: np.ndarray,
) -> dict[int, list[str]]:
    """Convert GPU overlap flags to the same dict format as the CPU version.

    Parameters
    ----------
    query : dict
        Output of ``load_snp_file()``.
    flags_2d : np.ndarray, uint8, shape (n_targets, n_loci)
        Per-locus overlap flags from the RSID kernel.

    Returns
    -------
    rsids : dict[int, list[str]]
        Mapping from target index to sorted list of overlapping locus names.
    """
    n_targets = flags_2d.shape[0]
    names = query["names"]
    # Pre-convert to numpy array for vectorized indexing
    names_arr = np.asarray(names)
    # Pre-compute per-target overlap counts to skip zero-overlap targets fast
    overlap_counts = flags_2d.sum(axis=1)
    rsids: dict[int, list[str]] = {}
    for t in range(n_targets):
        if overlap_counts[t] == 0:
            rsids[t] = []
        else:
            mask = flags_2d[t].astype(bool)
            rsids[t] = sorted(set(names_arr[mask].tolist()))
    return rsids


# ================================================================== #
#  GPU Execution: Single Query vs All Targets                         #
# ================================================================== #

def run_batch(
    query: dict,
    null_model: np.ndarray,
    chrom_cumsum: np.ndarray,
    targets: dict,
    n_reps: int = N_PERMS_DEFAULT,
    n_targets_corr: int | None = None,
    seed: int = 42,
    device: int = 0,
    gpu_targets: dict | None = None,
    gpu_null: object | None = None,
) -> dict:
    """Run RELI permutation testing on GPU: one query vs all targets.

    This is the core GPU-accelerated function.  It:
    1. Transfers all data to GPU memory.
    2. Pre-generates random indices on GPU.
    3. Launches a single kernel covering all (target, iteration) pairs.
    4. Computes Z-scores and P-values.
    5. Returns results as CPU numpy arrays.

    Parameters
    ----------
    query : dict
        Output of ``reli_io.load_snp_file()``.
    null_model : np.ndarray, uint64
        Null model positions from ``reli_io.load_null_model()``.
    chrom_cumsum : np.ndarray, uint64
        Cumulative chromosome sizes from ``reli_io.load_genome_build()``.
    targets : dict
        Output of ``reli_io.load_all_targets()``.
    n_reps : int
        Number of null permutations (default 2000).
    n_targets_corr : int, optional
        Bonferroni correction factor.  Defaults to number of targets.
    seed : int
        Random seed.
    device : int
        CUDA device ID.
    gpu_targets : dict, optional
        Pre-transferred GPU arrays from ``targets_to_gpu()``.
        When provided, skips redundant CPU→GPU transfer of peak data.
    gpu_null : object, optional
        Pre-transferred null model CuPy array.
        When provided, skips redundant CPU→GPU transfer of null model.

    Returns
    -------
    dict with keys:
        observed   : np.ndarray int, (N_targets,)
        mean       : np.ndarray float64, (N_targets,)
        sd         : np.ndarray float64, (N_targets,)
        zscore     : np.ndarray float64, (N_targets,)
        pval       : np.ndarray float64, (N_targets,)
        corr_pval  : np.ndarray float64, (N_targets,)
        enrichment : np.ndarray float64, (N_targets,)
        all_counts : np.ndarray int32, (N_targets, n_iters)
    """
    if not HAS_CUPY:
        raise RuntimeError(
            "CuPy is required for GPU execution. "
            "Install with: pip install cupy-cuda12x"
        )

    cp.cuda.Device(device).use()

    n_iters = n_reps + 1
    n_loci = len(query["chr_idx"])
    n_targets = len(targets["target_labels"])
    n_chroms = len(chrom_cumsum) - 1
    null_size = len(null_model)

    if n_targets_corr is None:
        n_targets_corr = n_targets

    print(f"  GPU RELI: {n_targets} targets x {n_iters} iters x "
          f"{n_loci} loci = {n_targets * n_iters:,} thread tasks")

    # ---- Transfer data to GPU ----
    t0 = time.time()
    d_null = gpu_null if gpu_null is not None else cp.asarray(null_model)
    d_cumsum = cp.asarray(chrom_cumsum)    # uint64
    d_query_chr = cp.asarray(query["chr_idx"])
    d_query_start = cp.asarray(query["start"])
    d_query_end = cp.asarray(query["end"])
    d_query_length = cp.asarray(query["length"])

    # Target peak data — reuse cached GPU arrays if available
    if gpu_targets is not None:
        d_all_start = gpu_targets["d_all_start"]
        d_all_end = gpu_targets["d_all_end"]
        d_offsets = gpu_targets["d_offsets"]
        d_sizes = gpu_targets["d_sizes"]
        d_chr_offsets = gpu_targets["d_chr_offsets"]
    else:
        d_all_start = cp.asarray(targets["all_start"])
        d_all_end = cp.asarray(targets["all_end"])
        d_offsets = cp.asarray(targets["target_offsets"])
        d_sizes = cp.asarray(targets["target_sizes"])
        d_chr_offsets = cp.asarray(targets["target_chr_offsets"])

    t_transfer = time.time() - t0
    cached_str = ""
    if gpu_targets:
        cached_str += " targets"
    if gpu_null is not None:
        cached_str += " null"
    if cached_str:
        cached_str = f" (cached:{cached_str})"
    print(f"  Data transfer to GPU: {t_transfer:.3f}s{cached_str}")

    # ---- Estimate VRAM and decide: single batch vs chunked ----
    vram_free = cp.cuda.Device(device).mem_info[0]

    # Primary random array: n_targets * n_reps * n_loci * 4 bytes (uint32)
    bytes_per_target_rand = n_reps * n_loci * 4
    # With max_retries, peak VRAM = primary * (1 + 2 * max_retries)
    # Also per target: overlap_counts (n_iters * 4) + rsid_flags (n_loci * 1)
    bytes_per_target_aux = n_iters * 4 + n_loci

    # Determine max_retries globally (same logic as before)
    n_primary_full = n_targets * n_reps * n_loci
    bytes_primary_full = n_primary_full * 4
    vram_budget = int(vram_free * 0.75)
    max_retries = MAX_RETRIES
    peak_bytes_full = bytes_primary_full * (1 + 2 * max_retries)
    if peak_bytes_full > vram_budget and bytes_primary_full > 0:
        max_retries = max(0, int((vram_budget / bytes_primary_full - 1) / 2))
        print(f"  VRAM-aware: reduced max_retries from {MAX_RETRIES} "
              f"to {max_retries} (free: {vram_free/1e9:.1f} GB)")

    # Compute total bytes needed for single-batch execution:
    # random indices (primary + retry) + overlap counts + rsid flags
    # CuPy randint uses ~3x output size internally (output + 2 temp buffers)
    bytes_rand_per_target = bytes_per_target_rand * (1 + max_retries) * 3
    bytes_total_per_target = bytes_rand_per_target + bytes_per_target_aux
    total_work_bytes = bytes_total_per_target * n_targets

    # Check if single batch fits (use 60% of free VRAM after shared data)
    vram_free_now = cp.cuda.Device(device).mem_info[0]
    single_batch = total_work_bytes < int(vram_free_now * 0.60)

    if single_batch:
        # ============================================================ #
        #  SINGLE-BATCH PATH (everything fits in VRAM)                  #
        # ============================================================ #
        t0 = time.time()
        d_random, d_retry = generate_random_indices(
            n_targets, n_reps, n_loci, null_size,
            max_retries=max_retries, seed=seed,
        )
        n_primary = n_targets * n_reps * n_loci
        t_rng = time.time() - t0
        print(f"  Random number generation: {t_rng:.3f}s "
              f"({n_primary:,} primary + {n_primary * max_retries:,} retry)")

        # Allocate output
        d_overlap = cp.zeros(n_targets * n_iters, dtype=cp.uint32)

        # Launch kernel
        t0 = time.time()
        launch_reli_kernel(
            d_null=d_null,
            d_cumsum=d_cumsum,
            d_query_chr=d_query_chr,
            d_query_start=d_query_start,
            d_query_end=d_query_end,
            d_query_length=d_query_length,
            n_loci=n_loci,
            d_all_peaks_start=d_all_start,
            d_all_peaks_end=d_all_end,
            d_target_offsets=d_offsets,
            d_target_sizes=d_sizes,
            d_target_chr_offsets=d_chr_offsets,
            n_targets=n_targets,
            d_random_indices=d_random,
            d_retry_indices=d_retry,
            max_retries=max_retries,
            d_overlap_counts=d_overlap,
            n_iters=n_iters,
            n_chroms=n_chroms,
        )
        t_kernel = time.time() - t0
        total_threads = n_targets * n_iters
        grid_size = (total_threads + BLOCK_SIZE - 1) // BLOCK_SIZE
        print(f"  Kernel execution: {t_kernel:.3f}s "
              f"({total_threads:,} threads, {grid_size} blocks)")

        # Compute statistics on CPU
        t0 = time.time()
        counts_2d = cp.asnumpy(d_overlap.reshape(n_targets, n_iters))
        results = reli_stats.compute_stats_from_counts(
            counts_2d, n_loci, n_targets_corr=n_targets_corr
        )
        results["all_counts"] = counts_2d.astype(np.int32)
        t_stats = time.time() - t0
        print(f"  Statistics computation: {t_stats:.3f}s")

        # RSID overlap kernel (observed iteration)
        t0_rsid = time.time()
        d_rsid_flags = launch_rsid_kernel(
            d_query_chr=d_query_chr,
            d_query_start=d_query_start,
            d_query_end=d_query_end,
            n_loci=n_loci,
            d_all_peaks_start=d_all_start,
            d_all_peaks_end=d_all_end,
            d_target_offsets=d_offsets,
            d_target_sizes=d_sizes,
            d_target_chr_offsets=d_chr_offsets,
            n_targets=n_targets,
            n_chroms=n_chroms,
        )
        results["rsid_flags"] = cp.asnumpy(d_rsid_flags).reshape(
            n_targets, n_loci
        )
        t_rsid = time.time() - t0_rsid
        print(f"  RSID kernel: {t_rsid:.3f}s")

        print(f"  Total GPU time: "
              f"{t_transfer + t_rng + t_kernel + t_stats + t_rsid:.3f}s")

        return results

    # ================================================================ #
    #  CHUNKED PATH: split targets into VRAM-sized chunks               #
    #                                                                    #
    #  Shared GPU data (peaks, query, null model, chrom_cumsum) stays    #
    #  on GPU across all chunks.  Only the per-target arrays (offsets,   #
    #  sizes, chr_offsets) and the random indices / output buffers are   #
    #  allocated and freed per chunk.  Stats are computed ONCE on the    #
    #  concatenated results so Bonferroni uses the full target count.    #
    # ================================================================ #

    # Determine how many targets fit per chunk.
    # Per-target VRAM: random indices (primary + retry) + overlap + rsid
    max_targets = max(1, int(vram_free_now * 0.60 / bytes_total_per_target))
    n_chunks = (n_targets + max_targets - 1) // max_targets
    print(f"  Target batching: {n_targets} targets in {n_chunks} chunks "
          f"of up to {max_targets}")

    # Reshape target_chr_offsets for easy per-chunk slicing:
    # flat array of shape [n_targets * (n_chroms+1)] -> 2D [n_targets, n_chroms+1]
    chr_off_2d = targets["target_chr_offsets"].reshape(n_targets, n_chroms + 1)

    # Accumulators (CPU) for cross-chunk concatenation
    all_counts_chunks: list[np.ndarray] = []
    all_rsid_chunks: list[np.ndarray] = []

    t_rng_total = 0.0
    t_kernel_total = 0.0
    t_rsid_total = 0.0

    for chunk_idx in range(n_chunks):
        t_start = chunk_idx * max_targets
        t_end = min(t_start + max_targets, n_targets)
        chunk_n = t_end - t_start

        print(f"  Chunk {chunk_idx + 1}/{n_chunks}: targets {t_start}-"
              f"{t_end - 1} ({chunk_n} targets)")

        # -- Transfer chunk-specific target metadata to GPU --
        # Offsets keep their ORIGINAL values (index into the full peak arrays
        # that are already on GPU), so the kernel indexes correctly.
        if gpu_targets is not None:
            d_offsets_chunk = gpu_targets["d_offsets"][t_start:t_end]
            d_sizes_chunk = gpu_targets["d_sizes"][t_start:t_end]
            # chr_offsets is flattened: reshape, slice, re-flatten
            n_chroms_p1 = n_chroms + 1
            d_chr_offsets_chunk = gpu_targets["d_chr_offsets"].reshape(
                n_targets, n_chroms_p1
            )[t_start:t_end].reshape(-1)
        else:
            d_offsets_chunk = cp.asarray(
                targets["target_offsets"][t_start:t_end]
            )
            d_sizes_chunk = cp.asarray(
                targets["target_sizes"][t_start:t_end]
            )
            d_chr_offsets_chunk = cp.asarray(
                chr_off_2d[t_start:t_end].flatten().astype(np.uint32)
            )

        # -- Re-estimate max_retries for this chunk --
        chunk_n_primary = chunk_n * n_reps * n_loci
        chunk_bytes_primary = chunk_n_primary * 4
        chunk_vram_free = cp.cuda.Device(device).mem_info[0]
        chunk_vram_budget = int(chunk_vram_free * 0.75)
        chunk_max_retries = MAX_RETRIES
        chunk_peak_bytes = chunk_bytes_primary * (1 + 2 * chunk_max_retries)
        if chunk_peak_bytes > chunk_vram_budget and chunk_bytes_primary > 0:
            chunk_max_retries = max(
                0, int((chunk_vram_budget / chunk_bytes_primary - 1) / 2)
            )
            print(f"    VRAM-aware: max_retries={chunk_max_retries} "
                  f"(free: {chunk_vram_free/1e9:.1f} GB)")

        # -- Generate random indices for this chunk --
        t0 = time.time()
        # Use a different seed per chunk for statistical independence
        chunk_seed = seed + chunk_idx * 1000
        d_random_chunk, d_retry_chunk = generate_random_indices(
            chunk_n, n_reps, n_loci, null_size,
            max_retries=chunk_max_retries, seed=chunk_seed,
        )
        t_rng_total += time.time() - t0

        # -- Allocate output for this chunk --
        d_overlap_chunk = cp.zeros(chunk_n * n_iters, dtype=cp.uint32)

        # -- Launch RELI kernel --
        t0 = time.time()
        launch_reli_kernel(
            d_null=d_null,
            d_cumsum=d_cumsum,
            d_query_chr=d_query_chr,
            d_query_start=d_query_start,
            d_query_end=d_query_end,
            d_query_length=d_query_length,
            n_loci=n_loci,
            d_all_peaks_start=d_all_start,
            d_all_peaks_end=d_all_end,
            d_target_offsets=d_offsets_chunk,
            d_target_sizes=d_sizes_chunk,
            d_target_chr_offsets=d_chr_offsets_chunk,
            n_targets=chunk_n,
            d_random_indices=d_random_chunk,
            d_retry_indices=d_retry_chunk,
            max_retries=chunk_max_retries,
            d_overlap_counts=d_overlap_chunk,
            n_iters=n_iters,
            n_chroms=n_chroms,
        )
        t_kernel_total += time.time() - t0

        # -- Transfer overlap counts to CPU and free GPU buffers --
        chunk_counts = cp.asnumpy(
            d_overlap_chunk.reshape(chunk_n, n_iters)
        )
        all_counts_chunks.append(chunk_counts)

        del d_random_chunk, d_retry_chunk, d_overlap_chunk
        cp.get_default_memory_pool().free_all_blocks()

        # -- Launch RSID kernel for this chunk --
        t0 = time.time()
        d_rsid_chunk = launch_rsid_kernel(
            d_query_chr=d_query_chr,
            d_query_start=d_query_start,
            d_query_end=d_query_end,
            n_loci=n_loci,
            d_all_peaks_start=d_all_start,
            d_all_peaks_end=d_all_end,
            d_target_offsets=d_offsets_chunk,
            d_target_sizes=d_sizes_chunk,
            d_target_chr_offsets=d_chr_offsets_chunk,
            n_targets=chunk_n,
            n_chroms=n_chroms,
        )
        chunk_rsid = cp.asnumpy(d_rsid_chunk).reshape(chunk_n, n_loci)
        all_rsid_chunks.append(chunk_rsid)
        t_rsid_total += time.time() - t0

        # Free chunk-specific GPU arrays
        del d_offsets_chunk, d_sizes_chunk, d_chr_offsets_chunk, d_rsid_chunk
        cp.get_default_memory_pool().free_all_blocks()

    # -- Concatenate all chunks and compute stats ONCE --
    print(f"  Chunk totals: RNG {t_rng_total:.3f}s, "
          f"kernel {t_kernel_total:.3f}s, RSID {t_rsid_total:.3f}s")

    t0 = time.time()
    counts_2d = np.vstack(all_counts_chunks)  # (n_targets, n_iters)
    rsid_flags = np.vstack(all_rsid_chunks)   # (n_targets, n_loci)

    results = reli_stats.compute_stats_from_counts(
        counts_2d, n_loci, n_targets_corr=n_targets_corr
    )
    results["all_counts"] = counts_2d.astype(np.int32)
    results["rsid_flags"] = rsid_flags
    t_stats = time.time() - t0
    print(f"  Statistics computation: {t_stats:.3f}s")

    t_total = t_transfer + t_rng_total + t_kernel_total + t_rsid_total + t_stats
    print(f"  Total GPU time: {t_total:.3f}s")

    return results


def run_single_query(
    snp_path: str | Path,
    null_model_path: str | Path,
    target_peaks_path: str | Path,
    genome_build_path: str | Path,
    n_reps: int = N_PERMS_DEFAULT,
    seed: int = 42,
) -> dict:
    """Run RELI for one query against one target on GPU.

    Convenience wrapper that handles all file loading.  Mainly useful
    for testing and validation against C++ output.

    Parameters
    ----------
    snp_path : path
        Query BED4 file.
    null_model_path : path
        Null model file.
    target_peaks_path : path
        Single target peak BED file.
    genome_build_path : path
        Genome build file (e.g., hg19.txt).
    n_reps : int
        Number of permutations.
    seed : int
        Random seed.

    Returns
    -------
    dict with RELI statistics for the single target.
    """
    # Load genome build
    chrom_cumsum, chrom_names = reli_io.load_genome_build(genome_build_path)
    chrom_to_idx = reli_io.make_chrom_to_idx(chrom_names)
    n_chroms = len(chrom_names)

    # Load query
    query = reli_io.load_snp_file(snp_path, chrom_to_idx)

    # Load null model
    null_model = reli_io.load_null_model(null_model_path)

    # Load single target -- wrap it in the batch format
    peaks_dir = os.path.dirname(target_peaks_path)
    target_filename = os.path.basename(target_peaks_path)
    label = target_filename.replace(".bed", "")

    peaks = reli_io.load_target_peaks(peaks_dir, target_filename, chrom_to_idx)
    n_peaks = len(peaks)

    # Build minimal batch target structure
    chr_off = [0] * (n_chroms + 1)
    if n_peaks > 0:
        current_chr = -1
        for j in range(n_peaks):
            c = int(peaks["chr"][j])
            if c != current_chr:
                for fill_c in range(current_chr + 1, c + 1):
                    chr_off[fill_c] = j
                current_chr = c
        for fill_c in range(current_chr + 1, n_chroms + 1):
            chr_off[fill_c] = n_peaks

    targets = {
        "all_chr": peaks["chr"] if n_peaks > 0 else np.array([], dtype=np.uint32),
        "all_start": peaks["start"] if n_peaks > 0 else np.array([], dtype=np.uint32),
        "all_end": peaks["end"] if n_peaks > 0 else np.array([], dtype=np.uint32),
        "target_offsets": np.array([0], dtype=np.uint32),
        "target_sizes": np.array([n_peaks], dtype=np.uint32),
        "target_chr_offsets": np.array(chr_off, dtype=np.uint32),
        "target_labels": [label],
        "target_metadata": [{"source": ".", "cell": ".", "tf": "."}],
    }

    return run_batch(
        query, null_model, chrom_cumsum, targets,
        n_reps=n_reps, n_targets_corr=1, seed=seed,
    )


# ================================================================== #
#  Output Writing                                                     #
# ================================================================== #

def write_results(
    output_dir: str | Path,
    query_name: str,
    targets: dict,
    results: dict,
    rsids: dict[int, list[str]],
    corr_multiplier: int,
    n_loci: int,
    null_model_name: str,
    species: str = "hg19",
    ancestry: str = ".",
) -> None:
    """Write .stats, .overlaps, and .rsids files for each target.

    Output format matches the C++ RELI binary exactly (17-column .stats).
    Matches RELI_impl.cpp ``output()`` (line 1219).

    Parameters
    ----------
    output_dir : path
        Directory to write output files.
    query_name : str
        Phenotype / query name (e.g., "SKIP_UPintr").
    targets : dict
        Target metadata from ``load_all_targets()``.
    results : dict
        Output of ``run_batch()``.
    rsids : dict
        Output of ``collect_observed_rsids()``.
    corr_multiplier : int
        Bonferroni correction factor.
    n_loci : int
        Total number of query loci.
    null_model_name : str
        Name of the null model (for output column).
    species : str
        Genome build name (default "hg19").
    ancestry : str
        Ancestry label (default ".").
    """
    os.makedirs(output_dir, exist_ok=True)

    corr_pval = np.minimum(results["pval"] * corr_multiplier, 1.0)

    n_targets = len(targets["target_labels"])
    for t in range(n_targets):
        label = targets["target_labels"][t]
        meta = targets["target_metadata"][t]

        obs = int(results["observed"][t])
        ratio = obs / n_loci if n_loci > 0 else 0.0
        enrichment = float(results["enrichment"][t])

        # .stats file (17 columns, matching C++ RELI output)
        stats_path = os.path.join(output_dir, f"{label}.RELI.stats")
        with open(stats_path, "w") as fh:
            fh.write(
                "Phenotype\tAncestry\tLabel\tCell\tTF\tSource\t"
                "Overlap\tTotal\tRatio\tMean\tSD\tZ-score\t"
                "Enrichment\tP-value\tCorrected P-value\t"
                "Null Model\tSpecies\ttrack\n"
            )
            fh.write(
                f"{query_name}\t{ancestry}\t{label}\t"
                f"{meta['cell']}\t{meta['tf']}\t{meta['source']}\t"
                f"{obs}\t{n_loci}\t{ratio:.6f}\t"
                f"{results['mean'][t]:.6f}\t{results['sd'][t]:.6f}\t"
                f"{results['zscore'][t]:.6f}\t{enrichment:.6f}\t"
                f"{results['pval'][t]:.6e}\t{corr_pval[t]:.6e}\t"
                f"{null_model_name}\t{species}\t{label}\n"
            )

        # .overlaps file (2001 integers, one per line)
        overlaps_path = os.path.join(output_dir, f"{label}.RELI.overlaps")
        with open(overlaps_path, "w") as fh:
            for i in range(results["all_counts"].shape[1]):
                fh.write(f"{results['all_counts'][t, i]}\n")

        # .rsids file (sorted unique locus names from observed iteration)
        rsids_path = os.path.join(output_dir, f"{label}.RELI.rsids")
        with open(rsids_path, "w") as fh:
            for name in rsids.get(t, []):
                fh.write(f"{name}\n")


def write_results_consolidated(
    output_dir: str | Path,
    query_name: str,
    targets: dict,
    results: dict,
    rsids: dict[int, list[str]],
    corr_multiplier: int,
    n_loci: int,
    null_model_name: str,
    species: str = "hg19",
    ancestry: str = ".",
) -> None:
    """Write consolidated output: one .stats TSV, one .rsids TSV, one .overlaps TSV.

    Instead of 816×3 individual files, writes 3 combined files per query.
    Dramatically reduces I/O overhead for large CLIP libraries.
    """
    os.makedirs(output_dir, exist_ok=True)

    corr_pval = np.minimum(results["pval"] * corr_multiplier, 1.0)
    n_targets = len(targets["target_labels"])

    # --- Combined .stats file ---
    stats_path = os.path.join(output_dir, f"{query_name}_all_results.tsv")
    with open(stats_path, "w") as fh:
        fh.write(
            "Phenotype\tAncestry\tLabel\tCell\tRBP\tSource\t"
            "Overlap\tTotal\tRatio\tMean\tSD\tZ-score\t"
            "Enrichment\tP-value\tCorrected P-value\t"
            "Null Model\tSpecies\ttrack\n"
        )
        for t in range(n_targets):
            label = targets["target_labels"][t]
            meta = targets["target_metadata"][t]
            obs = int(results["observed"][t])
            ratio = obs / n_loci if n_loci > 0 else 0.0
            enrichment = float(results["enrichment"][t])
            fh.write(
                f"{query_name}\t{ancestry}\t{label}\t"
                f"{meta['cell']}\t{meta['tf']}\t{meta['source']}\t"
                f"{obs}\t{n_loci}\t{ratio:.6f}\t"
                f"{results['mean'][t]:.6f}\t{results['sd'][t]:.6f}\t"
                f"{results['zscore'][t]:.6f}\t{enrichment:.6f}\t"
                f"{results['pval'][t]:.6e}\t{corr_pval[t]:.6e}\t"
                f"{null_model_name}\t{species}\t{label}\n"
            )

    # --- Combined .rsids file ---
    rsids_path = os.path.join(output_dir, f"{query_name}_all_rsids.tsv")
    with open(rsids_path, "w") as fh:
        fh.write("Target\tRBP\tCell\tLocus\n")
        for t in range(n_targets):
            label = targets["target_labels"][t]
            meta = targets["target_metadata"][t]
            for name in rsids.get(t, []):
                fh.write(f"{label}\t{meta['tf']}\t{meta['cell']}\t{name}\n")

    # --- Combined .overlaps file ---
    overlaps_path = os.path.join(output_dir, f"{query_name}_all_overlaps.tsv")
    with open(overlaps_path, "w") as fh:
        fh.write("Target\tRBP\tCell\tIteration\tOverlap_Count\n")
        for t in range(n_targets):
            label = targets["target_labels"][t]
            meta = targets["target_metadata"][t]
            # Only write observed (iter 0) and a summary, not all 2001 iterations
            obs = int(results["observed"][t])
            mean_null = float(results["mean"][t])
            fh.write(f"{label}\t{meta['tf']}\t{meta['cell']}\tobserved\t{obs}\n")
            fh.write(f"{label}\t{meta['tf']}\t{meta['cell']}\tnull_mean\t{mean_null:.2f}\n")


# ================================================================== #
#  Full Pipeline: All Queries                                         #
# ================================================================== #

# Standard RBP-RELI query names and their null model mappings
QUERY_NAMES = [
    "SKIP_AltEX", "SKIP_DNintr", "SKIP_UPintr", "SKIP_merged",
    "INCL_AltEX", "INCL_DNintr", "INCL_UPintr", "INCL_merged",
]

NULL_MODEL_MAP = {
    "AltEX": "Null_Model_BG_AltEX",
    "DNintr": "Null_Model_BG_DNintr",
    "UPintr": "Null_Model_BG_UPintr",
    "merged": "Null_Model_BG_merged",
}


def run_full_pipeline(
    input_dir: str | Path,
    output_dir: str | Path,
    clip_index: str | Path,
    peaks_dir: str | Path,
    genome_build: str | Path,
    n_reps: int = N_PERMS_DEFAULT,
    corr: int = 0,
    seed: int = 42,
    device: int = 0,
    queries: list[str] | None = None,
) -> None:
    """Run the full RBP-RELI GPU pipeline: all queries vs all targets.

    This is the top-level function that replaces ``run_reli.sh``.
    It loads target peaks once, then iterates over queries, swapping
    the query loci and null model for each.

    Parameters
    ----------
    input_dir : path
        Directory containing .snp and null model files.
    output_dir : path
        Output directory.
    clip_index : path
        Path to CLIPseq.index.
    peaks_dir : path
        Directory containing target peak BED files.
    genome_build : path
        Path to genome build file (e.g., hg19.txt).
    n_reps : int
        Number of permutations (default 2000).
    corr : int
        Bonferroni correction multiplier.  0 = auto from index.
    seed : int
        Random seed.
    device : int
        CUDA device ID.
    queries : list[str], optional
        List of query names to process.  Defaults to all 8.
    """
    print("=" * 60)
    print("GPU-RELI: CUDA-accelerated permutation testing")
    print("=" * 60)

    if HAS_CUPY:
        n_devices = cp.cuda.runtime.getDeviceCount()
        print(f"GPU: {n_devices} device(s)")
        dev = cp.cuda.Device(device)
        mem_total = dev.mem_info[1]
        print(f"  Device {device}: {mem_total / 1e9:.1f} GB VRAM")
    else:
        print("WARNING: CuPy not available.  GPU kernels will fail.")

    # Load genome build
    print("\nLoading genome build...")
    chrom_cumsum, chrom_names = reli_io.load_genome_build(genome_build)
    chrom_to_idx = reli_io.make_chrom_to_idx(chrom_names)
    n_chroms = len(chrom_names)
    print(f"  {n_chroms} chromosomes, total genome: "
          f"{chrom_cumsum[-1]:,} bp")

    # Load all target peak files (one-time cost)
    print("\nLoading all target peak files...")
    t0 = time.time()
    targets = reli_io.load_all_targets(
        peaks_dir, clip_index, chrom_to_idx, n_chroms
    )
    n_targets = len(targets["target_labels"])
    n_total_peaks = len(targets["all_start"])
    print(f"  {n_targets} targets, {n_total_peaks:,} total peaks, "
          f"loaded in {time.time() - t0:.1f}s")

    # Transfer targets to GPU once (reused across all queries)
    gpu_targets = targets_to_gpu(targets, device=device)

    # Bonferroni correction factor
    corr_multiplier = corr if corr > 0 else n_targets
    print(f"  Bonferroni correction: {corr_multiplier}")

    # Determine which queries to run
    if queries is None:
        queries = QUERY_NAMES

    # Cache null models (at most 4 distinct ones)
    null_cache: dict[str, np.ndarray] = {}
    null_gpu_cache: dict[str, object] = {}

    pipeline_start = time.time()

    for query_name in queries:
        print(f"\n{'=' * 60}")
        print(f"Query: {query_name}")
        print(f"{'=' * 60}")

        # Load query loci
        snp_path = os.path.join(input_dir, f"{query_name}.snp")
        if not os.path.exists(snp_path):
            print(f"  SKIP: {snp_path} not found")
            continue

        query = reli_io.load_snp_file(snp_path, chrom_to_idx)
        n_loci = len(query["chr_idx"])
        print(f"  Loaded {n_loci} query loci")

        # Load null model (cached across queries sharing a region type)
        # Extract region from the last component: SKIP_AltEX -> AltEX,
        # Yap_SKIP_AltEX -> AltEX, INCL_merged -> merged
        region_type = query_name.rsplit("_", 1)[1]  # AltEX, DNintr, etc.
        null_name = NULL_MODEL_MAP.get(region_type)
        if null_name is None:
            print(f"  SKIP: unknown region type '{region_type}'")
            continue

        if null_name not in null_cache:
            null_path = os.path.join(input_dir, null_name)
            if not os.path.exists(null_path):
                print(f"  SKIP: null model {null_path} not found")
                continue
            null_cache[null_name] = reli_io.load_null_model(null_path)
            print(f"  Loaded null model: "
                  f"{len(null_cache[null_name]):,} positions")
        else:
            print(f"  Using cached null model: {null_name}")
        null_model = null_cache[null_name]

        # Cache null model on GPU (reused across queries with same region type)
        if null_name not in null_gpu_cache:
            null_gpu_cache[null_name] = cp.asarray(null_model)
        gpu_null = null_gpu_cache[null_name]

        # Run GPU RELI
        results = run_batch(
            query, null_model, chrom_cumsum, targets,
            n_reps=n_reps, n_targets_corr=corr_multiplier,
            seed=seed, device=device,
            gpu_targets=gpu_targets,
            gpu_null=gpu_null,
        )

        # Collect RSIDs from GPU overlap flags
        rsids = collect_observed_rsids_gpu_from_flags(query, results["rsid_flags"])

        # Write output files
        write_results_consolidated(
            output_dir, query_name, targets, results, rsids,
            corr_multiplier, n_loci, null_name,
        )
        print(f"  Wrote results to {output_dir}/{query_name}_all_results.tsv")

        # Summary
        n_sig = int(np.sum(results["corr_pval"] < 0.05))
        max_z = float(np.max(results["zscore"]))
        print(f"  Max Z-score: {max_z:.2f}, "
              f"Bonferroni-significant targets: {n_sig}")

    total_time = time.time() - pipeline_start
    print(f"\n{'=' * 60}")
    print(f"All queries complete in {total_time:.1f}s")
    print(f"{'=' * 60}")
