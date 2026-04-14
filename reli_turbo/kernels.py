"""
reli_turbo.kernels -- CuPy CUDA kernels for GPU-accelerated RELI.

This module contains the core CUDA kernel that performs permutation
testing and overlap counting entirely on the GPU.  The kernel
processes all (target, iteration) pairs in a single launch using
Approach C (Full Batch) from the design document.

Kernel design:
  - One thread per (target, iteration) pair
  - N_targets x (N_perms+1) independent tasks per launch
  - Each thread: for each query locus, either uses observed coords
    (iter=0) or samples from null model + SNPfit (iter>0), then
    binary-searches the target's sorted peaks for overlap.

Reference implementation: RELI_impl.cpp
  - SNPfit(): lines 122-161 (6-param version used in sim())
  - overlapping3(): lines 315-370 (overlap conditions at lines 329-332)
  - sim(): lines 1272-1353 (main permutation loop)
"""

from __future__ import annotations

import cupy as cp
import numpy as np

# Default CUDA block size (threads per block)
BLOCK_SIZE = 256


# ================================================================== #
#  CUDA Kernel Source Code                                            #
# ================================================================== #

RELI_KERNEL_CODE = r"""
// -------------------------------------------------------------------
// reli_permutation_kernel
//
// Each thread handles one (target, iteration) pair.
// Thread global_id = target * N_iters + iter.
//
// For iter == 0 (observed): uses actual query locus coordinates.
// For iter >  0 (permutation): samples random position from null
//   model, converts via SNPfit to chr coords, checks overlap.
//
// Overlap counting uses binary search on the target's sorted peaks
// for the relevant chromosome, then a short linear scan checking the
// four overlap conditions from RELI_impl.cpp lines 329-332.
// -------------------------------------------------------------------

extern "C" __global__
void reli_permutation_kernel(
    // Null model: sorted cumulative genomic positions
    const unsigned long long* __restrict__ null_positions,
    const unsigned int N_null,

    // Chromosome structure: cumulative prefix sums, len = N_chroms + 1
    const unsigned long long* __restrict__ chrom_cumsum,
    const unsigned int N_chroms,

    // Query loci (shared across all threads)
    const unsigned int* __restrict__ query_chr,
    const unsigned int* __restrict__ query_start,
    const unsigned int* __restrict__ query_end,
    const unsigned int* __restrict__ query_length,
    const unsigned int N_loci,

    // All target peaks (concatenated, sorted within each target)
    const unsigned int* __restrict__ all_peaks_start,
    const unsigned int* __restrict__ all_peaks_end,

    // Target indexing
    const unsigned int* __restrict__ target_offsets,
    const unsigned int* __restrict__ target_sizes,
    // Per-target chromosome offset index: flat [N_targets * (N_chroms+1)]
    const unsigned int* __restrict__ target_chr_offsets,
    const unsigned int N_targets,

    // Pre-generated random indices into the null model.
    // Layout: random_indices[(target * N_perms + perm) * N_loci + locus]
    //         where perm = iter - 1 (0-based permutation index).
    // Retry indices for rejection sampling are interleaved after primary.
    const unsigned int* __restrict__ random_indices,
    const unsigned int* __restrict__ retry_indices,
    const unsigned int MAX_RETRIES,

    // Output: overlap counts, flat [N_targets * N_iters]
    unsigned int* __restrict__ overlap_counts,
    const unsigned int N_iters
) {
    unsigned int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int target = global_id / N_iters;
    unsigned int iter   = global_id % N_iters;

    if (target >= N_targets) return;

    unsigned int peak_offset = target_offsets[target];
    unsigned int peak_count  = target_sizes[target];
    unsigned int chr_idx_base = target * (N_chroms + 1);

    // If target has no peaks, overlap is always 0.
    if (peak_count == 0) {
        overlap_counts[target * N_iters + iter] = 0;
        return;
    }

    unsigned int count = 0;
    unsigned int N_perms = N_iters - 1;  // 2000

    for (unsigned int locus = 0; locus < N_loci; locus++) {
        unsigned int locus_chr, locus_start, locus_end;

        if (iter == 0) {
            // ---------------------------------------------------------
            // OBSERVED (iteration 0): use actual query coordinates.
            // Matches RELI_impl.cpp sim() lines 1283-1296.
            // ---------------------------------------------------------
            locus_chr   = query_chr[locus];
            locus_start = query_start[locus];
            locus_end   = query_end[locus];

        } else {
            // ---------------------------------------------------------
            // PERMUTATION (iterations 1..2000):
            // 1. Draw random index into null model.
            // 2. SNPfit: convert cumulative position to (chr, start, end).
            // 3. If position doesn't fit, retry with backup indices.
            //
            // Matches RELI_impl.cpp sim() lines 1299-1330:
            //   tIndex = distGen(randSeed)
            //   datagood = SNPfit(LD, keySNP, bin0[tIndex], ...)
            //   while (!datagood) { retry }
            // ---------------------------------------------------------
            unsigned int locus_len = query_length[locus];
            unsigned int perm_idx = iter - 1;  // 0-based
            bool fit = false;

            // Compute flat index for this (target, perm, locus) triple
            unsigned long long rand_flat =
                ((unsigned long long)target * N_perms + perm_idx)
                * N_loci + locus;

            // Primary random sample
            unsigned int null_idx = random_indices[rand_flat];
            unsigned long long cum_pos = null_positions[null_idx];

            // SNPfit: scan chromosomes to find where cum_pos lands.
            // Matches RELI_impl.cpp SNPfit() lines 146-153.
            // For singleton LD blocks (RBP-RELI), max_diff = 0.
            // Fit condition: cum_pos - length >= chrom_cumsum[c]
            //            AND cum_pos + length <= chrom_cumsum[c+1]
            for (unsigned int c = 0; c < N_chroms; c++) {
                unsigned long long lo = chrom_cumsum[c];
                unsigned long long hi = chrom_cumsum[c + 1];
                if (cum_pos >= lo + locus_len && cum_pos + locus_len <= hi) {
                    locus_chr = c;
                    // Coordinate conversion (RELI_impl.cpp line 149):
                    //   snp_end = int_A + floor(length/2) - cumsum[k]
                    //   snp_start = snp_end - length
                    locus_end   = (unsigned int)(cum_pos + (locus_len / 2) - lo);
                    locus_start = locus_end - locus_len;
                    fit = true;
                    break;
                }
            }

            // Rejection sampling: retry with backup random indices.
            // The C++ code loops: while(!datagood) { draw new random index }.
            if (!fit) {
                for (unsigned int retry = 0; retry < MAX_RETRIES && !fit; retry++) {
                    unsigned long long retry_flat = rand_flat * MAX_RETRIES + retry;
                    null_idx = retry_indices[retry_flat];
                    cum_pos = null_positions[null_idx];

                    for (unsigned int c = 0; c < N_chroms; c++) {
                        unsigned long long lo = chrom_cumsum[c];
                        unsigned long long hi = chrom_cumsum[c + 1];
                        if (cum_pos >= lo + locus_len && cum_pos + locus_len <= hi) {
                            locus_chr = c;
                            locus_end   = (unsigned int)(cum_pos + (locus_len / 2) - lo);
                            locus_start = locus_end - locus_len;
                            fit = true;
                            break;
                        }
                    }
                }
            }

            // If still no fit after all retries, skip this locus for
            // this permutation.  This matches the C++ behaviour where
            // the loop would eventually find a valid position -- but
            // with finite retries on GPU we accept the (very rare) miss.
            if (!fit) continue;
        }

        // ---------------------------------------------------------
        // OVERLAP DETECTION via binary search + linear scan.
        //
        // Look up this target's peaks for the locus's chromosome
        // using the per-target chromosome offset index.
        //
        // Matches RELI_impl.cpp overlapping3() lines 315-370.
        // ---------------------------------------------------------
        unsigned int chr_start_rel = target_chr_offsets[chr_idx_base + locus_chr];
        unsigned int chr_end_rel   = target_chr_offsets[chr_idx_base + locus_chr + 1];
        unsigned int chr_start_abs = peak_offset + chr_start_rel;
        unsigned int chr_end_abs   = peak_offset + chr_end_rel;

        // No peaks on this chromosome for this target.
        if (chr_start_abs >= chr_end_abs) continue;

        // Binary search: find insertion point for locus_start in the
        // sorted peak start array.  This narrows the scan window.
        unsigned int blo = chr_start_abs;
        unsigned int bhi = chr_end_abs;
        while (blo < bhi) {
            unsigned int mid = blo + (bhi - blo) / 2;
            if (all_peaks_start[mid] < locus_start) {
                blo = mid + 1;
            } else {
                bhi = mid;
            }
        }

        // Scan backward: peaks before 'blo' might still overlap if
        // they are wide (their end extends past locus_start).
        // C++ uses lookback_step = 50 (RELI_impl.h line 417).
        unsigned int scan_start =
            (blo > chr_start_abs + 50) ? blo - 50 : chr_start_abs;

        bool found_overlap = false;
        for (unsigned int k = scan_start; k < chr_end_abs && !found_overlap; k++) {
            unsigned int bs = all_peaks_start[k];
            unsigned int be = all_peaks_end[k];

            // Four overlap conditions matching RELI_impl.cpp lines 329-332:
            // 1. Peak fully contains locus:  bs <= ls && be >= le
            // 2. Peak start inside locus:    bs >= ls && bs+1 < le
            // 3. Peak end inside locus:      be > ls+1 && be <= le
            // 4. Locus fully contains peak:  bs >= ls && be <= le
            //
            // The +1 offsets prevent adjacent (touching) intervals
            // from being counted as overlapping.
            if ((bs <= locus_start && be >= locus_end) ||
                (bs >= locus_start && bs + 1 < locus_end) ||
                (be > locus_start + 1 && be <= locus_end) ||
                (bs >= locus_start && be <= locus_end)) {
                found_overlap = true;
            }

            // Early exit: if peak start is past locus end, no more
            // overlaps are possible (peaks are sorted by start).
            // Placed AFTER overlap check to match C++ (line 340).
            if (bs >= locus_end) break;
        }

        if (found_overlap) {
            count++;  // One overlap per locus is sufficient (break in C++)
        }
    }

    // Write the overlap count for this (target, iteration) pair.
    overlap_counts[target * N_iters + iter] = count;
}
""";


# ================================================================== #
#  Kernel Compilation and Launch Helpers                              #
# ================================================================== #

_compiled_kernel = None


def get_kernel() -> cp.RawKernel:
    """Compile and cache the RELI permutation kernel.

    Returns
    -------
    kernel : cp.RawKernel
        Compiled CUDA kernel ready for launch.
    """
    global _compiled_kernel
    if _compiled_kernel is None:
        _compiled_kernel = cp.RawKernel(
            RELI_KERNEL_CODE,
            "reli_permutation_kernel",
        )
    return _compiled_kernel


def launch_reli_kernel(
    *,
    d_null: cp.ndarray,
    d_cumsum: cp.ndarray,
    d_query_chr: cp.ndarray,
    d_query_start: cp.ndarray,
    d_query_end: cp.ndarray,
    d_query_length: cp.ndarray,
    n_loci: int,
    d_all_peaks_start: cp.ndarray,
    d_all_peaks_end: cp.ndarray,
    d_target_offsets: cp.ndarray,
    d_target_sizes: cp.ndarray,
    d_target_chr_offsets: cp.ndarray,
    n_targets: int,
    d_random_indices: cp.ndarray,
    d_retry_indices: cp.ndarray,
    max_retries: int,
    d_overlap_counts: cp.ndarray,
    n_iters: int,
    n_chroms: int,
    block_size: int = BLOCK_SIZE,
) -> None:
    """Launch the RELI permutation kernel on the GPU.

    All ``d_*`` parameters are CuPy device arrays already on the GPU.
    This function is synchronous -- it waits for the kernel to finish
    before returning.

    Parameters
    ----------
    d_null : cp.ndarray, uint64
        Null model cumulative positions.
    d_cumsum : cp.ndarray, uint64
        Chromosome cumulative prefix sums.
    d_query_chr, d_query_start, d_query_end, d_query_length : cp.ndarray, uint32
        Query locus arrays.
    n_loci : int
        Number of query loci.
    d_all_peaks_start, d_all_peaks_end : cp.ndarray, uint32
        Concatenated target peak start/end arrays.
    d_target_offsets, d_target_sizes, d_target_chr_offsets : cp.ndarray, uint32
        Target indexing arrays.
    n_targets : int
        Number of targets.
    d_random_indices : cp.ndarray, uint32
        Pre-generated random indices for permutation sampling.
    d_retry_indices : cp.ndarray, uint32
        Backup random indices for rejection sampling retries.
    max_retries : int
        Maximum number of rejection sampling retries per locus.
    d_overlap_counts : cp.ndarray, uint32
        Output array, shape (n_targets * n_iters,).
    n_iters : int
        Total iterations (n_perms + 1).
    n_chroms : int
        Number of chromosomes.
    block_size : int
        CUDA threads per block (default 256).
    """
    kernel = get_kernel()

    total_threads = n_targets * n_iters
    grid_size = (total_threads + block_size - 1) // block_size

    kernel(
        (grid_size,),
        (block_size,),
        (
            d_null,
            np.uint32(len(d_null)),
            d_cumsum,
            np.uint32(n_chroms),
            d_query_chr,
            d_query_start,
            d_query_end,
            d_query_length,
            np.uint32(n_loci),
            d_all_peaks_start,
            d_all_peaks_end,
            d_target_offsets,
            d_target_sizes,
            d_target_chr_offsets,
            np.uint32(n_targets),
            d_random_indices,
            d_retry_indices,
            np.uint32(max_retries),
            d_overlap_counts,
            np.uint32(n_iters),
        ),
    )
    cp.cuda.Stream.null.synchronize()


def generate_random_indices(
    n_targets: int,
    n_perms: int,
    n_loci: int,
    null_size: int,
    max_retries: int = 10,
    seed: int = 42,
) -> tuple[cp.ndarray, cp.ndarray]:
    """Pre-generate random null model indices on the GPU.

    Returns primary random indices and retry (backup) indices for
    rejection sampling.

    Parameters
    ----------
    n_targets : int
        Number of targets.
    n_perms : int
        Number of permutations (e.g., 2000).
    n_loci : int
        Number of query loci.
    null_size : int
        Size of the null model (number of positions).
    max_retries : int
        Number of retry indices per primary sample.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    d_random_indices : cp.ndarray, uint32
        Primary random indices, shape (n_targets * n_perms * n_loci,).
    d_retry_indices : cp.ndarray, uint32
        Retry indices, shape (n_targets * n_perms * n_loci * max_retries,).
    """
    cp.random.seed(seed)
    n_primary = n_targets * n_perms * n_loci
    n_retry = n_primary * max_retries

    d_random = cp.random.randint(0, null_size, size=n_primary, dtype=cp.uint32)
    d_retry = cp.random.randint(0, null_size, size=n_retry, dtype=cp.uint32)

    return d_random, d_retry


# ================================================================== #
#  RSID Overlap Kernel (observed iteration only)                      #
# ================================================================== #

RSID_KERNEL_CODE = r"""
// -------------------------------------------------------------------
// rsid_overlap_kernel
//
// One thread per target.  For each target, loops over all query loci
// using OBSERVED coordinates and writes a per-locus overlap flag.
// This replaces the slow CPU collect_observed_rsids() function.
//
// Output: overlap_flags[target * N_loci + locus] = 1 if overlapping.
// Uses the same binary search + 4 overlap conditions as the main kernel.
// -------------------------------------------------------------------

extern "C" __global__
void rsid_overlap_kernel(
    const unsigned int* __restrict__ query_chr,
    const unsigned int* __restrict__ query_start,
    const unsigned int* __restrict__ query_end,
    const unsigned int N_loci,
    const unsigned int* __restrict__ all_peaks_start,
    const unsigned int* __restrict__ all_peaks_end,
    const unsigned int* __restrict__ target_offsets,
    const unsigned int* __restrict__ target_sizes,
    const unsigned int* __restrict__ target_chr_offsets,
    const unsigned int N_targets,
    const unsigned int N_chroms,
    unsigned char* __restrict__ overlap_flags  // output: [N_targets * N_loci]
) {
    unsigned int target = blockIdx.x * blockDim.x + threadIdx.x;
    if (target >= N_targets) return;

    unsigned int peak_offset = target_offsets[target];
    unsigned int peak_count  = target_sizes[target];
    unsigned int chr_idx_base = target * (N_chroms + 1);
    unsigned int flag_base = target * N_loci;

    if (peak_count == 0) return;

    for (unsigned int locus = 0; locus < N_loci; locus++) {
        unsigned int locus_chr   = query_chr[locus];
        unsigned int locus_start = query_start[locus];
        unsigned int locus_end   = query_end[locus];

        // Look up this target's peaks for the locus's chromosome
        unsigned int chr_start_rel = target_chr_offsets[chr_idx_base + locus_chr];
        unsigned int chr_end_rel   = target_chr_offsets[chr_idx_base + locus_chr + 1];
        unsigned int chr_start_abs = peak_offset + chr_start_rel;
        unsigned int chr_end_abs   = peak_offset + chr_end_rel;

        if (chr_start_abs >= chr_end_abs) continue;

        // Binary search: find insertion point for locus_start
        unsigned int blo = chr_start_abs;
        unsigned int bhi = chr_end_abs;
        while (blo < bhi) {
            unsigned int mid = blo + (bhi - blo) / 2;
            if (all_peaks_start[mid] < locus_start) {
                blo = mid + 1;
            } else {
                bhi = mid;
            }
        }

        // Scan backward: peaks before 'blo' might still overlap
        unsigned int scan_start =
            (blo > chr_start_abs + 50) ? blo - 50 : chr_start_abs;

        bool found_overlap = false;
        for (unsigned int k = scan_start; k < chr_end_abs && !found_overlap; k++) {
            unsigned int bs = all_peaks_start[k];
            unsigned int be = all_peaks_end[k];

            // Four overlap conditions (RELI_impl.cpp lines 329-332)
            if ((bs <= locus_start && be >= locus_end) ||
                (bs >= locus_start && bs + 1 < locus_end) ||
                (be > locus_start + 1 && be <= locus_end) ||
                (bs >= locus_start && be <= locus_end)) {
                found_overlap = true;
            }

            if (bs >= locus_end) break;
        }

        if (found_overlap) {
            overlap_flags[flag_base + locus] = 1;
        }
    }
}
""";

_compiled_rsid_kernel = None


def get_rsid_kernel() -> cp.RawKernel:
    """Compile and cache the RSID overlap kernel."""
    global _compiled_rsid_kernel
    if _compiled_rsid_kernel is None:
        _compiled_rsid_kernel = cp.RawKernel(
            RSID_KERNEL_CODE,
            "rsid_overlap_kernel",
        )
    return _compiled_rsid_kernel


def launch_rsid_kernel(
    *,
    d_query_chr: cp.ndarray,
    d_query_start: cp.ndarray,
    d_query_end: cp.ndarray,
    n_loci: int,
    d_all_peaks_start: cp.ndarray,
    d_all_peaks_end: cp.ndarray,
    d_target_offsets: cp.ndarray,
    d_target_sizes: cp.ndarray,
    d_target_chr_offsets: cp.ndarray,
    n_targets: int,
    n_chroms: int,
    block_size: int = BLOCK_SIZE,
) -> cp.ndarray:
    """Launch the RSID overlap kernel on the GPU.

    One thread per target.  Returns a flat uint8 array of overlap flags.

    Returns
    -------
    d_flags : cp.ndarray, uint8, shape (n_targets * n_loci,)
        1 where query locus overlaps target, 0 otherwise.
    """
    kernel = get_rsid_kernel()

    d_flags = cp.zeros(n_targets * n_loci, dtype=cp.uint8)

    grid_size = (n_targets + block_size - 1) // block_size

    kernel(
        (grid_size,),
        (block_size,),
        (
            d_query_chr,
            d_query_start,
            d_query_end,
            np.uint32(n_loci),
            d_all_peaks_start,
            d_all_peaks_end,
            d_target_offsets,
            d_target_sizes,
            d_target_chr_offsets,
            np.uint32(n_targets),
            np.uint32(n_chroms),
            d_flags,
        ),
    )
    cp.cuda.Stream.null.synchronize()

    return d_flags
