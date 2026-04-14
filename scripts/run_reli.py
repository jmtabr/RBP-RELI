#!/usr/bin/env python3
"""Unified RELI runner -- Docker-first for both GPU and C++ backends.

Usage:
    # GPU backend (default) -- Docker with reli-turbo:latest
    python run_reli.py --input-dir DIR --output-dir DIR --backend gpu \\
        --index CLIPseq.index --data peaks_dir/ --build hg19.txt \\
        [--reps 2000] [--corr 0] [--queries Q1 Q2 ...] \\
        [--docker-image-gpu reli-turbo:latest]

    # C++ backend -- Docker with reli:patched
    python run_reli.py --input-dir DIR --output-dir DIR --backend cpp \\
        --index CLIPseq.index --data peaks_dir/ --build hg19.txt \\
        [--reps 2000] [--corr 0] [--queries Q1 Q2 ...] \\
        [--threads 16] [--docker-image-cpp reli:patched]

    # GPU local mode (no Docker, requires CuPy + reli_turbo installed)
    python run_reli.py --input-dir DIR --output-dir DIR --backend gpu --local \\
        --index CLIPseq.index --data peaks_dir/ --build hg19.txt

    # C++ direct binary (no Docker)
    python run_reli.py --input-dir DIR --output-dir DIR --backend cpp \\
        --reli-binary /path/to/RELI \\
        --index CLIPseq.index --data peaks_dir/ --build hg19.txt
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

QuerySpec = namedtuple("QuerySpec", ["name", "snp_path", "null_path", "region"])

NULL_MODEL_MAP = {
    "AltEX": "Null_Model_BG_AltEX",
    "DNintr": "Null_Model_BG_DNintr",
    "UPintr": "Null_Model_BG_UPintr",
    "merged": "Null_Model_BG_merged",
    "5UTR": "Null_Model_BG_5UTR",
    "CDS": "Null_Model_BG_CDS",
    "intron": "Null_Model_BG_intron",
    "3UTR": "Null_Model_BG_3UTR",
}


# ---------------------------------------------------------------------------
# Query discovery
# ---------------------------------------------------------------------------

def discover_queries(input_dir: str, explicit_queries: list[str] | None = None) -> list[QuerySpec]:
    """Discover query .snp files and their matching null models.

    Priority:
      1. If explicit_queries is provided, use those names.
      2. If manifest.json exists, read queries from it.
      3. Filename convention: glob *.snp, infer region from suffix.

    Returns list of QuerySpec namedtuples.
    """
    input_path = Path(input_dir)

    if explicit_queries:
        specs = []
        for name in explicit_queries:
            snp = input_path / f"{name}.snp"
            region = name.rsplit("_", 1)[-1]
            null_name = NULL_MODEL_MAP.get(region)
            if null_name is None:
                print(f"WARNING: Unknown region '{region}' for query '{name}', skipping")
                continue
            null = input_path / null_name
            specs.append(QuerySpec(name=name, snp_path=str(snp), null_path=str(null), region=region))
        return specs

    manifest_path = input_path / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as fh:
            manifest = json.load(fh)
        specs = []
        for q in manifest["queries"]:
            specs.append(QuerySpec(
                name=q["name"],
                snp_path=str(input_path / q["snp_file"]),
                null_path=str(input_path / q["null_model"]),
                region=q["region"],
            ))
        print(f"Discovered {len(specs)} queries from manifest.json")
        return specs

    # Filename convention fallback
    specs = []
    for snp_file in sorted(input_path.glob("*.snp")):
        name = snp_file.stem
        region = name.rsplit("_", 1)[-1]
        null_name = NULL_MODEL_MAP.get(region)
        if null_name is None:
            print(f"WARNING: Cannot infer region for '{name}', skipping")
            continue
        null = input_path / null_name
        specs.append(QuerySpec(name=name, snp_path=str(snp_file), null_path=str(null), region=region))
    print(f"Discovered {len(specs)} queries from *.snp filenames")
    return specs


def count_index_targets(index_path: str) -> int:
    """Count non-header lines in CLIPseq.index."""
    with open(index_path) as fh:
        return sum(1 for i, _ in enumerate(fh) if i > 0)


def count_snp_loci(snp_path: str) -> int:
    """Count lines in a .snp file."""
    with open(snp_path) as fh:
        return sum(1 for line in fh if line.strip())


# ---------------------------------------------------------------------------
# Docker volume-mount helpers
# ---------------------------------------------------------------------------

def _to_docker_path(host_path: str) -> str:
    """Convert a host path to a Docker-compatible mount source.

    On MSYS2/Git Bash, virtual paths like /tmp resolve differently for
    Python (C:\\tmp) vs the shell (C:\\Users\\...\\AppData\\Local\\Temp).
    We use ``cygpath -w`` when available to get the true Windows path,
    then normalise to forward-slash ``C:/...`` form for Docker.
    """
    p = os.path.abspath(host_path)
    # On MSYS2/Git Bash, try cygpath for correct resolution
    if sys.platform == "win32":
        try:
            result = subprocess.run(
                ["cygpath", "-w", host_path],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                p = result.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass  # cygpath not available, use abspath
    return p.replace("\\", "/")


def _docker_mounts(
    input_dir: str,
    output_dir: str,
    index_path: str,
    peaks_dir: str,
    build_path: str,
) -> tuple[list[str], dict[str, str]]:
    """Compute Docker ``-v`` args and a dict of container-side paths.

    Returns
    -------
    volume_args : list[str]
        Flat list like ``["-v", "host:container", "-v", ...]``.
    cpaths : dict
        Keys: ``input_dir``, ``output_dir``, ``index_file``, ``peaks_dir``,
        ``peaks_parent``, ``build_file``, ``build_dir``.
    """
    volume_args: list[str] = []
    cpaths: dict[str, str] = {}

    # -- input_dir  -> /workspace/inputs
    volume_args += ["-v", f"{_to_docker_path(input_dir)}:/workspace/inputs"]
    cpaths["input_dir"] = "/workspace/inputs"

    # -- output_dir -> /workspace/output
    volume_args += ["-v", f"{_to_docker_path(output_dir)}:/workspace/output"]
    cpaths["output_dir"] = "/workspace/output"

    # -- peaks_dir and index: the index file may live inside the peaks_dir
    #    or in a parent directory.  We mount the *parent* of both so that
    #    relative references inside the index keep working.
    peaks_abs = os.path.abspath(peaks_dir)
    index_abs = os.path.abspath(index_path)
    index_dirname = os.path.dirname(index_abs)
    peaks_dirname = os.path.basename(peaks_abs)

    # Find a common ancestor to mount
    if os.path.normcase(index_dirname) == os.path.normcase(peaks_abs):
        # index lives inside peaks_dir -> mount peaks_dir as /workspace/clip
        volume_args += ["-v", f"{_to_docker_path(peaks_abs)}:/workspace/clip"]
        cpaths["index_file"] = f"/workspace/clip/{os.path.basename(index_abs)}"
        cpaths["peaks_dir"] = "/workspace/clip"
    elif os.path.normcase(peaks_abs).startswith(os.path.normcase(index_dirname)):
        # peaks_dir is inside the dir containing the index
        # mount index_dirname as /workspace/clip
        volume_args += ["-v", f"{_to_docker_path(index_dirname)}:/workspace/clip"]
        rel_peaks = os.path.relpath(peaks_abs, index_dirname).replace("\\", "/")
        cpaths["index_file"] = f"/workspace/clip/{os.path.basename(index_abs)}"
        cpaths["peaks_dir"] = f"/workspace/clip/{rel_peaks}"
    elif os.path.normcase(index_dirname).startswith(os.path.normcase(peaks_abs)):
        # index dir is inside peaks_dir
        volume_args += ["-v", f"{_to_docker_path(peaks_abs)}:/workspace/clip"]
        rel_index = os.path.relpath(index_abs, peaks_abs).replace("\\", "/")
        cpaths["index_file"] = f"/workspace/clip/{rel_index}"
        cpaths["peaks_dir"] = "/workspace/clip"
    else:
        # Separate mounts for index parent and peaks_dir
        volume_args += ["-v", f"{_to_docker_path(peaks_abs)}:/workspace/clip/{peaks_dirname}"]
        volume_args += ["-v", f"{_to_docker_path(index_dirname)}:/workspace/clip_idx"]
        cpaths["index_file"] = f"/workspace/clip_idx/{os.path.basename(index_abs)}"
        cpaths["peaks_dir"] = f"/workspace/clip/{peaks_dirname}"

    # -- build file -> /workspace/genome/<filename>
    build_abs = os.path.abspath(build_path)
    build_dir_host = os.path.dirname(build_abs)
    volume_args += ["-v", f"{_to_docker_path(build_dir_host)}:/workspace/genome"]
    cpaths["build_file"] = f"/workspace/genome/{os.path.basename(build_abs)}"
    cpaths["build_dir"] = "/workspace/genome"

    return volume_args, cpaths


def _docker_env() -> dict[str, str]:
    """Return an env dict with MSYS_NO_PATHCONV=1 for Windows/Git-Bash."""
    env = os.environ.copy()
    env["MSYS_NO_PATHCONV"] = "1"
    return env


# ---------------------------------------------------------------------------
# GPU backend -- Docker
# ---------------------------------------------------------------------------

def _run_gpu_docker_query(
    spec: QuerySpec,
    output_dir: str,
    volume_args: list[str],
    cpaths: dict[str, str],
    n_reps: int,
    corr: int,
    docker_image: str,
) -> subprocess.CompletedProcess:
    """Run a single GPU-RELI query inside a Docker container."""
    snp_filename = os.path.basename(spec.snp_path)
    null_filename = os.path.basename(spec.null_path)
    query_output = f"{cpaths['output_dir']}/{spec.name}"

    cmd = [
        "docker", "run", "--rm", "--gpus", "all",
        *volume_args,
        docker_image,
        "-snp", f"{cpaths['input_dir']}/{snp_filename}",
        "-null", f"{cpaths['input_dir']}/{null_filename}",
        "-out", query_output,
        "-phenotype", spec.name,
        "-ancestry", ".",
        "-index", cpaths["index_file"],
        "-data", cpaths["peaks_dir"],
        "-build", cpaths["build_file"],
        "-rep", str(n_reps),
        "-corr", str(corr),
    ]

    env = _docker_env()
    return subprocess.run(cmd, env=env, capture_output=True, text=True)


def run_gpu_docker(
    queries: list[QuerySpec],
    output_dir: str,
    index_path: str,
    peaks_dir: str,
    build_path: str,
    n_reps: int,
    corr: int,
    docker_image: str,
) -> None:
    """Run RELI using the GPU backend via Docker (reli-turbo:latest).

    Each query is a separate ``docker run`` invocation in single-query mode.
    The GPU image processes ALL targets in one invocation per query.
    """
    print("=" * 60)
    print(f"RELI Runner: GPU backend (Docker: {docker_image})")
    print("=" * 60)

    n_targets = count_index_targets(index_path)
    corr_multiplier = corr if corr > 0 else n_targets
    print(f"  {n_targets} CLIP targets, Bonferroni correction: {corr_multiplier}")

    os.makedirs(output_dir, exist_ok=True)
    volume_args, cpaths = _docker_mounts(
        input_dir=os.path.dirname(queries[0].snp_path),
        output_dir=output_dir,
        index_path=index_path,
        peaks_dir=peaks_dir,
        build_path=build_path,
    )

    pipeline_start = time.time()

    for qi, spec in enumerate(queries, 1):
        print(f"\n{'=' * 60}")
        print(f"[{qi}/{len(queries)}] {spec.name}")
        print(f"{'=' * 60}")

        if not os.path.exists(spec.snp_path):
            print(f"  SKIP: SNP file not found: {spec.snp_path}")
            continue
        if not os.path.exists(spec.null_path):
            print(f"  SKIP: Null model not found: {spec.null_path}")
            continue

        n_loci = count_snp_loci(spec.snp_path)
        if n_loci == 0:
            print(f"  WARNING: 0 loci in {spec.snp_path}, skipping")
            continue
        print(f"  {n_loci} query loci")

        # Check for existing consolidated output (resume-safe)
        result_tsv = os.path.join(output_dir, spec.name, f"{spec.name}_all_results.tsv")
        if os.path.exists(result_tsv):
            print(f"  SKIP: Output already exists: {result_tsv}")
            continue

        query_out_host = os.path.join(output_dir, spec.name)
        os.makedirs(query_out_host, exist_ok=True)

        t0 = time.time()
        result = _run_gpu_docker_query(
            spec, output_dir, volume_args, cpaths,
            n_reps, corr_multiplier, docker_image,
        )
        dt = time.time() - t0

        if result.returncode != 0:
            print(f"  ERROR (exit {result.returncode}):")
            stderr_lines = (result.stderr or "").strip().split("\n")
            for line in stderr_lines[-10:]:
                print(f"    {line}")
        else:
            # Print the last few stdout lines for summary
            stdout_lines = (result.stdout or "").strip().split("\n")
            for line in stdout_lines[-5:]:
                print(f"  {line}")
            print(f"  Completed in {dt:.1f}s")

    total_time = time.time() - pipeline_start
    print(f"\n{'=' * 60}")
    print(f"All {len(queries)} queries complete in {total_time:.1f}s")
    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# GPU backend -- local (no Docker, requires CuPy + reli_turbo)
# ---------------------------------------------------------------------------

def run_gpu_local(
    queries: list[QuerySpec],
    output_dir: str,
    index_path: str,
    peaks_dir: str,
    build_path: str,
    n_reps: int,
    corr: int,
    seed: int,
    device: int,
) -> None:
    """Run RELI using the GPU backend via local import (no Docker)."""
    # Ensure reli_turbo is importable when running from scripts/ directory
    _repo_root = str(Path(__file__).resolve().parent.parent)
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)

    try:
        from reli_turbo import io as reli_io
        from reli_turbo.reli import (
            collect_observed_rsids_gpu_from_flags,
            run_batch,
            targets_to_gpu,
            write_results_consolidated,
        )
    except ImportError:
        print("ERROR: reli_turbo not importable. Install with: pip install -e .")
        print("Use Docker mode (remove --local) or install dependencies.")
        sys.exit(1)

    try:
        import cupy as cp
    except ImportError:
        print("ERROR: CuPy not available. GPU local backend requires CuPy.")
        print("Install with: pip install cupy-cuda12x")
        print("Or use Docker mode (remove --local).")
        sys.exit(1)

    import numpy as np

    print("=" * 60)
    print("RELI Runner: GPU backend (local, no Docker)")
    print("=" * 60)

    n_devices = cp.cuda.runtime.getDeviceCount()
    dev = cp.cuda.Device(device)
    mem_total = dev.mem_info[1]
    print(f"GPU: {n_devices} device(s), device {device}: {mem_total / 1e9:.1f} GB VRAM")

    # Load genome build
    print("\nLoading genome build...")
    chrom_cumsum, chrom_names = reli_io.load_genome_build(build_path)
    chrom_to_idx = reli_io.make_chrom_to_idx(chrom_names)
    n_chroms = len(chrom_names)
    print(f"  {n_chroms} chromosomes, total: {chrom_cumsum[-1]:,} bp")

    # Load all targets (one-time)
    print("\nLoading all target peak files...")
    t0 = time.time()
    targets = reli_io.load_all_targets(peaks_dir, index_path, chrom_to_idx, n_chroms)
    n_targets = len(targets["target_labels"])
    print(f"  {n_targets} targets, {len(targets['all_start']):,} total peaks, "
          f"loaded in {time.time() - t0:.1f}s")

    # Cache targets on GPU
    gpu_targets = targets_to_gpu(targets, device=device)

    # Bonferroni correction
    corr_multiplier = corr if corr > 0 else n_targets
    print(f"  Bonferroni correction: {corr_multiplier}")

    # Null model cache (CPU + GPU)
    null_cache: dict[str, np.ndarray] = {}
    null_gpu_cache: dict[str, object] = {}

    os.makedirs(output_dir, exist_ok=True)
    pipeline_start = time.time()

    for qi, spec in enumerate(queries, 1):
        print(f"\n{'=' * 60}")
        print(f"[{qi}/{len(queries)}] {spec.name}")
        print(f"{'=' * 60}")

        if not os.path.exists(spec.snp_path):
            print(f"  SKIP: SNP file not found: {spec.snp_path}")
            continue
        if not os.path.exists(spec.null_path):
            print(f"  SKIP: Null model not found: {spec.null_path}")
            continue

        # Load query loci
        query = reli_io.load_snp_file(spec.snp_path, chrom_to_idx)
        n_loci = len(query["chr_idx"])
        if n_loci == 0:
            print(f"  WARNING: 0 loci in {spec.snp_path}, skipping")
            continue
        print(f"  {n_loci} query loci")

        # Load null model (cached by region)
        null_name = os.path.basename(spec.null_path)
        if null_name not in null_cache:
            null_cache[null_name] = reli_io.load_null_model(spec.null_path)
            null_gpu_cache[null_name] = cp.asarray(null_cache[null_name])
            print(f"  Loaded null model: {len(null_cache[null_name]):,} positions")
        else:
            print(f"  Using cached null model: {null_name}")

        # Run GPU RELI
        t0 = time.time()
        results = run_batch(
            query, null_cache[null_name], chrom_cumsum, targets,
            n_reps=n_reps, n_targets_corr=corr_multiplier,
            seed=seed, device=device,
            gpu_targets=gpu_targets,
            gpu_null=null_gpu_cache[null_name],
        )
        dt = time.time() - t0

        # Collect RSIDs
        rsids = collect_observed_rsids_gpu_from_flags(query, results["rsid_flags"])

        # Write consolidated output
        write_results_consolidated(
            output_dir, spec.name, targets, results, rsids,
            corr_multiplier, n_loci, null_name,
        )

        n_sig = int(np.sum(results["corr_pval"] < 0.05))
        max_z = float(np.max(results["zscore"]))
        print(f"  {n_targets} targets in {dt:.1f}s | max Z={max_z:.2f} | "
              f"{n_sig} Bonferroni-significant")

    total_time = time.time() - pipeline_start
    print(f"\n{'=' * 60}")
    print(f"All {len(queries)} queries complete in {total_time:.1f}s")
    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# C++ backend -- Docker
# ---------------------------------------------------------------------------

def _run_one_cpp_target_binary(
    target: str,
    spec: QuerySpec,
    out_dir: str,
    index_path: str,
    peaks_dir: str,
    build_path: str,
    dbsnp_path: str,
    n_reps: int,
    corr: int,
    reli_binary: str,
) -> tuple[str, bool]:
    """Run C++ RELI for a single target via direct binary. Returns (target, success)."""
    stats_file = os.path.join(out_dir, f"{target}.RELI.stats")
    if os.path.exists(stats_file):
        return (target, True)  # resume-safe

    cmd = [
        reli_binary,
        "-snp", spec.snp_path,
        "-null", spec.null_path,
        "-ld", dbsnp_path,
        "-dbsnp", dbsnp_path,
        "-index", index_path,
        "-data", peaks_dir,
        "-target", target,
        "-build", build_path,
        "-out", out_dir,
        "-rep", str(n_reps),
        "-corr", str(corr),
        "-phenotype", spec.name,
    ]

    env = _docker_env()
    try:
        subprocess.run(cmd, env=env, capture_output=True, timeout=300)
        return (target, os.path.exists(stats_file))
    except Exception:
        return (target, False)


def _consolidate_cpp_stats(out_dir: str, query_name: str, output_dir: str) -> str | None:
    """Concatenate per-target .RELI.stats into a single TSV matching GPU output format.

    C++ RELI output columns:
        Formal Phenotype, Ancestry, Source, Cell, Formal Cell, Label,
        Intersect, Total, Ratio, Mean, Std, Z-score, Relative Risk,
        P-val, Corrected P-val, Null_Model, Species

    GPU output columns (target format):
        Phenotype, Ancestry, Label, Cell, RBP, Source, Overlap, Total,
        Ratio, Mean, SD, Z-score, Enrichment, P-value, Corrected P-value,
        Null Model, Species, track
    """
    result_path = os.path.join(output_dir, f"{query_name}_all_results.tsv")
    stats_files = sorted(Path(out_dir).glob("*.RELI.stats"))
    if not stats_files:
        return None

    # Column mapping: C++ name -> GPU name
    col_rename = {
        "Formal Phenotype": "Phenotype",
        "Formal Cell": None,  # drop
        "Label": "RBP",
        "Intersect": "Overlap",
        "Std": "SD",
        "Relative Risk": "Enrichment",
        "P-val": "P-value",
        "Corrected P-val": "Corrected P-value",
        "Null_Model": "Null Model",
    }

    # GPU column order
    gpu_columns = [
        "Phenotype", "Ancestry", "Label", "Cell", "RBP", "Source",
        "Overlap", "Total", "Ratio", "Mean", "SD", "Z-score",
        "Enrichment", "P-value", "Corrected P-value", "Null Model",
        "Species", "track",
    ]

    rows = []
    for sf in stats_files:
        with open(sf) as fh:
            header_line = fh.readline().strip().split("\t")
            data_line = fh.readline().strip().split("\t")
            if len(data_line) < len(header_line):
                continue
            row = dict(zip(header_line, data_line))

            # Rename columns
            renamed = {}
            for old_name, value in row.items():
                new_name = col_rename.get(old_name, old_name)
                if new_name is not None:
                    renamed[new_name] = value

            # Add "Label" = the BED filename (from the stats filename)
            bed_name = sf.name.replace(".RELI.stats", "")
            renamed["Label"] = bed_name

            # Add "track" column (GPU includes it, C++ doesn't)
            renamed.setdefault("track", bed_name)

            rows.append(renamed)

    if not rows:
        return None

    with open(result_path, "w", newline="\n") as out:
        out.write("\t".join(gpu_columns) + "\n")
        for row in rows:
            values = [str(row.get(col, ".")) for col in gpu_columns]
            out.write("\t".join(values) + "\n")

    return result_path


def run_cpp(
    queries: list[QuerySpec],
    output_dir: str,
    index_path: str,
    peaks_dir: str,
    build_path: str,
    n_reps: int,
    corr: int,
    threads: int,
    docker_image: str,
    reli_binary: str | None,
) -> None:
    """Run RELI using the C++ backend (Docker or direct binary).

    Each target is a separate invocation.  Parallelism via ThreadPoolExecutor.
    """
    use_docker = reli_binary is None

    print("=" * 60)
    if use_docker:
        print(f"RELI Runner: C++ backend (Docker: {docker_image})")
    else:
        print(f"RELI Runner: C++ backend (binary: {reli_binary})")
    print("=" * 60)

    # Read target list from index
    targets = []
    with open(index_path) as fh:
        fh.readline()  # skip header
        for line in fh:
            parts = line.strip().split("\t")
            if parts and parts[0]:
                targets.append(parts[0])
    n_targets = len(targets)
    print(f"  {n_targets} CLIP targets")

    corr_multiplier = corr if corr > 0 else n_targets
    print(f"  Bonferroni correction: {corr_multiplier}")

    os.makedirs(output_dir, exist_ok=True)

    # Ensure dummy_dbsnp exists in input_dir (needed by C++ RELI for -dbsnp/-ld)
    input_dir = os.path.dirname(queries[0].snp_path) if queries else output_dir
    dbsnp_path = os.path.join(input_dir, "dummy_dbsnp")
    if not os.path.exists(dbsnp_path):
        open(dbsnp_path, "w").close()

    # Compute Docker mounts once (reused for all targets)
    if use_docker:
        volume_args, cpaths = _docker_mounts(
            input_dir=input_dir,
            output_dir=output_dir,
            index_path=index_path,
            peaks_dir=peaks_dir,
            build_path=build_path,
        )

    pipeline_start = time.time()

    for qi, spec in enumerate(queries, 1):
        print(f"\n{'=' * 60}")
        print(f"[{qi}/{len(queries)}] {spec.name}: {n_targets} targets...")
        print(f"{'=' * 60}")

        if not os.path.exists(spec.snp_path):
            print(f"  SKIP: SNP file not found: {spec.snp_path}")
            continue
        if not os.path.exists(spec.null_path):
            print(f"  SKIP: Null model not found: {spec.null_path}")
            continue

        n_loci = count_snp_loci(spec.snp_path)
        if n_loci == 0:
            print(f"  WARNING: 0 loci in {spec.snp_path}, skipping")
            continue

        out_dir = os.path.join(output_dir, spec.name)
        os.makedirs(out_dir, exist_ok=True)

        # Filter to remaining targets (resume-safe)
        remaining = [t for t in targets
                     if not os.path.exists(os.path.join(out_dir, f"{t}.RELI.stats"))]
        print(f"  {n_loci} loci | {len(remaining)}/{n_targets} remaining targets")

        if remaining:
            t0 = time.time()
            completed = 0
            failed = 0

            if use_docker:
                # Single container with internal xargs parallelism
                snp_fname = os.path.basename(spec.snp_path)
                null_fname = os.path.basename(spec.null_path)
                target_list = "\n".join(remaining)

                # Write target list to a temp file in input_dir so the container can read it
                # Use Unix line endings (\n only) to avoid xargs parsing issues
                target_list_path = os.path.join(input_dir, f".targets_{spec.name}.txt")
                with open(target_list_path, "w", newline="\n") as f:
                    for t in remaining:
                        f.write(t + "\n")

                ci = cpaths["input_dir"]
                co = cpaths["output_dir"]
                bash_script = (
                    f'mkdir -p {co}/{spec.name} && '
                    f'> /tmp/dummy_dbsnp && > /tmp/dummy_ld && '
                    f'cat {ci}/.targets_{spec.name}.txt | xargs -P {threads} -I {{}} '
                    f'/reli/RELI '
                    f'-snp {ci}/{snp_fname} '
                    f'-null {ci}/{null_fname} '
                    f'-ld /tmp/dummy_ld '
                    f'-dbsnp /tmp/dummy_dbsnp '
                    f'-index {cpaths["index_file"]} '
                    f'-data {cpaths["peaks_dir"]} '
                    f'-target {{}} '
                    f'-build {cpaths["build_file"]} '
                    f'-out {co}/{spec.name} '
                    f'-rep {n_reps} -corr {corr_multiplier} '
                    f'-phenotype {spec.name} '
                    f'2>/dev/null'
                )

                cmd = (
                    ["docker", "run", "--rm"]
                    + volume_args
                    + [docker_image, "bash", "-c", bash_script]
                )

                # Debug: print the command
                print(f"    CMD: {' '.join(cmd[:8])} ... bash -c '<{len(bash_script)} chars>'")
                print(f"    BASH: {bash_script[:200]}...")
                result = subprocess.run(
                    cmd, env=_docker_env(), capture_output=True, text=True,
                    timeout=3600,
                )
                print(f"    STDOUT: {result.stdout[:200] if result.stdout else '(empty)'}")
                if result.returncode != 0:
                    # Show last few lines of stderr for debugging
                    stderr_lines = result.stderr.strip().split("\n")
                    for line in stderr_lines[-5:]:
                        print(f"    STDERR: {line}")
                # Count successes
                completed = sum(
                    1 for t in remaining
                    if os.path.exists(os.path.join(out_dir, f"{t}.RELI.stats"))
                )
                failed = len(remaining) - completed
                print(f"    {completed}/{len(remaining)} ok, {failed} failed")

            else:
                with ThreadPoolExecutor(max_workers=threads) as executor:
                    futures = {
                        executor.submit(
                            _run_one_cpp_target_binary,
                            t, spec, out_dir,
                            index_path, peaks_dir, build_path,
                            dbsnp_path, n_reps, corr_multiplier,
                            reli_binary,
                        ): t
                        for t in remaining
                    }
                    for future in as_completed(futures):
                        target_name, success = future.result()
                        if success:
                            completed += 1
                        else:
                            failed += 1
                        total_done = completed + failed
                        if total_done % 50 == 0 or total_done == len(remaining):
                            print(f"    Progress: {total_done}/{len(remaining)} "
                                  f"({completed} ok, {failed} failed)")

            dt = time.time() - t0
            print(f"  Completed in {dt:.1f}s ({completed} ok, {failed} failed)")

        # Consolidate per-target stats into single TSV
        result_path = _consolidate_cpp_stats(out_dir, spec.name, output_dir)
        if result_path:
            n_results = sum(1 for _ in open(result_path)) - 1
            print(f"  Consolidated {n_results} results -> {result_path}")

    total_time = time.time() - pipeline_start
    print(f"\n{'=' * 60}")
    print(f"All {len(queries)} queries complete in {total_time:.1f}s")
    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# Auto-parse results
# ---------------------------------------------------------------------------

def run_parse(output_dir: str, categories: str | None, queries: list[QuerySpec]) -> None:
    """Run parse_results logic on completed RELI output."""
    script_dir = Path(__file__).resolve().parent
    parse_script = script_dir / "parse_results.py"

    cmd = [sys.executable, str(parse_script), output_dir]
    if categories:
        cmd.extend(["--categories", categories])
    query_names = [q.name for q in queries]
    if query_names:
        cmd.extend(["--queries"] + query_names)

    print(f"\nRunning parse_results...")
    subprocess.run(cmd, check=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_reli.py",
        description="Unified RELI runner: Docker-first for GPU and C++ backends.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required
    p.add_argument("--input-dir", required=True, help="Directory with .snp and null model files")
    p.add_argument("--output-dir", required=True, help="Output directory for results")
    p.add_argument("--index", required=True, help="Path to CLIPseq.index")
    p.add_argument("--data", required=True, help="Directory containing CLIP peak BED files")
    p.add_argument("--build", required=True, help="Path to genome build file (e.g., hg19.txt)")

    # Backend selection
    p.add_argument("--backend", choices=["gpu", "cpp"], default="gpu",
                   help="Execution backend (default: gpu)")
    p.add_argument("--local", action="store_true",
                   help="GPU only: use local import instead of Docker (requires CuPy)")

    # Shared options
    p.add_argument("--reps", type=int, default=2000, help="Permutations (default: 2000)")
    p.add_argument("--corr", type=int, default=0,
                   help="Bonferroni correction (0 = auto from index)")
    p.add_argument("--queries", nargs="+", default=None,
                   help="Specific query names to run (default: auto-discover)")

    # Docker image options
    docker = p.add_argument_group("Docker options")
    docker.add_argument("--docker-image-gpu", default="reli-turbo:latest",
                        help="GPU Docker image (default: reli-turbo:latest)")
    docker.add_argument("--docker-image-cpp", default="reli:patched",
                        help="C++ Docker image (default: reli:patched)")

    # GPU options
    gpu = p.add_argument_group("GPU backend options")
    gpu.add_argument("--device", type=int, default=0, help="CUDA device ID (default: 0)")
    gpu.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    # C++ options
    cpp = p.add_argument_group("C++ backend options")
    cpp.add_argument("--threads", type=int, default=16,
                     help="Parallel targets for C++ backend (default: 16)")
    cpp.add_argument("--reli-binary", default=None,
                     help="Path to RELI binary (skips Docker for C++ backend)")

    # Post-processing
    post = p.add_argument_group("Post-processing")
    post.add_argument("--no-parse", action="store_true",
                      help="Skip automatic result parsing")
    post.add_argument("--categories", default=None,
                      help="Path to RBP category TSV for parse_results")

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Discover queries
    queries = discover_queries(args.input_dir, args.queries)
    if not queries:
        print("ERROR: No queries discovered. Check --input-dir contents.")
        sys.exit(1)

    print(f"Queries to run ({len(queries)}): {', '.join(q.name for q in queries)}")

    # Validate files exist
    missing = []
    for q in queries:
        if not os.path.exists(q.snp_path):
            missing.append(q.snp_path)
        if not os.path.exists(q.null_path):
            missing.append(q.null_path)
    if missing:
        print(f"WARNING: {len(missing)} missing files (will be skipped at runtime):")
        for m in missing[:5]:
            print(f"  {m}")

    # Dispatch to backend
    if args.backend == "gpu":
        if args.local:
            run_gpu_local(
                queries=queries,
                output_dir=args.output_dir,
                index_path=args.index,
                peaks_dir=args.data,
                build_path=args.build,
                n_reps=args.reps,
                corr=args.corr,
                seed=args.seed,
                device=args.device,
            )
        else:
            run_gpu_docker(
                queries=queries,
                output_dir=args.output_dir,
                index_path=args.index,
                peaks_dir=args.data,
                build_path=args.build,
                n_reps=args.reps,
                corr=args.corr,
                docker_image=args.docker_image_gpu,
            )
    else:
        run_cpp(
            queries=queries,
            output_dir=args.output_dir,
            index_path=args.index,
            peaks_dir=args.data,
            build_path=args.build,
            n_reps=args.reps,
            corr=args.corr,
            threads=args.threads,
            docker_image=args.docker_image_cpp,
            reli_binary=args.reli_binary,
        )

    # Auto-parse
    if not args.no_parse:
        run_parse(args.output_dir, args.categories, queries)


if __name__ == "__main__":
    main()
