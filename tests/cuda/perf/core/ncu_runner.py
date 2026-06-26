"""Nsight Compute runner — wraps ncu CLI, parses output, discovers kernels."""

import subprocess
import re
import json
import shutil
from typing import Optional

from .types import KernelProfile
from .classifier import classify

# ── Metrics to collect ──────────────────────────────────────────────

METRICS = (
    "sm__throughput.avg.pct_of_peak_sustained_elapsed,"
    "dram__bytes_read.sum,"
    "dram__bytes_write.sum,"
    "smsp__warps_active.avg.pct_of_peak_sustained_elapsed,"
    "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio,"
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,"
    "l1tex__throughput.avg.pct_of_peak_sustained_elapsed"
)


# ── Kernel discovery ────────────────────────────────────────────────

def _resolve_ncu() -> str:
    ncu = shutil.which("ncu")
    if ncu:
        return ncu

    for candidate in (
        "/usr/local/cuda/bin/ncu",
        "/usr/local/cuda-13.2/bin/ncu",
        "/usr/local/cuda-12/bin/ncu",
    ):
        ncu = shutil.which(candidate)
        if ncu:
            return ncu

    raise FileNotFoundError("ncu")


def list_kernels(binary: str) -> list[str]:
    """Discover all kernel symbols in a CUDA binary.

    Uses cuobjdump to extract __global__ function symbols.
    Falls back to ncu --list-sets if cuobjdump is unavailable.
    """
    # Method 1: cuobjdump (fast, no GPU needed)
    try:
        out = subprocess.check_output(
            ["cuobjdump", "-symbols", binary],
            stderr=subprocess.STDOUT,
            timeout=10
        ).decode()
        kernels = []
        for line in out.splitlines():
            # STT_FUNC entries are functions; kernel names typically end in _kernel
            if "STT_FUNC" in line:
                parts = line.split()
                if parts:
                    name = parts[-1]
                    if "_kernel" in name or name.startswith("_"):
                        kernels.append(name)
        if kernels:
            return sorted(set(kernels))
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    # Method 2: ncu --list-kernels (requires GPU)
    try:
        out = subprocess.check_output(
            ["ncu", "--list-kernels", binary],
            stderr=subprocess.STDOUT,
            timeout=30
        ).decode()
        kernels = []
        for line in out.splitlines():
            line = line.strip()
            if line and not line.startswith("==") and not line.startswith("Kernel"):
                # Format: "  kernel_name" or "kernel_name"
                name = line.strip()
                if name:
                    kernels.append(name)
        return sorted(set(kernels))
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    return []


# ── NCU execution ────────────────────────────────────────────────────

def run_ncu(binary: str, kernel: str,
            timeout: int = 120,
            sudo: bool = False) -> KernelProfile:
    """Run ncu for a single kernel and parse the output.

    Args:
        binary: Path to the CUDA benchmark binary.
        kernel: Kernel symbol name to profile.
        timeout: Max seconds to wait for ncu.
        sudo: Whether to prefix with sudo (needed on WSL2 for perf counters).

    Returns:
        KernelProfile with parsed metrics. Fields are 0.0 if parsing failed.

    Raises:
        subprocess.CalledProcessError: if ncu exits non-zero.
        FileNotFoundError: if ncu is not installed.
        subprocess.TimeoutExpired: if ncu hangs.
    """
    try:
        cmd = [_resolve_ncu()]
        if sudo:
            cmd.insert(0, "sudo")

        cmd += [
            "--kernel-name", kernel,
            "--launch-count", "1",
            "--metrics", METRICS,
            binary,
        ]

        out = subprocess.check_output(
            cmd,
            stderr=subprocess.STDOUT,
            timeout=timeout
        ).decode()
    except subprocess.CalledProcessError as e:
        stderr = e.output.decode() if e.output else str(e)
        raise RuntimeError(
            f"ncu failed for kernel '{kernel}': {stderr[:500]}"
        ) from e
    except FileNotFoundError:
        raise RuntimeError(
            "ncu (Nsight Compute CLI) not found. "
            "Install CUDA Toolkit or ensure ncu is in PATH."
        )

    return _parse(out, kernel)


def _parse(text: str, kernel: str) -> KernelProfile:
    """Parse ncu text output into a KernelProfile.

    Uses regex to extract each metric value from the ncu output table.
    ncu prints a table with columns: metric_name, metric_unit, value.
    """

    def extract(pattern: str, text: str, group: int = 1) -> float:
        m = re.search(pattern, text)
        return float(m.group(group)) if m else 0.0

    prof = KernelProfile(
        name=kernel,
        # SM throughput % — matches "sm__throughput ... 96.42%"
        sm_util=extract(r"sm__throughput.*?([0-9]+\.?[0-9]*)", text),
        # DRAM read — matches "dram__bytes_read.sum   Mbyte   123.45"
        dram_read_mb=extract(r"dram__bytes_read\.sum\s+Mbyte\s+([0-9]+\.?[0-9]*)", text),
        # DRAM write
        dram_write_mb=extract(r"dram__bytes_write\.sum\s+Mbyte\s+([0-9]+\.?[0-9]*)", text),
        # Active warps %
        warp_active=extract(r"warps_active.*?([0-9]+\.?[0-9]*)", text),
        # Barrier stall ratio
        barrier_stall=extract(r"barrier.*?([0-9]+\.?[0-9]*)", text),
        # Scoreboard stall ratio
        scoreboard_stall=extract(r"scoreboard.*?([0-9]+\.?[0-9]*)", text),
        # L1/TEX throughput %
        l1_util=extract(r"l1tex__throughput.*?([0-9]+\.?[0-9]*)", text),
    )

    # Detect all-zero profile → regex parse failure, not actually balanced
    if prof.sm_util == 0.0 and prof.dram_read_mb == 0.0 and prof.warp_active == 0.0:
        prof.tag = "parse_error"
        return prof

    prof.tag = classify(prof)
    return prof


# ── Batch execution ──────────────────────────────────────────────────

def run_batch(binary: str, kernels: list[str],
              timeout_per_kernel: int = 120,
              sudo: bool = False,
              verbose: bool = True) -> list[KernelProfile]:
    """Run ncu for multiple kernels, with progress reporting.

    Args:
        binary: Path to CUDA benchmark binary.
        kernels: List of kernel symbol names.
        timeout_per_kernel: Max seconds per ncu invocation.
        sudo: Whether to use sudo.
        verbose: Print progress to stderr.

    Returns:
        List of KernelProfile objects (one per kernel, even if ncu failed).
    """
    profiles = []
    for i, k in enumerate(kernels, 1):
        if verbose:
            print(f"[{i}/{len(kernels)}] profiling {k} ...", flush=True)
        try:
            prof = run_ncu(binary, k, timeout=timeout_per_kernel, sudo=sudo)
            profiles.append(prof)
        except Exception as e:
            if verbose:
                print(f"  [warn] {e}", flush=True)
            # Create an error marker profile
            err = KernelProfile(name=k)
            err.tag = f"error: {str(e)[:80]}"
            profiles.append(err)

    return profiles
