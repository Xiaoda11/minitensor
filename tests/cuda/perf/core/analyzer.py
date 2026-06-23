"""Analyzer: group kernels by workload type, estimate FLOPs, compute AI."""

from collections import defaultdict
from typing import Optional

from .types import KernelProfile


# ── FLOP estimation by kernel type ────────────────────────────────────
#
# These formulas estimate total FLOPs from kernel name + known shapes.
# Shapes are inferred from the kernel name suffix or passed explicitly.
# Callers can override by setting profile.estimated_flops directly.

# Default shapes used by minitensor benchmarks (can be overridden)
DEFAULT_SHAPES = {
    "vector_add":   16_777_216,         # N = 16M
    "matmul_naive": (1024, 1024, 1024),  # M, K, N
    "matmul_tiled": (1024, 1024, 1024),
    "softmax":      (1024, 1024),        # rows, cols
    "layernorm":    (1024, 1024),
    "attention":    (1, 128, 64),        # batch, S, D
}


def estimate_flops(kernel_name: str, shapes: Optional[dict] = None) -> float:
    """Estimate total FLOPs for a kernel based on its name and known workload type.

    Args:
        kernel_name: The kernel symbol name (e.g. 'matmul_tiled_kernel').
        shapes: Optional dict overriding DEFAULT_SHAPES per workload type.

    Returns:
        Estimated FLOPs, or 0.0 if the kernel type is unrecognized.
    """
    sh = {**DEFAULT_SHAPES, **(shapes or {})}
    name_lower = kernel_name.lower()

    if "matmul" in name_lower:
        M, K, N = sh.get("matmul_tiled", sh.get("matmul_naive", (1024, 1024, 1024)))
        return 2.0 * M * K * N

    if "attention" in name_lower:
        B, S, D = sh.get("attention", (1, 128, 64))
        # QK^T: 2*S²*D, softmax: ~8*S², PV: 2*S²*D
        return B * (4.0 * S * S * D + 8.0 * S * S)

    if "softmax" in name_lower:
        R, C = sh.get("softmax", (1024, 1024))
        return 8.0 * R * C

    if "layernorm" in name_lower:
        R, C = sh.get("layernorm", (1024, 1024))
        return 8.0 * R * C

    if "vector_add" in name_lower:
        N = sh.get("vector_add", 16_777_216)
        return float(N)

    return 0.0


def compute_ai(profile: KernelProfile) -> float:
    """Compute arithmetic intensity = FLOPs / DRAM bytes.

    Uses estimated_flops from the profile and actual DRAM traffic from ncu.
    Returns 0.0 if DRAM traffic is 0 (e.g. ncu failed).
    """
    if profile.total_dram_bytes <= 0 or profile.estimated_flops <= 0:
        return 0.0
    return profile.estimated_flops / profile.total_dram_bytes


# ── Workload grouping ─────────────────────────────────────────────────

def group_kernels(profiles: list[KernelProfile]) -> dict[str, list[KernelProfile]]:
    """Group kernel profiles by workload type based on kernel name.

    Groups: matmul, softmax, attention, layernorm, kv_cache, vector_add, other.
    """
    groups: dict[str, list[KernelProfile]] = defaultdict(list)

    for p in profiles:
        nl = p.name.lower()
        if "matmul" in nl:
            groups["matmul"].append(p)
        elif "softmax" in nl:
            groups["softmax"].append(p)
        elif "attention" in nl:
            groups["attention"].append(p)
        elif "layernorm" in nl:
            groups["layernorm"].append(p)
        elif "kv" in nl or "kv_cache" in nl:
            groups["kv_cache"].append(p)
        elif "vector_add" in nl:
            groups["vector_add"].append(p)
        else:
            groups["other"].append(p)

    return dict(groups)


# ── Summarization ─────────────────────────────────────────────────────

def summarize(groups: dict[str, list[KernelProfile]]) -> dict:
    """Compute per-group summary statistics.

    Error profiles (tag starting with 'error:' or 'parse_error') are
    excluded from aggregate statistics but counted separately.

    Returns:
        {group_name: {count, avg_sm, avg_ai, avg_dram_mb, tags, kernels, errors}}
    """
    summary = {}
    for k, profiles in groups.items():
        # Split successful vs error profiles
        ok = [p for p in profiles
              if not p.tag.startswith("error:") and p.tag != "parse_error"]
        err = [p for p in profiles
               if p.tag.startswith("error:") or p.tag == "parse_error"]
        n_ok = len(ok)
        summary[k] = {
            "count": len(profiles),
            "errors": len(err),
            "avg_sm": sum(p.sm_util for p in ok) / n_ok if n_ok else 0.0,
            "avg_dram_mb": sum(p.total_dram_mb for p in ok) / n_ok if n_ok else 0.0,
            "avg_ai": (
                sum(p.arithmetic_intensity for p in ok if p.arithmetic_intensity > 0)
                / max(1, sum(1 for p in ok if p.arithmetic_intensity > 0))
            ) if n_ok else None,
            "tags": [p.tag for p in ok],
            "kernels": [p.name for p in profiles],
        }
    return summary


def annotate_flops_and_ai(profiles: list[KernelProfile],
                          shapes: Optional[dict] = None) -> None:
    """Mutate profiles in-place: estimate FLOPs and compute AI for each."""
    for p in profiles:
        p.estimated_flops = estimate_flops(p.name, shapes)
        p.arithmetic_intensity = compute_ai(p)
