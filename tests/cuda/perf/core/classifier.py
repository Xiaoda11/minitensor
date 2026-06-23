"""Bottleneck classifier using Nsight Compute metrics.

Rule-based classification with tunable thresholds.
Uses the 7 metrics collected by ncu_runner to tag each kernel as:
  compute_bound, memory_bound, latency_bound, sync_bound, or balanced.
"""

from .types import KernelProfile

# Tunable thresholds (can be overridden per GPU)
THRESHOLDS = {
    "sm_util_compute": 80.0,       # SM util > this + low DRAM → compute
    "dram_read_memory": 200.0,     # DRAM read > this (MB) → memory
    "scoreboard_latency": 5.0,     # scoreboard stall > this (%) → latency
    "barrier_sync": 3.0,           # barrier stall > this (%) → sync
}


def classify(profile: KernelProfile) -> str:
    """Classify a single kernel profile into a bottleneck category.

    Priority order: compute > memory > latency > sync > balanced.
    """
    p = profile

    # compute-bound: SM is busy AND not waiting on DRAM
    if p.sm_util > THRESHOLDS["sm_util_compute"] and p.dram_read_mb < THRESHOLDS["dram_read_memory"]:
        return "compute_bound"

    # memory-bound: heavy DRAM traffic
    if p.dram_read_mb > THRESHOLDS["dram_read_memory"]:
        return "memory_bound"

    # latency-bound: stalled waiting for global memory (scoreboard)
    if p.scoreboard_stall > THRESHOLDS["scoreboard_latency"]:
        return "latency_bound"

    # sync-bound: stalled on __syncthreads() barriers
    if p.barrier_stall > THRESHOLDS["barrier_sync"]:
        return "sync_bound"

    return "balanced"


# Human-readable tag descriptions for reporting
TAG_DESCRIPTIONS = {
    "compute_bound": "ALU limited — SM throughput high, DRAM traffic low",
    "memory_bound":  "HBM bandwidth limited — DRAM read > 200 MB",
    "latency_bound": "Memory latency limited — scoreboard stall dominant",
    "sync_bound":    "Barrier synchronisation limited — __syncthreads() overhead",
    "balanced":      "No single bottleneck dominates",
    "parse_error":   "[ERROR] ncu output parsing failed — regex mismatch, check raw output",
}
