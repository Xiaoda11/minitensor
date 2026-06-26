"""Data types for CUDA kernel profiling."""

from dataclasses import dataclass, field
from typing import Optional


# ── GPU specifications ──────────────────────────────────────────────

@dataclass
class GpuSpec:
    """Theoretical peak values for a GPU model."""
    name: str
    fp32_tflops: float          # TFLOPS (CUDA core FP32)
    hbm_bandwidth_gbs: float    # GB/s
    sm_count: int               # # of SMs
    max_threads_per_sm: int = 1536
    max_blocks_per_sm: int = 16
    l2_cache_mb: float = 3.0


# Known GPU specs (used for roofline traces)
GPU_DATABASE = {
    "RTX 2060": GpuSpec(
        name="RTX 2060",
        fp32_tflops=6.5,
        hbm_bandwidth_gbs=336,
        sm_count=30,
        max_threads_per_sm=1024,
        l2_cache_mb=3.0,
    ),
    "RTX 3060": GpuSpec(
        name="RTX 3060",
        fp32_tflops=12.7,
        hbm_bandwidth_gbs=360,
        sm_count=28,
        max_threads_per_sm=1536,
        l2_cache_mb=3.0,
    ),
    "RTX 3060 Ti": GpuSpec(
        name="RTX 3060 Ti",
        fp32_tflops=16.2,
        hbm_bandwidth_gbs=448,
        sm_count=38,
        max_threads_per_sm=1536,
        l2_cache_mb=3.0,
    ),
    "RTX 3070": GpuSpec(
        name="RTX 3070",
        fp32_tflops=20.3,
        hbm_bandwidth_gbs=448,
        sm_count=46,
    ),
    "RTX 3080": GpuSpec(
        name="RTX 3080",
        fp32_tflops=29.8,
        hbm_bandwidth_gbs=760,
        sm_count=68,
    ),
    "RTX 4090": GpuSpec(
        name="RTX 4090",
        fp32_tflops=82.6,
        hbm_bandwidth_gbs=1008,
        sm_count=128,
        l2_cache_mb=72,
    ),
    "A100": GpuSpec(
        name="A100",
        fp32_tflops=19.5,
        hbm_bandwidth_gbs=1555,
        sm_count=108,
        l2_cache_mb=40,
    ),
}


def detect_gpu() -> Optional[GpuSpec]:
    """Try to auto-detect GPU via nvidia-smi and match against database."""
    import subprocess
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            timeout=5
        ).decode().strip()
        for key in sorted(GPU_DATABASE.keys(), key=len, reverse=True):
            if key in out:
                return GPU_DATABASE[key]
        # Unknown GPU — return a placeholder with conservative defaults
        print(f"[warn] Unknown GPU '{out}', using conservative defaults")
        return GpuSpec(name=out, fp32_tflops=10.0, hbm_bandwidth_gbs=400, sm_count=40)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return None


# ── Kernel profile ───────────────────────────────────────────────────

@dataclass
class KernelProfile:
    """Per-kernel metrics collected from ncu."""
    name: str                          # kernel symbol name
    sm_util: float = 0.0               # SM throughput % of peak
    dram_read_mb: float = 0.0          # DRAM bytes read (MB)
    dram_write_mb: float = 0.0         # DRAM bytes written (MB)
    l1_util: float = 0.0               # L1/TEX throughput % of peak
    warp_active: float = 0.0           # active warps % of peak
    barrier_stall: float = 0.0         # stalled on barrier ratio
    scoreboard_stall: float = 0.0      # stalled on scoreboard ratio

    # Derived fields (filled by analyzer)
    tag: str = "unknown"               # bottleneck classification
    estimated_flops: float = 0.0       # estimated total FLOPs
    arithmetic_intensity: float = 0.0  # FLOP / byte
    theoretical_bytes: float = 0.0     # theoretical minimum DRAM traffic (bytes)
    bytes_amplification: float = 0.0   # actual DRAM / theoretical (×)

    @property
    def total_dram_mb(self) -> float:
        return self.dram_read_mb + self.dram_write_mb

    @property
    def total_dram_bytes(self) -> float:
        # Nsight reports MB as mebibytes (2^20), not decimal (10^6)
        return self.total_dram_mb * 1_048_576

    @property
    def theoretical_mb(self) -> float:
        """Theoretical minimum data movement in MB (mebibytes)."""
        return self.theoretical_bytes / 1_048_576
