#!/usr/bin/env bash
# profile_kernel.sh — Nsight Compute 三刀流，一键 profile 一个 CUDA kernel
#
# Usage:
#   ./scripts/profile_kernel.sh <kernel_name>
#
# Examples:
#   ./scripts/profile_kernel.sh matmul_tiled_kernel
#   ./scripts/profile_kernel.sh softmax_warp_reduce_kernel
#
# Prerequisites:
#   - CUDA Toolkit with ncu in PATH (or auto-detected from /usr/local/cuda*/bin)
#   - Benchmark binary built: cd tests/cuda/build && cmake .. -DCMAKE_CUDA_ARCHITECTURES="75" && make -j$(nproc)
#   - WSL2: requires passwordless sudo for ncu

set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 <kernel_name>"
    echo ""
    echo "Known kernels in this project:"
    echo "  matmul_naive_kernel"
    echo "  matmul_tiled_kernel"
    echo "  softmax_warp_reduce_kernel"
    echo "  layernorm_welford_kernel"
    echo "  attention_fused_kernel_v2"
    exit 1
fi

KERNEL="$1"

# ── Find project root (script is in scripts/ dir) ──
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BINARY="$PROJECT_ROOT/tests/cuda/build/benchmark/cuda_kernel_benchmark"

if [ ! -f "$BINARY" ]; then
    echo "[ERROR] Benchmark binary not found: $BINARY"
    echo "  Build it first:"
    echo "    cd tests/cuda && mkdir -p build && cd build"
    echo '    cmake .. -DCMAKE_CUDA_ARCHITECTURES="75" && make -j$(nproc)'
    exit 1
fi

# ── Auto-detect ncu ──
NCU=""
for candidate in ncu /usr/local/cuda*/bin/ncu; do
    if command -v "$candidate" &>/dev/null; then
        NCU="$candidate"
        break
    fi
done
if [ -z "$NCU" ]; then
    echo "[ERROR] ncu not found. Install CUDA Toolkit or add ncu to PATH."
    exit 1
fi

# ── Check sudo (WSL2 needs it for perf counters) ──
SUDO=""
if ! $NCU --version &>/dev/null 2>&1; then
    if sudo -n true 2>/dev/null; then
        SUDO="sudo"
    else
        echo "[ERROR] ncu requires sudo on WSL2, but passwordless sudo is not configured."
        echo "  Run: sudo visudo -f /etc/sudoers.d/ncu"
        echo "  Add:  $(whoami) ALL=(ALL) NOPASSWD: $(which ncu 2>/dev/null || echo /usr/local/cuda/bin/ncu)"
        exit 1
    fi
fi

echo "=============================================="
echo "  Profile: $KERNEL"
echo "  Binary:  $BINARY"
echo "=============================================="

# ═══════════════════════════════════════════════════
# 第一刀：SpeedOfLight
# ═══════════════════════════════════════════════════
echo ""
echo "── 第一刀：SpeedOfLight ──"
echo ""

$SUDO $NCU --kernel-name "$KERNEL" --launch-count 1 \
    --section SpeedOfLight \
    "$BINARY"

# ═══════════════════════════════════════════════════
# 第二刀：Top 3 Stalls + SM% + Warp%
# ═══════════════════════════════════════════════════
echo ""
echo "── 第二刀：Top 3 Stalls ──"
echo ""

METRICS="sm__throughput.avg.pct_of_peak_sustained_elapsed,"
METRICS+="smsp__warps_active.avg.pct_of_peak_sustained_elapsed,"
METRICS+="smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,"
METRICS+="smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,"
METRICS+="smsp__average_warps_issue_stalled_wait_per_issue_active.ratio,"
METRICS+="smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio,"
METRICS+="smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio,"
METRICS+="smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio,"
METRICS+="smsp__average_warps_issue_stalled_membar_per_issue_active.ratio,"
METRICS+="smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active.ratio,"
METRICS+="smsp__average_warps_issue_stalled_other_per_issue_active.ratio"

$SUDO $NCU --kernel-name "$KERNEL" --launch-count 1 \
    --metrics "$METRICS" \
    "$BINARY"

echo ""
echo "── 诊断完成 ──"
echo "  SM Throughput  → 找 sm__throughput 那行"
echo "  Warp Active    → 找 warps_active 那行"
echo "  Top 3 Stalls   → stalled_ 开头的 9 行里，数值最大的 3 个"
echo "  ============="
echo "  查表: cuda/README.md → GPU Profiling → 第三刀"
