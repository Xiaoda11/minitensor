#!/usr/bin/env python3
"""minitensor-perf — CUDA kernel performance profiling and analysis.

Usage:
    # Profile specific kernels
    python main.py --bin ./cuda_kernel_benchmark --kernels matmul_tiled_kernel

    # Auto-discover and profile all kernels
    python main.py --bin ./cuda_kernel_benchmark --auto

    # Save results to JSON
    python main.py --bin ./cuda_kernel_benchmark --auto --output report.json

    # Also generate roofline plot
    python main.py --bin ./cuda_kernel_benchmark --auto --roofline roofline.png

    # Use sudo (required on WSL2 for GPU perf counters)
    python main.py --bin ./cuda_kernel_benchmark --auto --sudo
"""

import argparse
import json
import sys
from pathlib import Path

from core.types import detect_gpu, GpuSpec
from core.ncu_runner import run_ncu, run_batch, list_kernels
from core.analyzer import group_kernels, summarize, annotate_flops_and_ai
from core.classifier import TAG_DESCRIPTIONS


def main():
    parser = argparse.ArgumentParser(
        description="minitensor-perf — CUDA kernel profiling & roofline analysis"
    )
    parser.add_argument(
        "--bin", required=True,
        help="Path to CUDA benchmark binary"
    )
    parser.add_argument(
        "--kernels", default=None,
        help="Comma-separated kernel names to profile (e.g. 'matmul_tiled_kernel,softmax_naive_kernel')"
    )
    parser.add_argument(
        "--auto", action="store_true",
        help="Auto-discover kernels in binary via cuobjdump/ncu"
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Save results as JSON to this path"
    )
    parser.add_argument(
        "--roofline", default=None,
        help="Generate roofline plot and save to this path (e.g. 'roofline.png')"
    )
    parser.add_argument(
        "--gpu", default=None,
        help="GPU model override for roofline (e.g. 'RTX 2060', 'RTX 3060')"
    )
    parser.add_argument(
        "--sudo", action="store_true",
        help="Prefix ncu with sudo (required on WSL2 for perf counters)"
    )
    parser.add_argument(
        "--timeout", type=int, default=120,
        help="Timeout per kernel in seconds (default: 120)"
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress per-kernel progress output"
    )
    parser.add_argument(
        "--list-kernels", action="store_true",
        help="List all kernel symbols in the binary and exit"
    )

    args = parser.parse_args()

    # ── Binary check ──
    binary = Path(args.bin)
    if not binary.exists():
        print(f"[error] binary not found: {binary}", file=sys.stderr)
        sys.exit(1)

    # ── List-kernels mode ──
    if args.list_kernels:
        kernels = list_kernels(str(binary))
        if kernels:
            print(f"Found {len(kernels)} kernel(s):")
            for k in kernels:
                print(f"  {k}")
        else:
            print("No kernels found. Is this a CUDA binary?")
        return

    # ── Determine kernel list ──
    if args.auto:
        kernels = list_kernels(str(binary))
        if not kernels:
            print("[error] --auto mode failed to discover kernels. "
                  "Try --kernels to specify manually.", file=sys.stderr)
            sys.exit(1)
        print(f"Auto-discovered {len(kernels)} kernel(s): {', '.join(kernels)}")
    elif args.kernels:
        kernels = [k.strip() for k in args.kernels.split(",") if k.strip()]
        if not kernels:
            print("[error] --kernels specified but empty", file=sys.stderr)
            sys.exit(1)
    else:
        print("[error] Specify --kernels or --auto", file=sys.stderr)
        sys.exit(1)

    # ── Detect GPU ──
    from core.types import GPU_DATABASE
    gpu = None
    if args.gpu:
        gpu = GPU_DATABASE.get(args.gpu)
        if gpu is None:
            print(f"[warn] Unknown GPU '{args.gpu}', using auto-detect", file=sys.stderr)
    if gpu is None:
        gpu = detect_gpu()

    if gpu:
        print(f"\nGPU: {gpu.name} "
              f"({gpu.fp32_tflops:.1f} TFLOPS FP32, "
              f"{gpu.hbm_bandwidth_gbs:.0f} GB/s HBM)")
    else:
        print("\n[note] No GPU detected (nvidia-smi unavailable), roofline will use defaults")

    # ── Sudo pre-check ──
    if args.sudo:
        import subprocess
        try:
            subprocess.run(
                ["sudo", "-n", "true"],
                check=True, capture_output=True, timeout=5
            )
        except subprocess.CalledProcessError:
            print("[error] --sudo requires passwordless sudo (NOPASSWD).\n"
                  "  Run: sudo visudo -f /etc/sudoers.d/ncu\n"
                  "  Add:  <user> ALL=(ALL) NOPASSWD: /usr/local/cuda/bin/ncu\n"
                  "  Or run the script with sudo directly.", file=sys.stderr)
            sys.exit(1)
        except FileNotFoundError:
            print("[error] sudo not found", file=sys.stderr)
            sys.exit(1)

    # ── Profile ──
    print(f"\nProfiling {len(kernels)} kernel(s) ...\n")
    profiles = run_batch(
        str(binary), kernels,
        timeout_per_kernel=args.timeout,
        sudo=args.sudo,
        verbose=not args.quiet,
    )

    # ── Annotate with FLOPs and AI ──
    annotate_flops_and_ai(profiles)

    # ── Group & Summarize ──
    groups = group_kernels(profiles)
    summary = summarize(groups)

    # ── Print results ──
    print("\n" + "=" * 70)
    print("  PER-KERNEL RESULTS")
    print("=" * 70)
    print(f"{'Kernel':<35} {'SM%':>6} {'DRAM(MB)':>10} {'AI':>8} {'Tag':<18}")
    print("-" * 70)

    for p in profiles:
        tag_clean = p.tag.split(":")[0].strip() if ":" in p.tag else p.tag
        print(
            f"{p.name:<35} "
            f"{p.sm_util:>5.1f}% "
            f"{p.total_dram_mb:>9.1f} "
            f"{p.arithmetic_intensity:>7.1f} "
            f"{tag_clean:<18}"
        )

    # ── GROUP SUMMARY ──
    print("\n" + "=" * 70)
    print("  GROUP SUMMARY")
    print("=" * 70)
    print(f"{'Group':<15} {'Count':>5} {'Err':>3} {'Avg SM%':>8} {'Avg AI':>8} {'Tags'}")
    print("-" * 70)

    for group_name, stats in sorted(summary.items()):
        tag_summary = ", ".join(
            f"{t}({stats['tags'].count(t)})"
            for t in sorted(set(stats["tags"]))
        ) if stats["tags"] else "—"
        avg_ai = stats["avg_ai"]
        ai_str = f"{avg_ai:.1f}" if avg_ai is not None and avg_ai > 0 else "N/A"
        print(
            f"{group_name:<15} "
            f"{stats['count']:>5} "
            f"{stats.get('errors', 0):>3} "
            f"{stats['avg_sm']:>7.1f}% "
            f"{ai_str:>8} "
            f"{tag_summary}"
        )

    # ── Bottleneck legend ──
    print("\n" + "=" * 70)
    print("  TAG LEGEND")
    print("=" * 70)
    for tag, desc in TAG_DESCRIPTIONS.items():
        print(f"  {tag:<18} {desc}")

    # ── Roofline ──
    if args.roofline:
        print(f"\nGenerating roofline plot: {args.roofline}")
        from viz.roofline import plot
        plot(profiles, gpu=gpu, output_path=args.roofline)

    # ── JSON output ──
    if args.output:
        output_data = {
            "gpu": {
                "name": gpu.name if gpu else "unknown",
                "fp32_tflops": gpu.fp32_tflops if gpu else None,
                "hbm_bandwidth_gbs": gpu.hbm_bandwidth_gbs if gpu else None,
            },
            "profiles": [
                {
                    "name": p.name,
                    "sm_util": p.sm_util,
                    "dram_read_mb": p.dram_read_mb,
                    "dram_write_mb": p.dram_write_mb,
                    "total_dram_mb": p.total_dram_mb,
                    "l1_util": p.l1_util,
                    "warp_active": p.warp_active,
                    "barrier_stall": p.barrier_stall,
                    "scoreboard_stall": p.scoreboard_stall,
                    "estimated_flops": p.estimated_flops,
                    "arithmetic_intensity": p.arithmetic_intensity,
                    "tag": p.tag,
                }
                for p in profiles
            ],
            "summary": summary,
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
