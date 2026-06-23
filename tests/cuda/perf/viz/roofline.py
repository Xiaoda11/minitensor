"""Roofline visualisation — scatter kernels on FLOP/byte vs FLOP/s plot."""

from typing import Optional

from ..core.types import KernelProfile, GpuSpec


def plot(profiles: list[KernelProfile],
         gpu: Optional[GpuSpec] = None,
         output_path: Optional[str] = None,
         title: str = "Roofline Model"):
    """Generate a roofline plot: arithmetic intensity vs performance.

    X-axis: Arithmetic Intensity (FLOP/byte) — log scale
    Y-axis: Attainable GFLOPS — log scale

    The memory bandwidth ceiling (diagonal line) and compute ceiling
    (horizontal line) are drawn from gpu specs. Kernels are scattered
    as points, colored by their bottleneck tag.

    Args:
        profiles: List of KernelProfiles to plot.
        gpu: GPU spec for ceiling lines. Falls back to RTX 3060 defaults.
        output_path: If set, save to file instead of showing interactively.
        title: Plot title.
    """
    try:
        import matplotlib
        # Prefer Agg on headless (no display), only try TkAgg if showing interactively
        backend = "Agg"
        if output_path is None:
            try:
                import os
                if os.environ.get("DISPLAY"):
                    backend = "TkAgg"
            except Exception:
                pass
        matplotlib.use(backend)
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib not installed, skipping roofline plot")
        return

    if gpu is None:
        gpu = GpuSpec(name="unknown", fp32_tflops=12.7, hbm_bandwidth_gbs=360, sm_count=28)

    # Filter profiles with valid AI and GFLOPS
    valid = [p for p in profiles
             if p.arithmetic_intensity > 0 and p.estimated_flops > 0]

    if not valid:
        print("[warn] No profiles with valid arithmetic intensity, skipping plot")
        return

    # Convert TFLOPS → GFLOPS, GB/s stays as-is
    # GFLOPS achievable: from estimated_flops we'd need latency. We use SM util × peak
    # as a proxy for "attainable GFLOPS".
    peak_gflops = gpu.fp32_tflops * 1000.0

    ai_values = []
    perf_values = []
    names = []
    colors = []
    tag_color_map = {
        "compute_bound": "#e74c3c",
        "memory_bound":  "#3498db",
        "latency_bound": "#f39c12",
        "sync_bound":    "#9b59b6",
        "balanced":      "#2ecc71",
    }

    for p in valid:
        ai_values.append(p.arithmetic_intensity)
        # Attainable GFLOPS ≈ SM util % of peak
        attainable = (p.sm_util / 100.0) * peak_gflops if p.sm_util > 0 else 0.0
        perf_values.append(attainable)
        names.append(p.name.removesuffix("_kernel") if p.name.endswith("_kernel") else p.name)
        base_tag = p.tag.split(":")[0].strip()
        colors.append(tag_color_map.get(base_tag, "#95a5a6"))

    fig, ax = plt.subplots(figsize=(10, 7))

    # ── Roofline ceiling lines ──
    # Ridge point: TFLOPS × 1000 / GB/s = FLOP/byte
    #   (TFLOPS=10^12 FLOP/s, GB/s=10^9 bytes/s → factor of 10^3)
    ridge = gpu.fp32_tflops * 1000.0 / gpu.hbm_bandwidth_gbs
    ai_range = [min(ai_values) * 0.5, max(ai_values) * 2.0]

    # Memory bandwidth ceiling: GFLOPS = bandwidth × AI
    ai_line = [ai_range[0], ridge, ai_range[1]]
    bw_line = [
        gpu.hbm_bandwidth_gbs * ai_range[0],
        gpu.hbm_bandwidth_gbs * ridge,
        gpu.hbm_bandwidth_gbs * ridge,  # capped at compute ceiling
    ]
    ax.plot(ai_line[:2], bw_line[:2], "b--", linewidth=2,
            label=f"Memory BW ceiling ({gpu.hbm_bandwidth_gbs:.0f} GB/s)")

    # Compute ceiling: flat line at peak GFLOPS
    ax.axhline(y=peak_gflops, color="r", linestyle="--", linewidth=2,
               label=f"Compute ceiling ({gpu.fp32_tflops:.1f} TFLOPS)")

    # Ridge point marker
    ax.axvline(x=ridge, color="gray", linestyle=":", alpha=0.5)
    ax.annotate(f"Ridge={ridge:.1f}", xy=(ridge, peak_gflops * 0.1),
                fontsize=8, color="gray", ha="center")

    # ── Scatter points ──
    ax.scatter(ai_values, perf_values, c=colors, s=80, edgecolors="black",
               linewidth=0.5, zorder=5)

    # Label each point
    for i, name in enumerate(names):
        ax.annotate(name, (ai_values[i], perf_values[i]),
                    textcoords="offset points", xytext=(5, 5),
                    fontsize=7, alpha=0.8)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic Intensity (FLOP / byte)")
    ax.set_ylabel("Attainable Performance (GFLOPS)")
    ax.set_title(f"{title}\n{gpu.name}")
    ax.grid(True, alpha=0.3)

    # Single combined legend: ceiling lines + bottleneck colours
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="b", linestyle="--", linewidth=2,
               label=f"Memory BW ceiling ({gpu.hbm_bandwidth_gbs:.0f} GB/s)"),
        Line2D([0], [0], color="r", linestyle="--", linewidth=2,
               label=f"Compute ceiling ({gpu.fp32_tflops:.1f} TFLOPS)"),
    ] + [
        Patch(facecolor=c, edgecolor="black", label=t)
        for t, c in tag_color_map.items()
    ]
    ax.legend(handles=legend_elements, fontsize=7, loc="lower left",
              title="Legend")

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Roofline plot saved to {output_path}")
    else:
        plt.show()

    plt.close(fig)
