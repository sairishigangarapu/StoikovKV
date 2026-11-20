"""
Generates a publication-grade IEEE architecture diagram (`architecture_diagram.png`)
with clean non-crossing orthogonal routing and proper mathematical notation.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def draw_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    body: str,
    facecolor: str,
    edgecolor: str,
    title_color: str = "#0F172A",
    body_color: str = "#1E293B",
    title_size: float = 8.4,
    body_size: float = 7.1,
    lw: float = 1.4,
) -> None:
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.010,rounding_size=0.016",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=lw,
        zorder=3,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2.0,
        y + h * 0.74,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color=title_color,
        zorder=4,
    )
    ax.text(
        x + w / 2.0,
        y + h * 0.34,
        body,
        ha="center",
        va="center",
        fontsize=body_size,
        color=body_color,
        zorder=4,
        linespacing=1.28,
    )


def draw_arrow(
    ax: plt.Axes,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    color: str = "#334155",
    lw: float = 1.35,
    ls: str = "-",
) -> None:
    arr = FancyArrowPatch(
        (x0, y0),
        (x1, y1),
        arrowstyle="-|>",
        mutation_scale=10.5,
        linewidth=lw,
        linestyle=ls,
        color=color,
        zorder=5,
    )
    ax.add_patch(arr)


def draw_poly_arrow(
    ax: plt.Axes,
    pts: list[tuple[float, float]],
    color: str = "#334155",
    lw: float = 1.25,
    ls: str = "--",
) -> None:
    xs = [p[0] for p in pts[:-1]]
    ys = [p[1] for p in pts[:-1]]
    ax.plot(xs, ys, color=color, linewidth=lw, linestyle=ls, zorder=2.5)
    # Final segment with arrowhead
    draw_arrow(ax, pts[-2][0], pts[-2][1], pts[-1][0], pts[-1][1], color=color, lw=lw, ls=ls)


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    out_paths = [
        root / "benchmarks" / "results" / "architecture_diagram.png",
        root / "conference-latex-template_10-17-19" / "Conference-LaTeX-template_10-17-19" / "architecture_diagram.png",
    ]

    fig, ax = plt.subplots(figsize=(11.6, 5.2), dpi=300)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    # Zone 1: Offline Calibration
    ax.add_patch(
        FancyBboxPatch(
            (0.012, 0.11),
            0.228,
            0.84,
            boxstyle="round,pad=0.008,rounding_size=0.018",
            facecolor="#F8FAFC",
            edgecolor="#94A3B8",
            linewidth=1.1,
            linestyle="--",
            zorder=1,
        )
    )
    ax.text(
        0.126,
        0.925,
        "OFFLINE CALIBRATION (One-Time)",
        ha="center",
        va="center",
        fontsize=8.0,
        fontweight="bold",
        color="#475569",
        zorder=2,
    )

    # Zone 2: Online Inference
    ax.add_patch(
        FancyBboxPatch(
            (0.265, 0.11),
            0.723,
            0.84,
            boxstyle="round,pad=0.008,rounding_size=0.018",
            facecolor="#F0F9FF",
            edgecolor="#0284C7",
            linewidth=1.2,
            zorder=1,
        )
    )
    ax.text(
        0.626,
        0.925,
        "ONLINE INFERENCE   (100% FlashAttention-Compatible: output_attentions = False)",
        ha="center",
        va="center",
        fontsize=8.2,
        fontweight="bold",
        color="#0369A1",
        zorder=2,
    )

    # --- OFFLINE BOXES ---
    draw_box(
        ax, 0.024, 0.71, 0.204, 0.175,
        "1. Streaming Hook Profiler",
        "Hooks q/k/v_proj (64-q chunks)\nComputes entropy $H_h$ and\nneedle selectivity score $R_h$",
        facecolor="#FFFFFF", edgecolor="#64748B",
    )
    draw_box(
        ax, 0.024, 0.50, 0.204, 0.17,
        "2. Query Subspace $U_Q$",
        "Top-$r$ eigvecs of $Q^\\top Q + 0.1I$\nSign-aligned: $\\bar{Q}U_Q \\geq 0$\nPer-(layer, KV-head) $[d \\times r]$",
        facecolor="#EFF6FF", edgecolor="#2563EB",
    )
    draw_box(
        ax, 0.024, 0.295, 0.204, 0.165,
        "3. Dictionary $D$ & $W_{\\mathrm{ridge}}$",
        "Layer-shared $D$ $[d \\times m]$ ($m=128$)\n$W_{\\mathrm{ridge}}=(D^\\top D+\\lambda_2 I)^{-1}D^\\top$",
        facecolor="#F5F3FF", edgecolor="#7C3AED",
    )
    draw_box(
        ax, 0.024, 0.135, 0.204, 0.12,
        "4. Head Role Map",
        "Tier 0 (FP16) | Tier 1 | Tier 2",
        facecolor="#FEF3C7", edgecolor="#D97706",
    )

    draw_arrow(ax, 0.126, 0.71, 0.126, 0.675, color="#64748B")
    draw_arrow(ax, 0.126, 0.50, 0.126, 0.465, color="#64748B")
    draw_arrow(ax, 0.126, 0.295, 0.126, 0.260, color="#64748B")

    # --- LIVE COLUMN 1: INPUT, PROBE, CONTROLLER ---
    draw_box(
        ax, 0.282, 0.70, 0.206, 0.185,
        "Incoming KV Stream ($K_t, V_t$)",
        "Recent Local FP16 Window\n(last $n_b = 32$ tokens kept\nuncompressed across all heads)",
        facecolor="#FFFFFF", edgecolor="#0284C7",
    )
    draw_box(
        ax, 0.282, 0.43, 0.206, 0.22,
        "Write-Time Directional Probe",
        "$p_t = K_t U_Q$  ($O(dr)$, no $T\\times T$)\n$I_t = \\mathrm{sp}(p_{t,0})(1+0.25\\|\\mathrm{ReLU}(p_t)\\|)$\n$\\times\\sqrt{\\|V_t\\|_2} + B_{\\mathrm{sink}} + \\mathrm{Rarity}$",
        facecolor="#DBEAFE", edgecolor="#1D4ED8",
    )
    draw_box(
        ax, 0.282, 0.14, 0.206, 0.23,
        "Online Regime-Change\nDecay Controller",
        "EMA centroid on $V_t[:32]$\nStationary: $\\beta_t = 0.99$\nShift ($z > 1.4$): $\\beta_t = 0.65$",
        facecolor="#FFFBEB", edgecolor="#D97706",
    )

    draw_arrow(ax, 0.385, 0.70, 0.385, 0.655, color="#1D4ED8")
    draw_arrow(ax, 0.385, 0.43, 0.385, 0.375, color="#D97706")

    # --- LIVE COLUMN 2: 3-TIER HETEROGENEOUS ROUTER ---
    draw_box(
        ax, 0.532, 0.70, 0.208, 0.185,
        "Tier 0: Retrieval Heads",
        "Exact Uncompressed FP16\nProtected by selectivity $R_h$\n(0% retrieval distortion)",
        facecolor="#ECFDF5", edgecolor="#059669",
    )
    draw_box(
        ax, 0.532, 0.43, 0.208, 0.22,
        "Tier 1: Peaked Heads (H2O)",
        "Heavy-Hitter Eviction via $V_i(t)$\n$V_i(t)=\\beta_t V_i(t-1)+I_i$\nShields sinks & rarity anchors",
        facecolor="#FEF3C7", edgecolor="#D97706",
    )
    draw_box(
        ax, 0.532, 0.14, 0.208, 0.23,
        "Tier 2: Diffuse Heads (Sparse)",
        "1-Step Ridge Sparse Coding\n$Z=X W_{\\mathrm{ridge}}^\\top \\to$ soft-thresh $\\to$\ntop-$s$ sorted $|y|$ ($3s+2$ B/vec)",
        facecolor="#F3E8FF", edgecolor="#7C3AED",
    )

    # Clean non-crossing channel at x = 0.508 between Column 1 and Column 2
    draw_arrow(ax, 0.488, 0.792, 0.530, 0.792, color="#059669")
    # Overflow vertical bus from Local Window down to Tier 1 & Tier 2
    ax.plot([0.506, 0.506], [0.792, 0.255], color="#0284C7", lw=1.2, zorder=2.5)
    draw_arrow(ax, 0.506, 0.575, 0.530, 0.575, color="#0284C7", lw=1.2)
    draw_arrow(ax, 0.506, 0.255, 0.530, 0.255, color="#7C3AED", lw=1.2)
    # Probe -> Tier 1
    draw_arrow(ax, 0.488, 0.525, 0.530, 0.525, color="#1D4ED8")
    # Controller -> Tier 1 via x = 0.516
    draw_poly_arrow(ax, [(0.488, 0.30), (0.516, 0.30), (0.516, 0.47), (0.530, 0.47)], color="#D97706", ls="-")

    # --- LIVE COLUMN 3: MATRYOSHKA SELF-SPECULATIVE DECODER ---
    draw_box(
        ax, 0.778, 0.555, 0.196, 0.33,
        "Matryoshka Draft Phase",
        "Zero auxiliary model weights\nTier 2 reads only top-$m$\nprefix atoms ($m \\ll s$)\nDrafts $\\gamma=4$ tokens with\nreduced KV read bandwidth",
        facecolor="#E0F2FE", edgecolor="#0284C7",
    )
    draw_box(
        ax, 0.778, 0.14, 0.196, 0.335,
        "Lossless Verify & Rollback",
        "1 batched forward at full\n$s=16$ atoms (exact logits)\nStrict Leviathan rejection\nsampling / greedy parity\nRing-buffer exact crop()",
        facecolor="#FCE7F3", edgecolor="#BE185D",
    )

    # Clean vertical bus at x = 0.758 from 3 Tiers into Draft & Verify
    ax.plot([0.740, 0.758], [0.792, 0.792], color="#059669", lw=1.25, zorder=2.5)
    ax.plot([0.740, 0.758], [0.540, 0.540], color="#D97706", lw=1.25, zorder=2.5)
    ax.plot([0.740, 0.758], [0.255, 0.255], color="#7C3AED", lw=1.25, zorder=2.5)
    ax.plot([0.758, 0.758], [0.255, 0.792], color="#334155", lw=1.25, zorder=2.5)
    draw_arrow(ax, 0.758, 0.720, 0.776, 0.720, color="#0284C7", lw=1.35)
    draw_arrow(ax, 0.758, 0.310, 0.776, 0.310, color="#BE185D", lw=1.35)

    # Draft <-> Verify arrows
    draw_arrow(ax, 0.892, 0.555, 0.892, 0.478, color="#BE185D", lw=1.55)
    draw_arrow(ax, 0.840, 0.475, 0.840, 0.552, color="#0284C7", lw=1.2, ls=":")
    ax.text(0.936, 0.515, "verify &\naccept", ha="center", va="center", fontsize=6.7, color="#9D174D", fontweight="bold")
    ax.text(0.805, 0.515, "rollback", ha="center", va="center", fontsize=6.5, color="#0369A1")

    # --- CLEAN BOTTOM CORRIDOR FOR OFFLINE ARTIFACTS (y = 0.02 .. 0.085, ZERO TEXT CROSSING!) ---
    # 1. U_Q -> Write-Time Probe (direct short dashed arrow)
    draw_arrow(ax, 0.228, 0.585, 0.280, 0.585, color="#2563EB", ls="--", lw=1.25)
    # 2. D & W_ridge -> bottom corridor -> Tier 2
    draw_poly_arrow(
        ax,
        [(0.228, 0.35), (0.250, 0.35), (0.250, 0.068), (0.610, 0.068), (0.610, 0.138)],
        color="#7C3AED",
        ls="--",
        lw=1.2,
    )
    ax.text(0.43, 0.080, "Calibrated $D, W_{\\mathrm{ridge}}$", ha="center", va="center", fontsize=6.8, color="#6D28D9")
    # 3. Head Role Map -> bottom corridor -> 3-Tier Router
    draw_poly_arrow(
        ax,
        [(0.126, 0.135), (0.126, 0.028), (0.662, 0.028), (0.662, 0.138)],
        color="#D97706",
        ls="--",
        lw=1.2,
    )
    ax.text(0.43, 0.041, "Offline Per-KV-Head Role Assignment Map (Tier 0 / Tier 1 / Tier 2)", ha="center", va="center", fontsize=6.8, color="#B45309")

    plt.tight_layout(pad=0.2)
    for p in out_paths:
        fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
