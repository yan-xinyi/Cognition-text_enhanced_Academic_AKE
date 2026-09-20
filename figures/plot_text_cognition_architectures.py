#!/usr/bin/env python
"""Publication schematic comparing the frozen text-cognition AKE structures."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "text_cognition_architecture_source_data.csv"
OUT = HERE / "figure_text_cognition_architectures"

EXPECTED_METHODS = ["TEXT", "DC", "GIB", "GAZESUP", "CIB", "CA", "FS-CA", "ET-ATT"]

COLORS = {
    "text_fill": "#DCEAF7",
    "text_edge": "#2B6EA6",
    "cog_fill": "#FBE8D0",
    "cog_edge": "#CC7722",
    "joint_fill": "#E9E1F2",
    "joint_edge": "#76558F",
    "task_fill": "#DCEFE5",
    "task_edge": "#3C7A57",
    "aux_fill": "#F2F3F5",
    "aux_edge": "#6B7280",
    "ink": "#1F2937",
    "muted": "#667085",
    "group_1": "#F5F8FC",
    "group_2": "#FFF9F1",
    "group_3": "#F8F5FB",
}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 6.4,
        "axes.linewidth": 0.8,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def node(ax, x, y, w, h, text, kind="text", fontsize=6.2, lw=0.85, bold=False):
    fill = COLORS[f"{kind}_fill"]
    edge = COLORS[f"{kind}_edge"]
    patch = FancyBboxPatch(
        (x, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.004,rounding_size=0.008",
        facecolor=fill,
        edgecolor=edge,
        linewidth=lw,
        zorder=3,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold" if bold else "normal",
        color=COLORS["ink"],
        linespacing=1.05,
        zorder=4,
    )
    return patch


def arrow(ax, x1, y1, x2, y2, color="#667085", dashed=False, lw=0.85, rad=0.0):
    patch = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=7.5,
        linewidth=lw,
        linestyle=(0, (3, 2)) if dashed else "solid",
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=1.5,
        shrinkB=1.5,
        zorder=2,
    )
    ax.add_patch(patch)
    return patch


def method_label(ax, y, method, subtitle, accent=False):
    ax.text(
        0.176,
        y + 0.010,
        method,
        ha="left",
        va="center",
        fontsize=7.4,
        fontweight="bold",
        color=COLORS["joint_edge"] if accent else COLORS["ink"],
    )
    # The mechanism itself is encoded by the nodes and arrows; keeping only
    # the method name here prevents redundant prose from obscuring the paths.


def group_band(ax, y0, y1, label, fill, number):
    ax.add_patch(Rectangle((0.008, y0), 0.984, y1 - y0, facecolor=fill, edgecolor="#D4D9E2", lw=0.7, zorder=0))
    ax.add_patch(Rectangle((0.008, y0), 0.145, y1 - y0, facecolor=fill, edgecolor="#BAC3D1", lw=0.75, zorder=1))
    ax.text(0.027, y1 - 0.022, str(number), fontsize=8.5, fontweight="bold", color="#24476B", va="top")
    ax.text(0.027, (y0 + y1) / 2, label, fontsize=7.1, fontweight="bold", color="#24476B", va="center", ha="left", linespacing=1.18)


def draw_common_legend(ax):
    items = [
        ("text", "Text representation"),
        ("cog", "Proxy ET / cognitive path"),
        ("joint", "Joint representation"),
        ("task", "AKE decoder"),
    ]
    x_positions = [0.285, 0.440, 0.635, 0.785]
    for (kind, label), x in zip(items, x_positions):
        ax.add_patch(Rectangle((x, 0.920), 0.014, 0.014, facecolor=COLORS[f"{kind}_fill"], edgecolor=COLORS[f"{kind}_edge"], lw=0.8))
        ax.text(x + 0.019, 0.927, label, fontsize=5.55, va="center", color=COLORS["ink"])
    ax.plot([0.900, 0.919], [0.927, 0.927], color=COLORS["aux_edge"], lw=0.9, ls=(0, (3, 2)))
    ax.text(0.925, 0.927, "Training only", fontsize=5.55, va="center", color=COLORS["ink"])


def build_figure():
    with SOURCE.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    methods = [row["method"] for row in rows]
    if methods != EXPECTED_METHODS:
        raise RuntimeError(f"Unexpected method order: {methods}")

    fig, ax = plt.subplots(figsize=(7.2, 8.45))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.008, 0.982, "Comparison of text-cognition integration strategies", fontsize=10.3, fontweight="bold", ha="left", va="top", color="#17365D")
    ax.text(0.008, 0.953, "One text-only reference and seven cognitive integration structures under a matched SciBERT-BiLSTM-CRF benchmark", fontsize=6.25, ha="left", va="top", color=COLORS["muted"])
    draw_common_legend(ax)

    group_band(ax, 0.700, 0.900, "No cognition\nand direct fusion", COLORS["group_1"], 1)
    group_band(ax, 0.391, 0.692, "Cognitive\ntransformation\nand supervision", COLORS["group_2"], 2)
    group_band(ax, 0.060, 0.383, "Task-aware\norganisation", COLORS["group_3"], 3)

    # TEXT
    y = 0.846
    method_label(ax, y, "TEXT", "Text-only reference")
    node(ax, 0.300, y, 0.130, 0.047, "SciBERT token\nembeddings", "text")
    node(ax, 0.475, y, 0.125, 0.047, "BiLSTM text\nstates", "text")
    node(ax, 0.650, y, 0.115, 0.047, "Linear BIO\nemissions", "task")
    node(ax, 0.830, y, 0.090, 0.047, "CRF", "task", bold=True)
    arrow(ax, 0.430, y, 0.475, y)
    arrow(ax, 0.600, y, 0.650, y)
    arrow(ax, 0.765, y, 0.830, y)

    # DC
    y = 0.746
    method_label(ax, y, "DC", "Projected\nlate fusion")
    node(ax, 0.286, y + 0.020, 0.110, 0.038, "SciBERT", "text")
    node(ax, 0.423, y + 0.020, 0.110, 0.038, "BiLSTM", "text")
    node(ax, 0.286, y - 0.027, 0.110, 0.038, "ET5 + masks", "cog")
    node(ax, 0.423, y - 0.027, 0.110, 0.038, "MLP projection", "cog")
    node(ax, 0.590, y, 0.105, 0.045, "Concatenate", "joint", bold=True)
    node(ax, 0.735, y, 0.100, 0.045, "Linear", "task")
    node(ax, 0.870, y, 0.075, 0.045, "CRF", "task", bold=True)
    arrow(ax, 0.396, y + 0.020, 0.423, y + 0.020)
    arrow(ax, 0.396, y - 0.027, 0.423, y - 0.027, color=COLORS["cog_edge"])
    arrow(ax, 0.533, y + 0.020, 0.590, y + 0.006)
    arrow(ax, 0.533, y - 0.027, 0.590, y - 0.006, color=COLORS["cog_edge"])
    arrow(ax, 0.695, y, 0.735, y)
    arrow(ax, 0.835, y, 0.870, y)

    # GIB
    y = 0.636
    method_label(ax, y, "GIB", "Entropy-weighted\nresidual injection")
    node(ax, 0.275, y + 0.020, 0.105, 0.038, "SciBERT", "text")
    node(ax, 0.410, y + 0.020, 0.100, 0.038, "BiLSTM", "text")
    node(ax, 0.275, y - 0.027, 0.105, 0.038, "ET5", "cog")
    node(ax, 0.410, y - 0.027, 0.120, 0.038, "5 feature\nembeddings", "cog", fontsize=5.8)
    node(ax, 0.555, y - 0.027, 0.110, 0.038, "Entropy\nweights", "cog", fontsize=5.8)
    node(ax, 0.690, y, 0.105, 0.045, "Residual add\n+ LayerNorm", "joint", fontsize=5.8, bold=True)
    node(ax, 0.825, y, 0.100, 0.045, "Linear + CRF", "task")
    arrow(ax, 0.380, y + 0.020, 0.410, y + 0.020)
    arrow(ax, 0.380, y - 0.027, 0.410, y - 0.027, color=COLORS["cog_edge"])
    arrow(ax, 0.530, y - 0.027, 0.555, y - 0.027, color=COLORS["cog_edge"])
    arrow(ax, 0.510, y + 0.020, 0.690, y + 0.008)
    arrow(ax, 0.665, y - 0.027, 0.690, y - 0.008, color=COLORS["cog_edge"])
    arrow(ax, 0.795, y, 0.825, y)

    # GAZESUP
    y = 0.538
    method_label(ax, y, "GAZESUP", "Auxiliary gaze\nprediction")
    node(ax, 0.285, y + 0.014, 0.105, 0.040, "SciBERT", "text")
    node(ax, 0.425, y + 0.014, 0.105, 0.040, "BiLSTM", "text")
    node(ax, 0.585, y + 0.014, 0.105, 0.040, "Linear", "task")
    node(ax, 0.735, y + 0.014, 0.085, 0.040, "CRF", "task", bold=True)
    node(ax, 0.555, y - 0.035, 0.115, 0.032, "5D gaze head", "aux", fontsize=5.8)
    node(ax, 0.725, y - 0.035, 0.140, 0.032, "Masked MSE vs ET5", "aux", fontsize=5.8)
    arrow(ax, 0.390, y + 0.014, 0.425, y + 0.014)
    arrow(ax, 0.530, y + 0.014, 0.585, y + 0.014)
    arrow(ax, 0.690, y + 0.014, 0.735, y + 0.014)
    arrow(ax, 0.477, y - 0.006, 0.555, y - 0.035, dashed=True, color=COLORS["aux_edge"])
    arrow(ax, 0.670, y - 0.035, 0.725, y - 0.035, dashed=True, color=COLORS["aux_edge"])
    ax.text(0.875, y - 0.035, "training only", fontsize=5.5, color=COLORS["muted"], va="center")

    # CIB
    y = 0.438
    method_label(ax, y, "CIB", "Variational\ncompression")
    node(ax, 0.275, y + 0.020, 0.105, 0.038, "SciBERT", "text")
    node(ax, 0.275, y - 0.027, 0.105, 0.038, "ET5", "cog")
    node(ax, 0.410, y - 0.027, 0.115, 0.038, "Variational\nencoder", "cog", fontsize=5.8)
    node(ax, 0.555, y - 0.027, 0.085, 0.038, "Latent z\n+ KL", "cog", fontsize=5.7)
    node(ax, 0.555, y + 0.020, 0.100, 0.038, "Concatenate", "joint")
    node(ax, 0.690, y + 0.020, 0.095, 0.038, "BiLSTM", "joint")
    node(ax, 0.825, y + 0.020, 0.105, 0.038, "Linear + CRF", "task")
    arrow(ax, 0.380, y - 0.027, 0.410, y - 0.027, color=COLORS["cog_edge"])
    arrow(ax, 0.525, y - 0.027, 0.555, y - 0.027, color=COLORS["cog_edge"])
    arrow(ax, 0.380, y + 0.020, 0.555, y + 0.020)
    arrow(ax, 0.598, y - 0.008, 0.598, y + 0.001, color=COLORS["cog_edge"])
    arrow(ax, 0.655, y + 0.020, 0.690, y + 0.020)
    arrow(ax, 0.785, y + 0.020, 0.825, y + 0.020)

    # CA
    y = 0.328
    method_label(ax, y, "CA", "Shared/private\nalignment")
    node(ax, 0.270, y + 0.026, 0.100, 0.036, "SciBERT", "text")
    node(ax, 0.405, y + 0.026, 0.105, 0.036, "Text\nbackbone", "text", fontsize=5.8)
    node(ax, 0.270, y - 0.028, 0.100, 0.036, "ET5 + masks", "cog")
    node(ax, 0.405, y - 0.028, 0.120, 0.036, "Text-conditioned\n5D weights", "cog", fontsize=5.6)
    node(ax, 0.560, y + 0.026, 0.125, 0.036, "Shared + private\ntext encoders", "joint", fontsize=5.6)
    node(ax, 0.560, y - 0.028, 0.125, 0.036, "Shared + private\ncognitive encoders", "joint", fontsize=5.5)
    node(ax, 0.725, y + 0.026, 0.100, 0.036, "Text CRF", "task", fontsize=5.8, bold=True)
    node(ax, 0.725, y - 0.028, 0.100, 0.036, "Cognitive CRF", "aux", fontsize=5.6)
    node(ax, 0.855, y, 0.115, 0.040, "GRL modality\ndiscriminator", "aux", fontsize=5.5)
    arrow(ax, 0.370, y + 0.026, 0.405, y + 0.026)
    arrow(ax, 0.370, y - 0.028, 0.405, y - 0.028, color=COLORS["cog_edge"])
    arrow(ax, 0.510, y + 0.026, 0.560, y + 0.026)
    arrow(ax, 0.525, y - 0.028, 0.560, y - 0.028, color=COLORS["cog_edge"])
    arrow(ax, 0.685, y + 0.026, 0.725, y + 0.026)
    arrow(ax, 0.685, y - 0.028, 0.725, y - 0.028, dashed=True, color=COLORS["aux_edge"])
    arrow(ax, 0.670, y + 0.012, 0.855, y + 0.006, dashed=True, color=COLORS["aux_edge"], rad=-0.10)
    arrow(ax, 0.670, y - 0.014, 0.855, y - 0.006, dashed=True, color=COLORS["aux_edge"], rad=0.10)

    # FS-CA
    y = 0.218
    method_label(ax, y, "FS-CA", "Feature gate\nbefore CA")
    node(ax, 0.275, y + 0.020, 0.105, 0.038, "SciBERT", "text")
    node(ax, 0.275, y - 0.027, 0.105, 0.038, "ET5", "cog")
    node(ax, 0.410, y - 0.027, 0.115, 0.038, "Global 5D\nfeature gate", "cog", fontsize=5.8, bold=True)
    node(ax, 0.565, y, 0.155, 0.047, "CogAlign shared/private\nalignment core", "joint", fontsize=5.8)
    node(ax, 0.770, y, 0.110, 0.047, "Text CRF", "task", bold=True)
    arrow(ax, 0.380, y - 0.027, 0.410, y - 0.027, color=COLORS["cog_edge"])
    arrow(ax, 0.380, y + 0.020, 0.565, y + 0.010)
    arrow(ax, 0.525, y - 0.027, 0.565, y - 0.010, color=COLORS["cog_edge"])
    arrow(ax, 0.720, y, 0.770, y)

    # ET-ATT: highlighted hero row.
    y = 0.106
    ax.add_patch(FancyBboxPatch((0.163, 0.068), 0.820, 0.078, boxstyle="round,pad=0.005,rounding_size=0.008", facecolor="none", edgecolor=COLORS["joint_edge"], lw=1.25, zorder=1))
    method_label(ax, y, "ET-ATT", "Early fusion;\ntoken attention", accent=True)
    node(ax, 0.272, y, 0.120, 0.048, "SciBERT +\nmasked ET5", "joint", fontsize=5.8, bold=True)
    node(ax, 0.418, y, 0.120, 0.048, "BiLSTM fused\ntoken states h_i", "joint", fontsize=5.7)
    node(ax, 0.564, y, 0.120, 0.048, "Scalar score +\nsequence softmax a_i", "joint", fontsize=5.5)
    node(ax, 0.710, y, 0.115, 0.048, "Global context\ng = sum a_i h_i", "joint", fontsize=5.7)
    node(ax, 0.848, y, 0.120, 0.048, "Local h_i + global g\n-> Linear + CRF", "task", fontsize=5.20, bold=True)
    arrow(ax, 0.392, y, 0.418, y, color=COLORS["joint_edge"], lw=1.0)
    arrow(ax, 0.538, y, 0.564, y, color=COLORS["joint_edge"], lw=1.0)
    arrow(ax, 0.684, y, 0.710, y, color=COLORS["joint_edge"], lw=1.0)
    arrow(ax, 0.825, y, 0.848, y, color=COLORS["joint_edge"], lw=1.0)
    ax.text(0.010, 0.043, "ET-ATT attention is learned indirectly from the AKE loss; no gaze-attention target is imposed.", fontsize=5.30, color=COLORS["muted"], ha="left")
    ax.text(0.010, 0.029, "ET5 = NFIX, FFD, GPT, TRT and FIXPROP. Solid arrows: task forward path; dashed arrows: auxiliary training objectives.", fontsize=5.35, color=COLORS["muted"], ha="left")
    ax.text(0.010, 0.015, "Proxy ET is used at inference by DC, GIB, CIB and ET-ATT; GAZESUP, CA and FS-CA use cognition during training only.", fontsize=5.25, color=COLORS["muted"], ha="left")

    fig.savefig(OUT.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


if __name__ == "__main__":
    build_figure()
    print(f"WROTE {OUT}.[svg|pdf|png|tiff]")
