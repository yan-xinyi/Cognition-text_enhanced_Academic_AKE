#!/usr/bin/env python
"""Publication figure for Proxy-ET coverage expansion and AKE recovery."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "figures" / "figure2"
OUT = DATA
DOMAINS = ("PMC", "LIS", "IEEE")
SYSTEMS = ("TEXT", "CogAlign", "ET-ATT")
TRANSITIONS = (
    "Directly covered",
    "Proxy-completed\nfrom partial",
    "Proxy-completed\nfrom zero",
)
TRANSITION_KEYS = (
    "Directly covered",
    "Proxy-completed from partial",
    "Proxy-completed from zero",
)
PHRASE_TYPES = (
    "Single word\nterm",
    "Noun\ncompound",
    "Modifier-noun\nphrase",
    "Technical\n/ other",
)
PHRASE_KEYS = (
    "Single-word term",
    "Noun compound",
    "Modifier-noun phrase",
    "Technical-form / other",
)


COLORS = {
    "TEXT": "#6B7280",
    "CogAlign": "#2878B5",
    "ET-ATT": "#D97706",
    "Full": "#2A9D8F",
    "Partial": "#88C0D0",
    "Zero": "#D9DEE7",
}


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 6.2,
        "axes.titlesize": 7.2,
        "axes.labelsize": 6.5,
        "xtick.labelsize": 5.7,
        "ytick.labelsize": 5.7,
        "legend.fontsize": 5.8,
        "axes.linewidth": 0.65,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    }
)


def panel_label(ax, label: str) -> None:
    ax.text(-0.14, 1.08, label, transform=ax.transAxes, fontsize=8, fontweight="bold", va="top")


def draw_panel_a(ax: plt.Axes, source: pd.DataFrame) -> None:
    frame = source.copy()
    x = []
    labels = []
    domain_centres = []
    for domain_index, domain in enumerate(DOMAINS):
        start = domain_index * 2.8
        x.extend([start, start + 0.9])
        labels.extend(["Observed", "Proxy"])
        domain_centres.append(start + 0.45)

    bottoms = np.zeros(len(x), dtype=float)
    for level in ("Zero", "Partial", "Full"):
        values = []
        for domain in DOMAINS:
            for source_name in ("Observed ET", "Proxy ET"):
                row = frame[
                    (frame["domain"] == domain)
                    & (frame["coverage_source"] == source_name)
                    & (frame["coverage_level"] == level)
                ]
                values.append(float(row["percentage"].iloc[0]) if len(row) else 0.0)
        ax.bar(x, values, width=0.68, bottom=bottoms, color=COLORS[level], edgecolor="white", linewidth=0.45, label=level)
        bottoms += np.asarray(values)

    full_values = []
    for domain in DOMAINS:
        for source_name in ("Observed ET", "Proxy ET"):
            row = frame[
                (frame["domain"] == domain)
                & (frame["coverage_source"] == source_name)
                & (frame["coverage_level"] == "Full")
            ]
            full_values.append(float(row["percentage"].iloc[0]))
    for xpos, full in zip(x, full_values):
        ax.text(xpos, 102.0, f"{full:.1f}%", ha="center", va="bottom", fontsize=5.3, color="#1F2937")

    ax.set_ylim(0, 110)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Gold-keyphrase occurrences (%)")
    ax.set_xticks(x, labels)
    for centre, domain in zip(domain_centres, DOMAINS):
        ax.text(centre, -0.23, domain, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=6.3, fontweight="bold")
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(title="Complete phrase coverage", ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.24), title_fontsize=5.8)
    ax.set_title("Observed-to-Proxy coverage expansion", pad=22)
    panel_label(ax, "a")


def draw_panel_b(axes: list[plt.Axes], source: pd.DataFrame) -> None:
    offsets = {"TEXT": -0.16, "CogAlign": 0.0, "ET-ATT": 0.16}
    markers = {"TEXT": "o", "CogAlign": "s", "ET-ATT": "D"}
    x = np.arange(len(TRANSITION_KEYS), dtype=float)
    for domain_index, (ax, domain) in enumerate(zip(axes, DOMAINS)):
        frame = source[(source["split"] == "test") & (source["domain"] == domain)].copy()
        n_by_transition = frame.groupby("coverage_transition")["n_gold_occurrences"].first().to_dict()
        for index, transition in enumerate(TRANSITION_KEYS):
            if int(n_by_transition.get(transition, 0)) < 20:
                ax.axvspan(index - 0.42, index + 0.42, color="#F3F4F6", zorder=0)
        for system in SYSTEMS:
            values = []
            lower = []
            upper = []
            for transition in TRANSITION_KEYS:
                row = frame[(frame["coverage_transition"] == transition) & (frame["system"] == system)]
                values.append(float(row["recovery_rate_percent"].iloc[0]))
                lower.append(float(row["recovery_ci_lower_percent"].iloc[0]))
                upper.append(float(row["recovery_ci_upper_percent"].iloc[0]))
            values_array = np.asarray(values)
            error = np.vstack([values_array - np.asarray(lower), np.asarray(upper) - values_array])
            ax.errorbar(
                x + offsets[system], values_array, yerr=error, fmt=markers[system],
                ms=3.2, capsize=1.8, capthick=0.65, elinewidth=0.7,
                color=COLORS[system], markeredgecolor="white", markeredgewidth=0.4,
                label=system, zorder=3,
            )
        ax.set_title(domain, pad=4, fontweight="bold")
        ax.set_ylim(0, 66)
        ax.set_yticks([0, 20, 40, 60])
        ax.set_xticks(x, TRANSITIONS)
        ax.grid(axis="y", color="#E5E7EB", linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)
        if domain_index == 0:
            ax.set_ylabel("Gold-keyphrase recovery rate (%)")
        else:
            ax.set_yticklabels([])
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y", length=0)
        for index, transition in enumerate(TRANSITION_KEYS):
            n = int(n_by_transition.get(transition, 0))
            color = "#9CA3AF" if n < 20 else "#4B5563"
            ax.text(index, -0.22, f"n={n}", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=5.1, color=color)
    axes[0].legend(ncol=3, loc="upper left", bbox_to_anchor=(0.0, 1.31))
    axes[1].text(0.5, 1.28, "Recovery by coverage transition", transform=axes[1].transAxes, ha="center", va="bottom", fontsize=7.2, fontweight="bold")
    panel_label(axes[0], "b")


def draw_panel_c(axes: list[plt.Axes], source: pd.DataFrame) -> None:
    cmap = LinearSegmentedColormap.from_list("delta", ["#B45353", "#F7F7F5", "#2A9D8F"])
    norm = TwoSlopeNorm(vmin=-16.0, vcenter=0.0, vmax=16.0)
    image = None
    for domain_index, (ax, domain) in enumerate(zip(axes, DOMAINS)):
        frame = source[
            (source["split"] == "test")
            & (source["domain"] == domain)
            & (source["system"].isin(("CogAlign", "ET-ATT")))
        ].copy()
        matrix = np.full((2, len(PHRASE_KEYS)), np.nan)
        counts = np.zeros((2, len(PHRASE_KEYS)), dtype=int)
        for row_index, system in enumerate(("CogAlign", "ET-ATT")):
            for col_index, phrase_type in enumerate(PHRASE_KEYS):
                row = frame[(frame["system"] == system) & (frame["phrase_type"] == phrase_type)]
                if len(row):
                    matrix[row_index, col_index] = float(row["delta_vs_text_pp"].iloc[0])
                    counts[row_index, col_index] = int(row["n_gold_occurrences"].iloc[0])
        image = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
        for row_index in range(2):
            for col_index in range(len(PHRASE_KEYS)):
                value = matrix[row_index, col_index]
                if not np.isfinite(value):
                    continue
                color = "white" if abs(value) >= 8.0 else "#1F2937"
                ax.text(col_index, row_index - 0.08, f"{value:+.1f}", ha="center", va="center", fontsize=5.8, fontweight="bold", color=color)
                ax.text(col_index, row_index + 0.20, f"n={counts[row_index, col_index]}", ha="center", va="center", fontsize=5.1, color=color)
        ax.set_title(domain, pad=4, fontweight="bold")
        ax.set_xticks(np.arange(len(PHRASE_KEYS)), PHRASE_TYPES)
        ax.set_yticks([0, 1], ["CogAlign", "ET-ATT"] if domain_index == 0 else ["", ""])
        ax.tick_params(length=0, labelsize=5.3)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks(np.arange(-0.5, len(PHRASE_KEYS), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.0)
        ax.tick_params(which="minor", bottom=False, left=False)
    axes[1].text(0.5, 1.25, "Recovery differences within Proxy-expanded keyphrases", transform=axes[1].transAxes, ha="center", va="bottom", fontsize=7.2, fontweight="bold")
    panel_label(axes[0], "c")
    figure = axes[-1].figure
    color_axis = figure.add_axes([0.36, 0.045, 0.28, 0.012])
    colorbar = figure.colorbar(image, cax=color_axis, orientation="horizontal")
    colorbar.set_label("Recovery difference vs TEXT (percentage points)", fontsize=5.8)
    colorbar.ax.tick_params(labelsize=5.3, length=2)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel_a = pd.read_csv(DATA / "panel_a_source.csv")
    panel_b = pd.read_csv(DATA / "panel_b_source.csv")
    panel_c = pd.read_csv(DATA / "panel_c_source.csv")

    fig = plt.figure(figsize=(7.2, 7.55), constrained_layout=False)
    outer = fig.add_gridspec(3, 1, height_ratios=[0.95, 1.35, 1.12], hspace=0.78)
    ax_a = fig.add_subplot(outer[0, 0])
    middle = outer[1, 0].subgridspec(1, 3, wspace=0.12)
    axes_b = [fig.add_subplot(middle[0, index]) for index in range(3)]
    bottom = outer[2, 0].subgridspec(1, 3, wspace=0.12)
    axes_c = [fig.add_subplot(bottom[0, index]) for index in range(3)]

    draw_panel_a(ax_a, panel_a)
    draw_panel_b(axes_b, panel_b)
    draw_panel_c(axes_c, panel_c)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.96, bottom=0.14)

    stem = OUT / "figure2_proxy_et_coverage_and_recovery"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    contract = {
        "archetype": "quantitative grid",
        "backend": "python",
        "final_size": "approximately 183 mm wide",
        "panel_map": {
            "a": "Observed-to-Proxy complete/partial/zero keyphrase coverage on test",
            "b": "Absolute Top-5 gold-keyphrase recovery by coverage transition with document-bootstrap 95% confidence intervals",
            "c": "Recovery difference vs TEXT for phrase types within Proxy-expanded test keyphrases",
        },
        "statistics": {
            "split": "test primary; validation directional replication is supplied separately",
            "seeds": [42, 52, 62],
            "bootstrap_replicates": 10000,
            "bootstrap_unit": "document; all three seeds remain clustered within sampled documents",
            "sparse_rule": "n < 20 is shaded grey and not interpreted",
            "subgroup_p_values": "not computed",
        },
        "source_data": [
            "figure2a_observed_to_proxy_coverage.csv",
            "figure2b_recovery_by_transition.csv",
            "figure2c_proxy_expanded_phrase_type.csv",
        ],
    }
    (OUT / "figure_contract_and_notes.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "PASS", "output": str(stem)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
