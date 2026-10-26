"""Draw the README's one-row, four-panel evaluation chart.

General data: assets/figures/general/results.json (General-trained RL checkpoint).
Sokoban data: https://github.com/Valen-Team/Valen/blob/453f221/docs/technical.md (500 single-step questions).
The shared Preview display name follows the README; provenance is in that pinned historical source.
Style reference: https://huggingface.co/internlm/Atria-Dawn-Preview
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, HPacker, OffsetImage, TextArea
from matplotlib.path import Path as PlotPath
from matplotlib.patches import PathPatch

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "assets/figures/evaluation-results"
COLORS = {"valen": "#009C83", "qwen": "#E6E6E6"}
INK, MUTED = "#111111", "#6D6D6D"
ORDER = (("2B", "rlcd30k"), ("0.8B", "baseline"), ("2B", "baseline"))
# size, method, accuracy (%), mean end-to-end latency (ms)
SOKOBAN = [
    ("0.8B", "baseline", 31.60, 152.54),
    ("0.8B", "sft100k", 80.60, 114.06),
    ("0.8B", "rlcd30k", 83.00, 113.95),
    ("2B", "baseline", 27.20, 155.73),
    ("2B", "sft100k", 83.60, 126.82),
    ("2B", "rlcd30k", 87.60, 127.31),
]


def brand_icons():
    # Reuse only the eye viewport of the unmodified v5 logo.
    return {
        "valen": plt.imread(ROOT / "assets/branding/valen-logo-v5.png")[120:562, 50:714],
        "qwen": plt.imread(ROOT / "assets/branding/qwen/qwen-icon.png"),
    }


def icon_box(icon, height):
    return OffsetImage(icon, zoom=height / icon.shape[0], interpolation="lanczos")


def rounded_bar(ax, x, value, lower, upper, color):
    """Round the top corners only; keep the actual height equal to the data."""
    half, radius_x, radius_y = .31, .055, (upper - lower) * .007
    left, right = x - half, x + half
    vertices = [(left, lower), (left, value-radius_y), (left, value),
                (left+radius_x, value), (right-radius_x, value),
                (right, value), (right, value-radius_y), (right, lower), (left, lower)]
    codes = [PlotPath.MOVETO, PlotPath.LINETO, PlotPath.CURVE3, PlotPath.CURVE3,
             PlotPath.LINETO, PlotPath.CURVE3, PlotPath.CURVE3, PlotPath.LINETO,
             PlotPath.CLOSEPOLY]
    ax.add_patch(PathPatch(PlotPath(vertices, codes), facecolor=color, edgecolor="none"))


def draw_panel(ax, task, metric, rows, value_index, limits, icons):
    lower, upper = limits
    ax.set_xlim(-.55, 2.55)
    ax.set_ylim(lower, upper)
    ax.set_xticks([0, 1, 2], ["2B", "0.8B", "2B"])
    ax.tick_params(axis="x", length=0, pad=8, labelsize=10, colors=MUTED)
    ax.set_yticks([lower, upper] if lower else [])
    ax.tick_params(axis="y", length=0, pad=3, labelsize=8, colors=MUTED)
    for name, spine in ax.spines.items():
        spine.set_visible(name == "bottom")
        spine.set_color("#DCDCDC")
        spine.set_linewidth(.8)
    ax.text(.5, 1.37, task, ha="center", va="bottom", transform=ax.transAxes,
            fontsize=20, weight="bold", color=INK)
    ax.text(.5, 1.23, metric, ha="center", va="bottom", transform=ax.transAxes,
            fontsize=12.5, color=MUTED)
    if value_index == 3:
        valen_latency = next(row[3] for row in rows if row[1] == "rlcd30k")
        baseline_latency = next(row[3] for row in rows if row[:2] == ("2B", "baseline"))
        speedup = baseline_latency / valen_latency
        saved_ms = baseline_latency - valen_latency
        ax.text(.5, 1.08, f"{speedup:.2f}× vs Qwen3.5-2B · −{saved_ms:.2f} ms",
                ha="center", va="bottom", transform=ax.transAxes,
                fontsize=10, weight="bold", color="#006C5B")
    for x, row in enumerate(rows):
        brand = "valen" if row[1] == "rlcd30k" else "qwen"
        value = row[value_index]
        rounded_bar(ax, x, value, lower, upper, COLORS[brand])
        # Match the reference: highlighted values above bars, baseline values inside.
        highlighted = brand == "valen"
        ax.annotate(f"{value:.2f}", (x, value), xytext=(0, 4 if highlighted else -7),
                    textcoords="offset points", ha="center",
                    va="bottom" if highlighted else "top", fontsize=13,
                    weight="bold" if highlighted else "normal",
                    color="#006C5B" if highlighted else MUTED)
        ax.add_artist(AnnotationBbox(icon_box(icons[brand], 22), (x, value),
                                    xybox=(0, 23 if highlighted else 7), xycoords="data",
                                    boxcoords="offset points", box_alignment=(.5, 0),
                                    frameon=False, pad=0, annotation_clip=False))


def main():
    snapshot = json.loads((ROOT / "assets/figures/general/results.json").read_text())
    general = {(r["size"], r["method"]):
               (r["size"], r["method"], r["accuracy_pct"], r["mean_latency_ms"])
               for r in snapshot["models"]}
    sokoban = {(r[0], r[1]): r for r in SOKOBAN}
    general_rows, sokoban_rows = ([data[key] for key in ORDER] for data in (general, sokoban))
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11, "text.color": INK,
        "svg.fonttype": "path", "pdf.fonttype": 42,
        "svg.hashsalt": "valen-four-panel-results",
    })
    icons = brand_icons()
    fig, axes = plt.subplots(1, 4, figsize=(15.8, 4.6), facecolor="white")
    fig.subplots_adjust(left=.025, right=.985, bottom=.21, top=.682, wspace=.24)
    panels = [
        ("General", "Accuracy (%) ↑", general_rows, 2, (70, 82)),
        ("General", "Latency (ms) ↓", general_rows, 3, (140, 260)),
        ("Sokoban", "Accuracy (%) ↑", sokoban_rows, 2, (0, 100)),
        ("Sokoban", "Latency (ms) ↓", sokoban_rows, 3, (100, 180)),
    ]
    for ax, panel in zip(axes, panels):
        draw_panel(ax, *panel, icons)
    legend_items = []
    for brand, name in [("valen", "Valen-Preview-0923"), ("qwen", "Qwen3.5-0.8B"),
                        ("qwen", "Qwen3.5-2B")]:
        legend_items.append(HPacker(children=[icon_box(icons[brand], 20),
            TextArea(name, textprops={"fontsize": 14, "color": "#333333",
                                     "weight": "bold" if brand == "valen" else "normal"})],
            align="center", pad=0, sep=6))
    legend = HPacker(children=legend_items, align="center", pad=0, sep=24)
    fig.add_artist(AnnotationBbox(legend, (.5, .065), xycoords=fig.transFigure,
                                 frameon=False, box_alignment=(.5, .5), pad=0))
    for suffix in ("png", "svg", "pdf"):
        kwargs = {"dpi": 220} if suffix == "png" else {"metadata": {"Date": None}} if suffix == "svg" else {
            "metadata": {"CreationDate": None, "ModDate": None}}
        fig.savefig(OUTPUT.with_suffix("."+suffix), facecolor="white", **kwargs)
    svg = OUTPUT.with_suffix(".svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines())+"\n")
    plt.close(fig)
    print(f"Wrote {OUTPUT.relative_to(ROOT)}.png / .svg / .pdf")


if __name__ == "__main__":
    main()
