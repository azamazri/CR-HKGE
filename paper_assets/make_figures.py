"""Generate IEEE-paper figures for CR-HKGE (grayscale, vector-quality PNG @300dpi)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

OUT = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
})

EDGE = "#222222"
FILL = "#f2f2f2"
FILL2 = "#dddddd"
FILL3 = "#c9c9c9"


def box(ax, x, y, w, h, text, fc=FILL, fs=9, bold=False):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                       linewidth=1.1, edgecolor=EDGE, facecolor=fc)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, weight="bold" if bold else "normal", wrap=True)


def arrow(ax, x1, y1, x2, y2, style="-|>"):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=11,
                        linewidth=1.1, color=EDGE, shrinkA=2, shrinkB=2)
    ax.add_patch(a)


# ---------- Figure 1: Overall research pipeline ----------
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(3.4, 4.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 12); ax.axis("off")
    steps = [
        ("Aromatique product catalog\n(340 perfume products)", FILL),
        ("Global fragrance reference\nenrichment (inspired_by)", FILL2),
        ("Fragrance-specific\nheterogeneous KG\n(998 entities, 9,250 triples)", FILL),
        ("Content-based positive-pair\nconstruction (surrogate\nsupervision)", FILL2),
        ("CR-HKGE training\n(alternating BPR + TransR)", FILL3),
        ("Top-K recommendation\nevaluation", FILL),
    ]
    y = 10.4; h = 1.5; w = 8.4; x = 0.8
    centers = []
    for txt, fc in steps:
        box(ax, x, y, w, h, txt, fc=fc, fs=8.5)
        centers.append((x + w / 2, y))
        y -= 1.78
    for i in range(len(centers) - 1):
        cx, cy = centers[i]
        arrow(ax, cx, cy, cx, centers[i + 1][1] + h)
    fig.tight_layout(pad=0.2)
    fig.savefig(os.path.join(OUT, "fig1_pipeline.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------- Figure 2: Heterogeneous KG schema ----------
def fig_schema():
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 9); ax.axis("off")

    def node(x, y, t, fc, w=2.5, h=0.9, fs=8):
        box(ax, x - w / 2, y - h / 2, w, h, t, fc=fc, fs=fs)

    # local product center
    node(5, 4.5, "product", FILL3, w=2.4, h=1.0, fs=9)
    # local attributes
    node(1.6, 7.6, "accord", FILL)
    node(5, 8.0, "note", FILL)
    node(8.4, 7.6, "family", FILL)
    # global side
    node(8.6, 4.5, "global_ref", FILL2, w=2.6)
    node(8.6, 1.4, "global_accord", FILL, w=2.8)
    node(4.8, 1.0, "global_family", FILL, w=2.8)
    node(1.4, 2.3, "product\n(sem.)", FILL, w=2.2, h=1.0, fs=8)

    def edge(x1, y1, x2, y2, lab, off=0.0, fs=7):
        arrow(ax, x1, y1, x2, y2)
        ax.text((x1 + x2) / 2 + off, (y1 + y2) / 2, lab, ha="center", va="center",
                fontsize=fs, style="italic",
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none"))

    edge(4.2, 4.9, 2.0, 7.2, "has_accord", -0.1)
    edge(5.0, 5.0, 5.0, 7.6, "has_visual_note")
    edge(5.8, 4.9, 7.9, 7.2, "belongs_to_family", 0.2)
    edge(6.2, 4.5, 7.3, 4.5, "inspired_by", 0.0, fs=7)
    edge(2.2, 3.0, 4.0, 4.2, "sem_similar", -0.2)
    edge(8.6, 4.0, 8.6, 1.9, "has_global_accord", 1.0, fs=6.5)
    edge(8.0, 4.1, 5.4, 1.5, "belongs_to_\nglobal_family", -0.3, fs=6.5)
    fig.tight_layout(pad=0.2)
    fig.savefig(os.path.join(OUT, "fig2_schema.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------- Figure 3: CR-HKGE architecture ----------
def fig_arch():
    fig, ax = plt.subplots(figsize=(3.4, 4.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 12); ax.axis("off")
    layers = [
        ("Layer 1: TransR KG Embedding\n(entity / relation embeddings)", FILL),
        ("Layer 2: Relation-Type Attention\n(learnable weight per relation)", FILL2),
        ("Layer 3: Cross-Reference Propagation\n(global -> local via inspired_by)", FILL3),
        ("Layer 4: Bi-Interaction Aggregation\n+ layer concatenation", FILL2),
        ("Prediction: profile-product score", FILL),
    ]
    y = 10.2; h = 1.55; w = 8.8; x = 0.6
    centers = []
    for txt, fc in layers:
        box(ax, x, y, w, h, txt, fc=fc, fs=8.3)
        centers.append((x + w / 2, y))
        y -= 2.0
    for i in range(len(centers) - 1):
        cx, cy = centers[i]
        arrow(ax, cx, cy, cx, centers[i + 1][1] + h)
    # novelty markers (N3=relation attention, N2=cross-reference)
    ax.text(9.7, 10.2 - 2.0 + h / 2, "N3", fontsize=8, weight="bold", ha="center")
    ax.text(9.7, 10.2 - 4.0 + h / 2, "N2", fontsize=8, weight="bold", ha="center")
    fig.tight_layout(pad=0.2)
    fig.savefig(os.path.join(OUT, "fig3_architecture.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


fig_pipeline()
fig_schema()
fig_arch()
print("figures written to", OUT)
