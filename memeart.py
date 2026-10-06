import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Wedge

from chart import GREEN, RED, BLUE, AMBER, PURPLE, GRAY, _add_watermark, _save_fig, _wrap_title

logger = logging.getLogger("marketpulse.memeart")

# Simple, original line-art cartoon icons -- NOT copies of any existing meme template/character
# (Wojak, Pepe, "this is fine" dog, etc.). Those are copyrighted and this project's own
# precedent (see chart.py's TradingView note in README) is "render it ourselves, free and
# keyless" rather than scrape or reuse someone else's protected image. Each mood is a plain
# round face or simple icon built from matplotlib primitives, in the same flat-color palette
# chart.py already uses everywhere else, so a meme post still looks like it belongs in the
# same email as the real charts.
MEME_MOODS = ("crying", "panic", "rocket", "diamond_hands", "confused", "facepalm", "cool")

FACE_FILL = "#fde68a"
FACE_EDGE = "#92400e"


def _new_fig():
    fig, ax = plt.subplots(figsize=(5, 4.2), dpi=140)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")
    return fig, ax


def _face(ax, color=FACE_FILL, edge=FACE_EDGE):
    ax.add_patch(Circle((5, 5.3), 2.6, facecolor=color, edgecolor=edge, linewidth=2.5, zorder=1))


def _draw_crying(ax):
    _face(ax)
    ax.add_patch(Circle((4.1, 5.6), 0.28, facecolor="#1a1a1a", zorder=2))
    ax.add_patch(Circle((5.9, 5.6), 0.28, facecolor="#1a1a1a", zorder=2))
    for cx in (4.1, 5.9):
        ax.add_patch(Polygon([(cx - 0.18, 5.2), (cx + 0.18, 5.2), (cx, 4.3)], facecolor=BLUE, zorder=2))
    ax.plot([4.4, 5.6], [4.2, 4.2], color="#1a1a1a", linewidth=3, solid_capstyle="round")
    arc_x = [4.5 + 0.1 * i for i in range(11)]
    ax.plot(arc_x, [4.0 - 0.15 * abs(5 - i) / 5 for i in range(11)], color="#1a1a1a", linewidth=3)


def _draw_panic(ax):
    _face(ax, color="#fecaca", edge=RED)
    for cx in (4.1, 5.9):
        ax.add_patch(Circle((cx, 5.6), 0.45, facecolor="white", edgecolor="#1a1a1a", linewidth=2, zorder=2))
        ax.add_patch(Circle((cx, 5.6), 0.18, facecolor="#1a1a1a", zorder=3))
    ax.add_patch(Polygon([(4.3, 4.0), (4.7, 4.5), (5.0, 4.0), (5.3, 4.5), (5.7, 4.0)],
                          closed=False, facecolor="none", edgecolor="#1a1a1a", linewidth=3, zorder=2))
    ax.add_patch(Polygon([(7.3, 6.6), (7.55, 7.3), (7.8, 6.6), (7.55, 6.3)], facecolor=BLUE, zorder=2))


def _draw_rocket(ax):
    body = Polygon([(4.5, 2.5), (5.5, 2.5), (5.7, 6.5), (5.0, 7.8), (4.3, 6.5)],
                    facecolor="white", edgecolor="#1a1a1a", linewidth=2.5, zorder=2)
    ax.add_patch(body)
    ax.add_patch(Circle((5.0, 6.0), 0.45, facecolor=BLUE, edgecolor="#1a1a1a", linewidth=2, zorder=3))
    ax.add_patch(Polygon([(4.5, 3.4), (3.6, 2.2), (4.5, 2.7)], facecolor=RED, zorder=2))
    ax.add_patch(Polygon([(5.5, 3.4), (6.4, 2.2), (5.5, 2.7)], facecolor=RED, zorder=2))
    for i, (dx, h) in enumerate(((0, 1.6), (-0.35, 1.1), (0.35, 1.1))):
        ax.add_patch(Polygon(
            [(5.0 + dx - 0.35, 2.5), (5.0 + dx + 0.35, 2.5), (5.0 + dx, 2.5 - h)],
            facecolor=AMBER if i == 0 else "#fbbf24", zorder=1,
        ))


def _draw_diamond_hands(ax):
    ax.add_patch(FancyBboxPatch((3.2, 2.8), 3.6, 2.6, boxstyle="round,pad=0,rounding_size=0.5",
                                 facecolor="#fcd34d", edgecolor="#1a1a1a", linewidth=2.5, zorder=1))
    for x in (4.0, 4.9, 5.8):
        ax.add_patch(FancyBboxPatch((x - 0.25, 5.2), 0.5, 1.4, boxstyle="round,pad=0,rounding_size=0.25",
                                     facecolor="#fcd34d", edgecolor="#1a1a1a", linewidth=2, zorder=1))
    diamond = Polygon([(5, 8.3), (6.1, 6.9), (5, 5.6), (3.9, 6.9)],
                       facecolor="#bfdbfe", edgecolor=BLUE, linewidth=2.5, zorder=2)
    ax.add_patch(diamond)
    ax.plot([3.9, 6.1], [6.9, 6.9], color=BLUE, linewidth=1.2, zorder=3)
    ax.plot([5, 5], [5.9, 7.9], color=BLUE, linewidth=1, zorder=3)


def _draw_confused(ax):
    _face(ax)
    for cx in (4.1, 5.9):
        ax.add_patch(Circle((cx, 5.6), 0.22, facecolor="#1a1a1a", zorder=2))
    ax.plot([4.0, 4.8], [4.3, 4.5], color="#1a1a1a", linewidth=3, solid_capstyle="round")
    ax.text(7.6, 7.3, "?", fontsize=34, fontweight="bold", color=PURPLE, ha="center", va="center")
    ax.text(6.6, 6.4, "?", fontsize=20, fontweight="bold", color=PURPLE, ha="center", va="center", alpha=0.7)


def _draw_facepalm(ax):
    _face(ax)
    ax.add_patch(Circle((5.9, 5.6), 0.26, facecolor="#1a1a1a", zorder=2))
    ax.add_patch(Wedge((4.6, 5.4), 1.8, 60, 300, facecolor="#fde68a", edgecolor=FACE_EDGE, linewidth=2.5, zorder=3))
    for dx in (-0.35, 0, 0.35):
        ax.plot([4.6 + dx, 4.3 + dx], [6.9, 8.3], color=FACE_EDGE, linewidth=2.5, zorder=2)


def _draw_cool(ax):
    _face(ax, color="#bbf7d0", edge=GREEN)
    ax.add_patch(FancyBboxPatch((3.5, 5.3), 1.3, 0.85, boxstyle="round,pad=0,rounding_size=0.12",
                                 facecolor="#1a1a1a", zorder=2))
    ax.add_patch(FancyBboxPatch((5.2, 5.3), 1.3, 0.85, boxstyle="round,pad=0,rounding_size=0.12",
                                 facecolor="#1a1a1a", zorder=2))
    ax.plot([4.8, 5.2], [5.7, 5.7], color="#1a1a1a", linewidth=4, zorder=2)
    ax.plot([4.3, 5.7], [4.2, 4.2], color="#1a1a1a", linewidth=3, solid_capstyle="round", zorder=2)


_DRAW_FUNCS = {
    "crying": _draw_crying,
    "panic": _draw_panic,
    "rocket": _draw_rocket,
    "diamond_hands": _draw_diamond_hands,
    "confused": _draw_confused,
    "facepalm": _draw_facepalm,
    "cool": _draw_cool,
}


def generate_meme_image(mood, caption, label=None, stats_out=None):
    """Renders one of the fixed, original cartoon moods in MEME_MOODS with `caption` as the
    title above it. Falls back to "confused" for an unrecognized mood rather than returning
    None, since a meme post without ANY image defeats the point less gracefully than a slightly
    wrong expression. Returns PNG bytes, same contract as every chart.py generate_* function
    (so poster.py/report.py need no special-casing for meme images)."""
    draw_fn = _DRAW_FUNCS.get(mood)
    if draw_fn is None:
        logger.warning("Unrecognized meme mood '%s', falling back to 'confused'", mood)
        mood = "confused"
        draw_fn = _draw_confused

    fig, ax = _new_fig()
    draw_fn(ax)

    title = _wrap_title(caption or label or "", width=36)
    if title:
        ax.text(5, 9.4, title, fontsize=14, fontweight="bold", color="#1a1a1a", ha="center", va="top")

    if stats_out is not None:
        stats_out.update({"source": "memeart", "mood": mood, "caption": caption})
    return _save_fig(fig)
