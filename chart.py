import io
import logging
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import yfinance as yf

logger = logging.getLogger("marketpulse.chart")

GREEN = "#16a34a"
RED = "#dc2626"
BLUE = "#2563eb"
AMBER = "#b45309"
PURPLE = "#7c3aed"
GRAY = "#6b7280"
TEAL = "#0d9488"
BOX_FILL = "#eff6ff"
GRID_COLOR = "#e5e7eb"
PIE_PALETTE = [BLUE, GREEN, AMBER, RED, PURPLE, TEAL, GRAY]


def _save_fig(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _wrap_title(text, width=42, max_lines=2):
    if not text:
        return text
    wrapped_lines = textwrap.wrap(text, width=width)
    if len(wrapped_lines) > max_lines:
        wrapped_lines = wrapped_lines[:max_lines]
        wrapped_lines[-1] = wrapped_lines[-1].rstrip(".,;: ") + "…"
    return "\n".join(wrapped_lines)


def _fetch_history(ticker):
    for period, interval in (("5d", "1h"), ("1mo", "1d")):
        try:
            data = yf.Ticker(ticker).history(period=period, interval=interval)
        except Exception as exc:
            logger.warning("yfinance fetch failed for %s (%s/%s): %s", ticker, period, interval, exc)
            continue
        if not data.empty and len(data) >= 2:
            return data
    return None


def generate_price_chart(ticker, label=None):
    if not ticker:
        return None

    ticker = ticker.strip().replace("/", "").replace(" ", "")
    data = _fetch_history(ticker)
    if data is None:
        logger.warning("No usable price history for ticker %s", ticker)
        return None

    closes = data["Close"]
    first_price = float(closes.iloc[0])
    last_price = float(closes.iloc[-1])
    pct_change = (last_price - first_price) / first_price * 100
    color = GREEN if pct_change >= 0 else RED
    arrow = "▲" if pct_change >= 0 else "▼"

    fig, ax = plt.subplots(figsize=(6, 3.5), dpi=140)
    ax.plot(closes.index, closes.values, color=color, linewidth=1.8)
    ax.fill_between(closes.index, closes.values, first_price, color=color, alpha=0.08)

    title = _wrap_title(label or ticker, width=42)
    ax.set_title(f"{title}\n{arrow} {pct_change:+.2f}%", color=color, fontsize=13, fontweight="bold", loc="left")
    ax.text(0.99, 0.97, f"{last_price:,.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=11, color="#1a1a1a")

    locator = mdates.AutoDateLocator(minticks=4, maxticks=7)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    ax.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="x", labelsize=8, colors="#666666")
    ax.tick_params(axis="y", labelsize=8, colors="#666666")
    fig.patch.set_facecolor("white")

    fig.tight_layout()
    return _save_fig(fig)


def generate_bar_chart(spec):
    if not spec:
        return None

    title = spec.get("title") or ""
    labels = spec.get("labels") or []
    values = spec.get("values") or []
    unit = spec.get("unit") or ""

    if not labels or not values or len(labels) != len(values):
        logger.warning("Malformed bar_chart spec, skipping: %s", spec)
        return None

    try:
        values = [float(v) for v in values]
    except (TypeError, ValueError):
        logger.warning("Non-numeric values in bar_chart spec, skipping: %s", spec)
        return None

    is_delta = any(v < 0 for v in values) and any(v > 0 for v in values)
    colors = [(GREEN if v >= 0 else RED) for v in values] if is_delta else [BLUE] * len(values)
    fmt = "{:+.2f}{}" if is_delta else "{:.2f}{}"

    fig, ax = plt.subplots(figsize=(6, 3.5), dpi=140)
    bars = ax.bar(labels, values, color=colors, width=0.6)

    ax.set_title(_wrap_title(title, width=42), fontsize=14, fontweight="bold", loc="left", color="#1a1a1a")
    ax.axhline(0, color="#999999", linewidth=0.8)

    for bar, v in zip(bars, values):
        offset = max(abs(v) * 0.03, 0.3 if not unit else 0.02 * max(abs(x) for x in values))
        va = "bottom" if v >= 0 else "top"
        y = bar.get_height() + (offset if v >= 0 else -offset)
        ax.text(bar.get_x() + bar.get_width() / 2, y, fmt.format(v, unit), ha="center", va=va, fontsize=9, color="#1a1a1a")

    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="x", labelsize=9, colors="#444444")
    ax.tick_params(axis="y", labelsize=8, colors="#666666")
    fig.patch.set_facecolor("white")

    fig.tight_layout()
    return _save_fig(fig)


def generate_histogram(spec):
    if not spec:
        return None

    title = spec.get("title") or ""
    values = spec.get("values") or []
    unit = spec.get("unit") or ""

    if len(values) < 5:
        logger.warning("Malformed histogram spec, skipping: %s", spec)
        return None

    try:
        values = [float(v) for v in values]
    except (TypeError, ValueError):
        logger.warning("Non-numeric values in histogram spec, skipping: %s", spec)
        return None

    bins = min(max(int(spec.get("bins") or 8), 4), 20)

    fig, ax = plt.subplots(figsize=(6, 3.5), dpi=140)
    ax.hist(values, bins=bins, color=BLUE, edgecolor="white", linewidth=0.8)

    ax.set_title(_wrap_title(title, width=42), fontsize=14, fontweight="bold", loc="left", color="#1a1a1a")
    if unit:
        ax.set_xlabel(unit, fontsize=9, color="#666666")
    ax.set_ylabel("Count", fontsize=9, color="#666666")

    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="x", labelsize=8, colors="#666666")
    ax.tick_params(axis="y", labelsize=8, colors="#666666")
    fig.patch.set_facecolor("white")

    fig.tight_layout()
    return _save_fig(fig)


def generate_pie_chart(spec):
    if not spec:
        return None

    title = spec.get("title") or ""
    labels = spec.get("labels") or []
    values = spec.get("values") or []

    if not labels or not values or len(labels) != len(values) or len(labels) < 2:
        logger.warning("Malformed pie_chart spec, skipping: %s", spec)
        return None

    try:
        values = [abs(float(v)) for v in values]
    except (TypeError, ValueError):
        logger.warning("Non-numeric values in pie_chart spec, skipping: %s", spec)
        return None

    if sum(values) <= 0:
        logger.warning("Pie chart values sum to zero, skipping: %s", spec)
        return None

    colors = [PIE_PALETTE[i % len(PIE_PALETTE)] for i in range(len(labels))]

    fig, ax = plt.subplots(figsize=(6, 4), dpi=140)
    ax.pie(
        values,
        labels=labels,
        colors=colors,
        autopct="%1.0f%%",
        startangle=90,
        textprops={"fontsize": 9.5, "color": "#1a1a1a"},
        wedgeprops={"linewidth": 1.5, "edgecolor": "white"},
    )
    ax.set_title(_wrap_title(title, width=42), fontsize=14, fontweight="bold", loc="left", color="#1a1a1a")
    fig.patch.set_facecolor("white")

    fig.tight_layout()
    return _save_fig(fig)


def generate_trend_chart(spec):
    if not spec:
        return None

    title = spec.get("title") or ""
    labels = spec.get("labels") or []
    values = spec.get("values") or []
    unit = spec.get("unit") or ""
    fit = (spec.get("fit") or "linear").lower()

    if not labels or not values or len(labels) != len(values) or len(labels) < 3:
        logger.warning("Malformed trend_chart spec, skipping: %s", spec)
        return None

    try:
        values = [float(v) for v in values]
    except (TypeError, ValueError):
        logger.warning("Non-numeric values in trend_chart spec, skipping: %s", spec)
        return None

    x = np.arange(len(values))
    degree = 3 if fit == "cubic" and len(values) >= 4 else 1

    fig, ax = plt.subplots(figsize=(6, 3.5), dpi=140)
    ax.plot(x, values, "o-", color=BLUE, linewidth=1.6, markersize=5, label="Actual")

    try:
        coeffs = np.polyfit(x, values, degree)
        x_smooth = np.linspace(x.min(), x.max(), 100)
        y_smooth = np.polyval(coeffs, x_smooth)
        fit_label = "Cubic trend" if degree == 3 else "Linear trend"
        ax.plot(x_smooth, y_smooth, "--", color=AMBER, linewidth=1.8, label=fit_label)
    except Exception as exc:
        logger.warning("Trend fit failed: %s", exc)

    ax.set_title(_wrap_title(title, width=42), fontsize=14, fontweight="bold", loc="left", color="#1a1a1a")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    if unit:
        ax.set_ylabel(unit, fontsize=9, color="#666666")

    ax.legend(loc="best", fontsize=8.5, frameon=False)
    ax.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="x", labelsize=8, colors="#666666")
    ax.tick_params(axis="y", labelsize=8, colors="#666666")
    fig.patch.set_facecolor("white")

    fig.tight_layout()
    return _save_fig(fig)


def resolve_visual(result, label=None):
    """Dispatch to the right chart renderer based on result['visual_type']."""
    visual_type = result.get("visual_type") or "none"
    if visual_type == "price_chart":
        return generate_price_chart(result.get("ticker"), label=label)
    if visual_type == "bar_chart":
        return generate_bar_chart(result.get("bar_chart"))
    if visual_type == "histogram":
        return generate_histogram(result.get("histogram"))
    if visual_type == "pie_chart":
        return generate_pie_chart(result.get("pie_chart"))
    if visual_type == "trend_chart":
        return generate_trend_chart(result.get("trend_chart"))
    if visual_type == "flowchart":
        return generate_flowchart(result.get("flowchart"))
    return None


def generate_flowchart(spec):
    if not spec:
        return None

    steps = [s for s in (spec.get("steps") or []) if s]
    if len(steps) < 2:
        logger.warning("Malformed flowchart spec, skipping: %s", spec)
        return None

    n = len(steps)
    box_height = 0.8
    gap = 0.5
    total_height = n * box_height + (n - 1) * gap

    fig, ax = plt.subplots(figsize=(6, max(2.5, total_height + 0.6)), dpi=140)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, total_height + 0.3)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    for i, step in enumerate(steps):
        y_top = total_height + 0.15 - i * (box_height + gap)
        y_bottom = y_top - box_height
        wrapped = textwrap.fill(step, width=30)

        box = FancyBboxPatch(
            (0.05, y_bottom),
            0.9,
            box_height,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=1.5,
            edgecolor=BLUE,
            facecolor=BOX_FILL,
        )
        ax.add_patch(box)
        ax.text(0.5, (y_top + y_bottom) / 2, wrapped, ha="center", va="center", fontsize=10.5, color="#1a1a1a")

        if i < n - 1:
            ax.annotate(
                "",
                xy=(0.5, y_bottom - gap + 0.05),
                xytext=(0.5, y_bottom - 0.05),
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.5),
            )

    fig.tight_layout()
    return _save_fig(fig)
