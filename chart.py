import io
import json
import logging
import textwrap
import urllib.error
import urllib.parse
import urllib.request

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import mplfinance as mpf
import numpy as np
import yfinance as yf
from PIL import Image, ImageDraw

logger = logging.getLogger("marketpulse.chart")

WIKI_API_URL = "https://en.wikipedia.org/w/api.php"
WIKI_USER_AGENT = "MarketPulseAI/1.0 (single-user personal news digest)"

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


def _fetch_daily_history(ticker, period="3mo"):
    """Daily-bar history for technical chart types (candlestick/Renko/PnF) -- these are
    conventionally read on daily candles, unlike the simple at-a-glance price_chart above."""
    try:
        data = yf.Ticker(ticker).history(period=period, interval="1d")
    except Exception as exc:
        logger.warning("yfinance daily fetch failed for %s (%s): %s", ticker, period, exc)
        return None
    if data is None or data.empty:
        return None
    data = data.dropna(subset=["Open", "High", "Low", "Close"])
    return data if len(data) >= 10 else None


def detect_candlestick_patterns(data, lookback=5):
    """Rule-based detection of well-defined reversal patterns on the most recent `lookback`
    candles. Pure arithmetic on real OHLC data -- the model never asserts these, so there is
    no fabrication risk; a pattern is only ever shown if the price data actually satisfies its
    textbook definition. Returns a list of (position, pattern_name, direction) tuples."""
    patterns = []
    n = len(data)
    opens = data["Open"].values
    highs = data["High"].values
    lows = data["Low"].values
    closes = data["Close"].values

    start = max(2, n - lookback)
    for i in range(start, n):
        o, h, l, c = opens[i], highs[i], lows[i], closes[i]
        rng = h - l
        if rng <= 0:
            continue
        body = abs(c - o)
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l

        if body <= 0.1 * rng:
            patterns.append((i, "Doji", "neutral"))
        elif body > 0 and lower_wick >= 2 * body and upper_wick <= 0.3 * body:
            patterns.append((i, "Hammer", "bullish"))
        elif body > 0 and upper_wick >= 2 * body and lower_wick <= 0.3 * body:
            patterns.append((i, "Inverted Hammer", "bearish"))

        po, pc = opens[i - 1], closes[i - 1]
        if pc < po and c > o and o < pc and c > po:
            patterns.append((i, "Bullish Engulfing", "bullish"))
        elif pc > po and c < o and o > pc and c < po:
            patterns.append((i, "Bearish Engulfing", "bearish"))

        if i >= 2:
            o1, c1 = opens[i - 2], closes[i - 2]
            o2, c2 = opens[i - 1], closes[i - 1]
            body1 = abs(c1 - o1)
            body2 = abs(c2 - o2)
            midpoint1 = (o1 + c1) / 2
            if c1 < o1 and body1 > 0 and body2 <= 0.3 * body1 and c > o and c > midpoint1:
                patterns.append((i, "Morning Star", "bullish"))
            elif c1 > o1 and body1 > 0 and body2 <= 0.3 * body1 and c < o and c < midpoint1:
                patterns.append((i, "Evening Star", "bearish"))

    return patterns


def _technical_style():
    mc = mpf.make_marketcolors(up=GREEN, down=RED, edge="inherit", wick="inherit", volume="inherit")
    return mpf.make_mpf_style(marketcolors=mc, gridcolor=GRID_COLOR, facecolor="white", figcolor="white")


def generate_candlestick_chart(ticker, label=None):
    if not ticker:
        return None

    ticker = ticker.strip().replace("/", "").replace(" ", "")
    data = _fetch_daily_history(ticker)
    if data is None:
        logger.warning("No usable daily history for ticker %s", ticker)
        return None

    first_close = float(data["Close"].iloc[0])
    last_close = float(data["Close"].iloc[-1])
    pct_change = (last_close - first_close) / first_close * 100
    arrow = "▲" if pct_change >= 0 else "▼"
    title = f"{_wrap_title(label or ticker, width=42)}\n{arrow} {pct_change:+.2f}%"

    try:
        fig, axlist = mpf.plot(
            data, type="candle", style=_technical_style(), returnfig=True,
            figsize=(7, 4.8), title=title, tight_layout=True,
        )
    except Exception as exc:
        logger.warning("Candlestick chart failed for %s: %s", ticker, exc)
        return None

    ax = axlist[0]
    span = float(data["High"].max() - data["Low"].min()) or 1.0
    for i, name, direction in detect_candlestick_patterns(data, lookback=5):
        color = GREEN if direction == "bullish" else RED if direction == "bearish" else GRAY
        y = float(data["High"].iloc[i])
        ax.annotate(
            name,
            xy=(i, y),
            xytext=(i, y + span * 0.12),
            ha="center",
            fontsize=8,
            color=color,
            fontweight="bold",
            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2),
        )

    return _save_fig(fig)


def generate_renko_chart(ticker, label=None):
    if not ticker:
        return None

    ticker = ticker.strip().replace("/", "").replace(" ", "")
    data = _fetch_daily_history(ticker)
    if data is None:
        logger.warning("No usable daily history for ticker %s", ticker)
        return None

    title = _wrap_title(label or ticker, width=42)
    try:
        fig, axlist = mpf.plot(
            data, type="renko", style=_technical_style(), returnfig=True,
            figsize=(7, 4.8), title=title, tight_layout=True,
        )
    except Exception as exc:
        logger.warning("Renko chart failed for %s: %s", ticker, exc)
        return None
    return _save_fig(fig)


def generate_pnf_chart(ticker, label=None):
    if not ticker:
        return None

    ticker = ticker.strip().replace("/", "").replace(" ", "")
    data = _fetch_daily_history(ticker)
    if data is None:
        logger.warning("No usable daily history for ticker %s", ticker)
        return None

    title = _wrap_title(label or ticker, width=42)
    try:
        fig, axlist = mpf.plot(
            data, type="pnf", style=_technical_style(), returnfig=True,
            figsize=(7, 4.8), title=title, tight_layout=True,
        )
    except Exception as exc:
        logger.warning("Point & Figure chart failed for %s: %s", ticker, exc)
        return None
    return _save_fig(fig)


def generate_ohlc_chart(ticker, label=None):
    if not ticker:
        return None

    ticker = ticker.strip().replace("/", "").replace(" ", "")
    data = _fetch_daily_history(ticker)
    if data is None:
        logger.warning("No usable daily history for ticker %s", ticker)
        return None

    title = _wrap_title(label or ticker, width=42)
    try:
        fig, axlist = mpf.plot(
            data, type="ohlc", style=_technical_style(), returnfig=True,
            figsize=(7, 4.8), title=title, tight_layout=True,
        )
    except Exception as exc:
        logger.warning("OHLC chart failed for %s: %s", ticker, exc)
        return None
    return _save_fig(fig)


def _to_heikin_ashi(data):
    o, h, l, c = data["Open"].values, data["High"].values, data["Low"].values, data["Close"].values
    ha_close = (o + h + l + c) / 4
    ha_open = np.empty(len(data))
    ha_open[0] = (o[0] + c[0]) / 2
    for i in range(1, len(data)):
        ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2
    ha_high = np.maximum.reduce([h, ha_open, ha_close])
    ha_low = np.minimum.reduce([l, ha_open, ha_close])

    ha = data.copy()
    ha["Open"], ha["High"], ha["Low"], ha["Close"] = ha_open, ha_high, ha_low, ha_close
    return ha


def generate_heikin_ashi_chart(ticker, label=None):
    if not ticker:
        return None

    ticker = ticker.strip().replace("/", "").replace(" ", "")
    data = _fetch_daily_history(ticker)
    if data is None:
        logger.warning("No usable daily history for ticker %s", ticker)
        return None

    ha = _to_heikin_ashi(data)
    title = f"{_wrap_title(label or ticker, width=42)}\n(Heikin-Ashi)"
    try:
        fig, axlist = mpf.plot(
            ha, type="candle", style=_technical_style(), returnfig=True,
            figsize=(7, 4.8), title=title, tight_layout=True,
        )
    except Exception as exc:
        logger.warning("Heikin-Ashi chart failed for %s: %s", ticker, exc)
        return None
    return _save_fig(fig)


def _kagi_vertices(closes, reversal_pct=4.0):
    """Collapses a close-price series into Kagi swing vertices: price only records a new
    vertex once it reverses by more than reversal_pct from the last extreme, otherwise it
    just rides along extending the current high/low."""
    n = len(closes)
    verts = [(0, closes[0])]
    direction = 0  # 0 = undetermined, 1 = rising, -1 = falling
    extreme_idx, extreme = 0, closes[0]

    for i in range(1, n):
        price = closes[i]
        if direction >= 0 and price >= extreme:
            extreme_idx, extreme, direction = i, price, 1
            verts[-1] = (extreme_idx, extreme)
        elif direction <= 0 and price <= extreme:
            extreme_idx, extreme, direction = i, price, -1
            verts[-1] = (extreme_idx, extreme)
        elif direction >= 0 and price <= extreme * (1 - reversal_pct / 100):
            verts.append((extreme_idx, extreme))
            extreme_idx, extreme, direction = i, price, -1
            verts.append((extreme_idx, extreme))
        elif direction <= 0 and price >= extreme * (1 + reversal_pct / 100):
            verts.append((extreme_idx, extreme))
            extreme_idx, extreme, direction = i, price, 1
            verts.append((extreme_idx, extreme))

    if verts[-1][0] != n - 1:
        verts.append((n - 1, closes[-1]))
    return verts


def generate_kagi_chart(ticker, label=None, reversal_pct=4.0):
    if not ticker:
        return None

    ticker = ticker.strip().replace("/", "").replace(" ", "")
    data = _fetch_daily_history(ticker, period="6mo")
    if data is None:
        logger.warning("No usable daily history for ticker %s", ticker)
        return None

    closes = data["Close"].values
    verts = _kagi_vertices(closes, reversal_pct=reversal_pct)

    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=140)
    for (i1, v1), (i2, v2) in zip(verts, verts[1:]):
        color = GREEN if v2 >= v1 else RED
        ax.plot([i1, i2], [v1, v1], color=color, linewidth=1.8)
        ax.plot([i2, i2], [v1, v2], color=color, linewidth=1.8)

    first_close, last_close = float(closes[0]), float(closes[-1])
    pct_change = (last_close - first_close) / first_close * 100
    arrow = "▲" if pct_change >= 0 else "▼"
    title = _wrap_title(label or ticker, width=42)
    ax.set_title(
        f"{title}\n{arrow} {pct_change:+.2f}% (Kagi, {reversal_pct:.0f}% reversal)",
        fontsize=12.5, fontweight="bold", loc="left", color="#1a1a1a",
    )

    tick_idx = np.linspace(0, len(data) - 1, min(6, len(data))).astype(int)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([data.index[i].strftime("%b %d") for i in tick_idx], fontsize=8, color="#666666")
    ax.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", labelsize=8, colors="#666666")
    fig.patch.set_facecolor("white")

    fig.tight_layout()
    return _save_fig(fig)


def generate_area_chart(ticker, label=None):
    if not ticker:
        return None

    ticker = ticker.strip().replace("/", "").replace(" ", "")
    data = _fetch_daily_history(ticker, period="6mo")
    if data is None:
        logger.warning("No usable daily history for ticker %s", ticker)
        return None

    closes = data["Close"]
    first_price = float(closes.iloc[0])
    last_price = float(closes.iloc[-1])
    pct_change = (last_price - first_price) / first_price * 100
    color = GREEN if pct_change >= 0 else RED
    arrow = "▲" if pct_change >= 0 else "▼"
    floor = float(closes.values.min()) * 0.995

    fig, ax = plt.subplots(figsize=(6, 3.5), dpi=140)
    ax.fill_between(closes.index, closes.values, floor, color=color, alpha=0.35)
    ax.plot(closes.index, closes.values, color=color, linewidth=1.5)
    ax.set_ylim(bottom=floor)

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


def generate_volume_chart(ticker, label=None):
    if not ticker:
        return None

    ticker = ticker.strip().replace("/", "").replace(" ", "")
    data = _fetch_daily_history(ticker, period="3mo")
    if data is None or "Volume" not in data or float(data["Volume"].sum()) <= 0:
        logger.warning("No usable volume history for ticker %s", ticker)
        return None

    colors = [GREEN if c >= o else RED for o, c in zip(data["Open"].values, data["Close"].values)]

    fig, ax = plt.subplots(figsize=(6, 3.5), dpi=140)
    ax.bar(data.index, data["Volume"].values, color=colors, width=1.0)

    title = _wrap_title(label or ticker, width=42)
    ax.set_title(f"{title}\nVolume", fontsize=13, fontweight="bold", loc="left", color="#1a1a1a")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v / 1e6:.1f}M" if v >= 1e6 else f"{v / 1e3:.0f}K")

    locator = mdates.AutoDateLocator(minticks=4, maxticks=7)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="x", labelsize=8, colors="#666666")
    ax.tick_params(axis="y", labelsize=8, colors="#666666")
    fig.patch.set_facecolor("white")

    fig.tight_layout()
    return _save_fig(fig)


def generate_volume_profile_chart(ticker, label=None, bins=20):
    if not ticker:
        return None

    ticker = ticker.strip().replace("/", "").replace(" ", "")
    data = _fetch_daily_history(ticker, period="6mo")
    if data is None or "Volume" not in data or float(data["Volume"].sum()) <= 0:
        logger.warning("No usable volume history for ticker %s", ticker)
        return None

    closes = data["Close"].values
    volumes = data["Volume"].values
    lo, hi = float(closes.min()), float(closes.max())
    if hi <= lo:
        logger.warning("Flat price range, skipping volume profile for %s", ticker)
        return None

    edges = np.linspace(lo, hi, bins + 1)
    bucket_idx = np.clip(np.digitize(closes, edges) - 1, 0, bins - 1)
    bucket_volume = np.zeros(bins)
    for idx, vol in zip(bucket_idx, volumes):
        bucket_volume[idx] += vol
    centers = (edges[:-1] + edges[1:]) / 2
    poc = int(np.argmax(bucket_volume))
    colors = [AMBER if i == poc else BLUE for i in range(bins)]

    fig, ax = plt.subplots(figsize=(6, 4.2), dpi=140)
    ax.barh(centers, bucket_volume, height=(edges[1] - edges[0]) * 0.9, color=colors)
    ax.axhline(centers[poc], color=AMBER, linewidth=0.8, linestyle="--")

    title = _wrap_title(label or ticker, width=42)
    ax.set_title(f"{title}\nVolume Profile (6mo)", fontsize=13, fontweight="bold", loc="left", color="#1a1a1a")
    ax.set_xlabel("Volume", fontsize=9, color="#666666")
    ax.set_ylabel("Price", fontsize=9, color="#666666")

    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="x", labelsize=8, colors="#666666")
    ax.tick_params(axis="y", labelsize=8, colors="#666666")
    fig.patch.set_facecolor("white")

    fig.tight_layout()
    return _save_fig(fig)


YIELD_CURVE_TENORS = [("3M", "^IRX"), ("5Y", "^FVX"), ("10Y", "^TNX"), ("30Y", "^TYX")]


def generate_yield_curve_chart(label=None):
    current, prior = [], []
    for name, sym in YIELD_CURVE_TENORS:
        try:
            hist = yf.Ticker(sym).history(period="2mo", interval="1d")
        except Exception as exc:
            logger.warning("Yield curve fetch failed for %s: %s", sym, exc)
            return None
        if hist is None or hist.empty:
            logger.warning("No yield data for %s", sym)
            return None
        current.append(float(hist["Close"].iloc[-1]))
        prior.append(float(hist["Close"].iloc[0]))

    labels = [t[0] for t in YIELD_CURVE_TENORS]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(6, 3.8), dpi=140)
    ax.plot(x, prior, "o--", color=GRAY, linewidth=1.4, markersize=5, label="~1mo ago")
    ax.plot(x, current, "o-", color=BLUE, linewidth=2, markersize=6, label="Current")

    ax.set_title(_wrap_title(label or "US Treasury Yield Curve", width=42), fontsize=13, fontweight="bold", loc="left", color="#1a1a1a")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Yield (%)", fontsize=9, color="#666666")
    ax.legend(loc="best", fontsize=8.5, frameon=False)

    ax.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="x", labelsize=9, colors="#444444")
    ax.tick_params(axis="y", labelsize=8, colors="#666666")
    fig.patch.set_facecolor("white")

    fig.tight_layout()
    return _save_fig(fig)


def generate_seasonality_chart(ticker, label=None, years=5):
    if not ticker:
        return None

    ticker = ticker.strip().replace("/", "").replace(" ", "")
    try:
        data = yf.Ticker(ticker).history(period=f"{years + 1}y", interval="1d")
    except Exception as exc:
        logger.warning("Seasonality fetch failed for %s: %s", ticker, exc)
        return None
    if data is None or data.empty:
        logger.warning("No usable history for seasonality chart %s", ticker)
        return None

    data = data.dropna(subset=["Close"]).copy()
    data["_year"] = data.index.year
    this_year = int(data.index.max().year)

    fig, ax = plt.subplots(figsize=(6, 3.8), dpi=140)
    plotted = 0
    for yr, group in data.groupby("_year"):
        if len(group) < 20:
            continue
        closes = group["Close"].values
        cum_return = (closes / closes[0] - 1) * 100
        is_current = yr == this_year
        ax.plot(
            np.arange(len(cum_return)), cum_return,
            color=BLUE if is_current else GRAY,
            linewidth=2.2 if is_current else 1.1,
            alpha=1.0 if is_current else 0.55,
            label=str(yr),
        )
        plotted += 1

    if plotted < 2:
        plt.close(fig)
        logger.warning("Not enough distinct years for seasonality chart %s", ticker)
        return None

    title = _wrap_title(label or ticker, width=42)
    ax.set_title(f"{title}\nSeasonality by year", fontsize=13, fontweight="bold", loc="left", color="#1a1a1a")
    ax.set_xlabel("Trading day of year", fontsize=9, color="#666666")
    ax.set_ylabel("Cumulative return (%)", fontsize=9, color="#666666")
    ax.axhline(0, color="#999999", linewidth=0.8)
    ax.legend(loc="best", fontsize=7.5, frameon=False, ncol=2)

    ax.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="x", labelsize=8, colors="#666666")
    ax.tick_params(axis="y", labelsize=8, colors="#666666")
    fig.patch.set_facecolor("white")

    fig.tight_layout()
    return _save_fig(fig)


def _caption_image(image_bytes, title):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        logger.warning("Could not open fetched Wikipedia image: %s", exc)
        return None

    # pithumbsize is a hint, not a hard cap (real-world photos have come back well over
    # 1MB) -- resize ourselves so inline email attachments stay small regardless.
    if max(img.size) > 640:
        img.thumbnail((640, 640))

    caption_height = 22
    canvas = Image.new("RGB", (img.width, img.height + caption_height), "white")
    canvas.paste(img, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, img.height + 4), f"Image: Wikipedia — {title}", fill=(100, 100, 100))

    buf = io.BytesIO()
    canvas.save(buf, format="JPEG", quality=85)
    buf.seek(0)
    return buf.read()


def fetch_wikipedia_image(query, label=None):
    """Pulls a real-world photo/infographic for a story's anchor entity (a company, commodity,
    place, or institution) straight from Wikipedia -- free, no API key. Returns None (caller
    skips the visual) if no matching article or no usable image exists; never forces a weak or
    generic match."""
    if not query:
        return None

    try:
        search_qs = urllib.parse.urlencode(
            {"action": "query", "list": "search", "srsearch": query, "format": "json", "srlimit": 1}
        )
        req = urllib.request.Request(f"{WIKI_API_URL}?{search_qs}", headers={"User-Agent": WIKI_USER_AGENT})
        with urllib.request.urlopen(req, timeout=8) as resp:
            search_data = json.loads(resp.read())
        results = (search_data.get("query") or {}).get("search") or []
        if not results:
            logger.info("No Wikipedia article found for query %r", query)
            return None
        title = results[0]["title"]

        image_qs = urllib.parse.urlencode(
            {"action": "query", "titles": title, "prop": "pageimages", "format": "json", "pithumbsize": 640}
        )
        req = urllib.request.Request(f"{WIKI_API_URL}?{image_qs}", headers={"User-Agent": WIKI_USER_AGENT})
        with urllib.request.urlopen(req, timeout=8) as resp:
            image_data = json.loads(resp.read())
        pages = (image_data.get("query") or {}).get("pages") or {}
        thumbnail = next(iter(pages.values()), {}).get("thumbnail") or {}
        image_url = thumbnail.get("source")
        if not image_url:
            logger.info("No image available on Wikipedia page %r", title)
            return None

        img_req = urllib.request.Request(image_url, headers={"User-Agent": WIKI_USER_AGENT})
        with urllib.request.urlopen(img_req, timeout=8) as resp:
            image_bytes = resp.read()
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, KeyError, TimeoutError) as exc:
        logger.warning("Wikipedia image fetch failed for query %r: %s", query, exc)
        return None

    return _caption_image(image_bytes, title)


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
    horizontal = (spec.get("orientation") or "vertical").lower() == "horizontal"

    if horizontal:
        fig, ax = plt.subplots(figsize=(6, max(3, 0.5 * len(labels) + 1)), dpi=140)
        bars = ax.barh(labels, values, color=colors, height=0.6)
        ax.invert_yaxis()

        ax.set_title(_wrap_title(title, width=42), fontsize=14, fontweight="bold", loc="left", color="#1a1a1a")
        ax.axvline(0, color="#999999", linewidth=0.8)

        for bar, v in zip(bars, values):
            offset = max(abs(v) * 0.03, 0.3 if not unit else 0.02 * max(abs(x) for x in values))
            ha = "left" if v >= 0 else "right"
            x = bar.get_width() + (offset if v >= 0 else -offset)
            ax.text(x, bar.get_y() + bar.get_height() / 2, fmt.format(v, unit), ha=ha, va="center", fontsize=9, color="#1a1a1a")

        ax.grid(axis="x", color=GRID_COLOR, linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis="y", labelsize=9, colors="#444444")
        ax.tick_params(axis="x", labelsize=8, colors="#666666")
        fig.patch.set_facecolor("white")

        fig.tight_layout()
        return _save_fig(fig)

    # Rotate labels when there are many bars or labels are long, to prevent overlap
    max_label_len = max(len(str(l)) for l in labels) if labels else 0
    rotate = max_label_len > 7 or len(labels) > 4
    fig_height = 4.2 if rotate else 3.5

    fig, ax = plt.subplots(figsize=(6, fig_height), dpi=140)
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
    if rotate:
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
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
    if len(labels) > 5:
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    fig.patch.set_facecolor("white")

    fig.tight_layout()
    return _save_fig(fig)


def generate_dumbbell_chart(spec):
    if not spec:
        return None

    title = spec.get("title") or ""
    labels = spec.get("labels") or []
    start_values = spec.get("start_values") or []
    end_values = spec.get("end_values") or []
    start_label = spec.get("start_label") or "Start"
    end_label = spec.get("end_label") or "End"
    unit = spec.get("unit") or ""

    if not labels or len(labels) != len(start_values) or len(labels) != len(end_values):
        logger.warning("Malformed dumbbell_chart spec, skipping: %s", spec)
        return None
    try:
        start_values = [float(v) for v in start_values]
        end_values = [float(v) for v in end_values]
    except (TypeError, ValueError):
        logger.warning("Non-numeric values in dumbbell_chart spec, skipping: %s", spec)
        return None

    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6, max(3, 0.5 * len(labels) + 1)), dpi=140)
    for i, (s, e) in enumerate(zip(start_values, end_values)):
        color = GREEN if e >= s else RED
        ax.plot([s, e], [i, i], color=GRAY, linewidth=2, zorder=1)
        ax.scatter([s], [i], color=GRAY, s=60, zorder=2, label=start_label if i == 0 else None)
        ax.scatter([e], [i], color=color, s=60, zorder=2, label=end_label if i == 0 else None)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9, color="#444444")
    ax.invert_yaxis()
    if unit:
        ax.set_xlabel(unit, fontsize=9, color="#666666")
    ax.set_title(_wrap_title(title, width=42), fontsize=14, fontweight="bold", loc="left", color="#1a1a1a")
    ax.legend(loc="best", fontsize=8.5, frameon=False)

    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="x", labelsize=8, colors="#666666")
    fig.patch.set_facecolor("white")

    fig.tight_layout()
    return _save_fig(fig)


def _multi_series_bar_spec(spec):
    labels = spec.get("labels") or []
    series = spec.get("series") or []
    if not labels or not series:
        return None, None
    if any(len(s.get("values") or []) != len(labels) for s in series):
        return None, None
    try:
        series_values = [[float(v) for v in s["values"]] for s in series]
    except (TypeError, ValueError, KeyError):
        return None, None
    return labels, series_values


def generate_grouped_bar_chart(spec):
    if not spec:
        return None

    title = spec.get("title") or ""
    unit = spec.get("unit") or ""
    series = spec.get("series") or []
    labels, series_values = _multi_series_bar_spec(spec)
    if labels is None:
        logger.warning("Malformed grouped_bar_chart spec, skipping: %s", spec)
        return None

    n_series = len(series)
    x = np.arange(len(labels))
    width = 0.8 / n_series
    palette = [BLUE, GREEN, AMBER, PURPLE, TEAL, RED]

    fig, ax = plt.subplots(figsize=(6, 3.8), dpi=140)
    for i, (s, values) in enumerate(zip(series, series_values)):
        offset = (i - (n_series - 1) / 2) * width
        ax.bar(x + offset, values, width=width * 0.92, color=palette[i % len(palette)], label=s.get("name") or f"Series {i + 1}")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    if unit:
        ax.set_ylabel(unit, fontsize=9, color="#666666")
    ax.set_title(_wrap_title(title, width=42), fontsize=14, fontweight="bold", loc="left", color="#1a1a1a")
    ax.axhline(0, color="#999999", linewidth=0.8)
    ax.legend(loc="best", fontsize=8.5, frameon=False)

    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="x", labelsize=9, colors="#444444")
    ax.tick_params(axis="y", labelsize=8, colors="#666666")
    if max(len(str(l)) for l in labels) > 7 or len(labels) > 4:
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    fig.patch.set_facecolor("white")

    fig.tight_layout()
    return _save_fig(fig)


def generate_stacked_bar_chart(spec):
    if not spec:
        return None

    title = spec.get("title") or ""
    unit = spec.get("unit") or ""
    series = spec.get("series") or []
    labels, series_values = _multi_series_bar_spec(spec)
    if labels is None:
        logger.warning("Malformed stacked_bar_chart spec, skipping: %s", spec)
        return None

    palette = [BLUE, GREEN, AMBER, PURPLE, TEAL, RED]
    x = np.arange(len(labels))
    bottoms = np.zeros(len(labels))

    fig, ax = plt.subplots(figsize=(6, 3.8), dpi=140)
    for i, (s, values) in enumerate(zip(series, series_values)):
        values = np.array(values)
        ax.bar(x, values, bottom=bottoms, width=0.6, color=palette[i % len(palette)], label=s.get("name") or f"Series {i + 1}")
        bottoms += values

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    if unit:
        ax.set_ylabel(unit, fontsize=9, color="#666666")
    ax.set_title(_wrap_title(title, width=42), fontsize=14, fontweight="bold", loc="left", color="#1a1a1a")
    ax.legend(loc="best", fontsize=8.5, frameon=False)

    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="x", labelsize=9, colors="#444444")
    ax.tick_params(axis="y", labelsize=8, colors="#666666")
    if max(len(str(l)) for l in labels) > 7 or len(labels) > 4:
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    fig.patch.set_facecolor("white")

    fig.tight_layout()
    return _save_fig(fig)


def generate_waterfall_chart(spec):
    if not spec:
        return None

    title = spec.get("title") or ""
    labels = spec.get("labels") or []
    values = spec.get("values") or []
    unit = spec.get("unit") or ""

    if not labels or not values or len(labels) != len(values) or len(labels) < 2:
        logger.warning("Malformed waterfall_chart spec, skipping: %s", spec)
        return None
    try:
        values = [float(v) for v in values]
    except (TypeError, ValueError):
        logger.warning("Non-numeric values in waterfall_chart spec, skipping: %s", spec)
        return None

    all_labels = list(labels) + ["Total"]
    running = np.cumsum(values)
    total = float(running[-1])
    bar_starts = list(running - values) + [0.0]
    bar_values = list(values) + [total]
    colors = [(GREEN if v >= 0 else RED) for v in values] + [BLUE]

    x = np.arange(len(all_labels))
    fig, ax = plt.subplots(figsize=(6, 3.8), dpi=140)
    ax.bar(x, bar_values, bottom=bar_starts, width=0.6, color=colors)

    for i, (v, s) in enumerate(zip(bar_values, bar_starts)):
        y = s + v
        va = "bottom" if v >= 0 else "top"
        text = f"{v:+.1f}{unit}" if i < len(values) else f"{v:.1f}{unit}"
        ax.text(i, y, text, ha="center", va=va, fontsize=8.5, color="#1a1a1a")

    ax.set_xticks(x)
    ax.set_xticklabels(all_labels)
    ax.axhline(0, color="#999999", linewidth=0.8)
    ax.set_title(_wrap_title(title, width=42), fontsize=14, fontweight="bold", loc="left", color="#1a1a1a")

    tops = [s + v for s, v in zip(bar_starts, bar_values)]
    bottoms = [min(s, s + v) for s, v in zip(bar_starts, bar_values)]
    span = max(tops) - min(bottoms) or 1.0
    ax.set_ylim(min(bottoms) - span * 0.08, max(tops) + span * 0.12)

    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="x", labelsize=9, colors="#444444")
    ax.tick_params(axis="y", labelsize=8, colors="#666666")
    if max(len(str(l)) for l in all_labels) > 7 or len(all_labels) > 4:
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    fig.patch.set_facecolor("white")

    fig.tight_layout()
    return _save_fig(fig)


def generate_slope_chart(spec):
    if not spec:
        return None

    title = spec.get("title") or ""
    labels = spec.get("labels") or []
    start_values = spec.get("start_values") or []
    end_values = spec.get("end_values") or []
    start_label = spec.get("start_label") or "Before"
    end_label = spec.get("end_label") or "After"
    unit = spec.get("unit") or ""

    if not labels or len(labels) != len(start_values) or len(labels) != len(end_values):
        logger.warning("Malformed slope_chart spec, skipping: %s", spec)
        return None
    try:
        start_values = [float(v) for v in start_values]
        end_values = [float(v) for v in end_values]
    except (TypeError, ValueError):
        logger.warning("Non-numeric values in slope_chart spec, skipping: %s", spec)
        return None

    fig, ax = plt.subplots(figsize=(5.5, max(3, 0.6 * len(labels) + 1.5)), dpi=140)
    for lbl, s, e in zip(labels, start_values, end_values):
        color = GREEN if e >= s else RED
        ax.plot([0, 1], [s, e], "o-", color=color, linewidth=1.8, markersize=5)
        ax.text(-0.03, s, f"{lbl}  {s:.1f}{unit}", ha="right", va="center", fontsize=8.5, color="#333333")
        ax.text(1.03, e, f"{e:.1f}{unit}", ha="left", va="center", fontsize=8.5, color="#333333")

    ax.set_xlim(-0.6, 1.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([start_label, end_label], fontsize=10, color="#444444")
    ax.set_yticks([])
    ax.set_title(_wrap_title(title, width=42), fontsize=14, fontweight="bold", loc="left", color="#1a1a1a")

    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#cccccc")
    fig.patch.set_facecolor("white")

    fig.tight_layout()
    return _save_fig(fig)


def generate_bullet_chart(spec):
    if not spec:
        return None

    title = spec.get("title") or ""
    unit = spec.get("unit") or ""
    ranges = spec.get("ranges") or []

    if spec.get("value") is None or spec.get("target") is None or len(ranges) != 3:
        logger.warning("Malformed bullet_chart spec, skipping: %s", spec)
        return None
    try:
        value = float(spec["value"])
        target = float(spec["target"])
        ranges = sorted(float(r) for r in ranges)
    except (TypeError, ValueError):
        logger.warning("Non-numeric values in bullet_chart spec, skipping: %s", spec)
        return None

    fig, ax = plt.subplots(figsize=(6, 1.8), dpi=140)
    band_colors = ["#e5e7eb", "#d1d5db", "#9ca3af"]
    prev = 0.0
    for edge, color in zip(ranges, band_colors):
        ax.barh(0, edge - prev, left=prev, height=0.6, color=color)
        prev = edge

    ax.barh(0, value, left=0, height=0.25, color=BLUE)
    ax.axvline(target, color="#1a1a1a", linewidth=2.5, ymin=0.15, ymax=0.85)

    ax.set_yticks([])
    ax.set_ylim(-0.4, 0.4)
    ax.set_xlim(0, max(ranges[-1], value, target) * 1.05)
    ax.text(value, 0.15, f"{value:.1f}{unit}", ha="center", va="bottom", fontsize=9, color="#1a1a1a")
    ax.set_title(_wrap_title(title, width=50), fontsize=13, fontweight="bold", loc="left", color="#1a1a1a")

    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="x", labelsize=8, colors="#666666")
    fig.patch.set_facecolor("white")

    fig.tight_layout()
    return _save_fig(fig)


def generate_donut_chart(spec):
    if not spec:
        return None

    title = spec.get("title") or ""
    labels = spec.get("labels") or []
    values = spec.get("values") or []

    if not labels or not values or len(labels) != len(values) or len(labels) < 2:
        logger.warning("Malformed donut_chart spec, skipping: %s", spec)
        return None
    try:
        values = [abs(float(v)) for v in values]
    except (TypeError, ValueError):
        logger.warning("Non-numeric values in donut_chart spec, skipping: %s", spec)
        return None
    if sum(values) <= 0:
        logger.warning("Donut chart values sum to zero, skipping: %s", spec)
        return None

    colors = [PIE_PALETTE[i % len(PIE_PALETTE)] for i in range(len(labels))]

    fig, ax = plt.subplots(figsize=(6, 4), dpi=140)
    ax.pie(
        values,
        labels=labels,
        colors=colors,
        autopct="%1.0f%%",
        startangle=90,
        pctdistance=0.82,
        textprops={"fontsize": 9.5, "color": "#1a1a1a"},
        wedgeprops={"linewidth": 1.5, "edgecolor": "white", "width": 0.42},
    )
    ax.set_title(_wrap_title(title, width=42), fontsize=14, fontweight="bold", loc="left", color="#1a1a1a")
    fig.patch.set_facecolor("white")

    fig.tight_layout()
    return _save_fig(fig)


def _treemap_rects(items, x, y, w, h, horizontal=True):
    """Simple recursive binary-split treemap: splits the (value, label) list into two
    roughly equal-sum halves and alternates split axis each level, producing a 2D grid
    of rectangles sized proportional to value without needing a real hierarchy."""
    if len(items) == 1:
        return [(x, y, w, h, items[0])]

    total = sum(v for v, _ in items)
    running = 0.0
    split = 1
    for i, (v, _) in enumerate(items):
        running += v
        if running >= total / 2:
            split = i + 1
            break
    split = max(1, min(split, len(items) - 1))
    left, right = items[:split], items[split:]
    frac = (sum(v for v, _ in left) / total) if total else 0.5

    if horizontal:
        w_left = w * frac
        return (
            _treemap_rects(left, x, y, w_left, h, horizontal=False)
            + _treemap_rects(right, x + w_left, y, w - w_left, h, horizontal=False)
        )
    h_left = h * frac
    return (
        _treemap_rects(left, x, y, w, h_left, horizontal=True)
        + _treemap_rects(right, x, y + h_left, w, h - h_left, horizontal=True)
    )


def generate_treemap_chart(spec):
    if not spec:
        return None

    title = spec.get("title") or ""
    labels = spec.get("labels") or []
    values = spec.get("values") or []

    if not labels or not values or len(labels) != len(values) or len(labels) < 2:
        logger.warning("Malformed treemap_chart spec, skipping: %s", spec)
        return None
    try:
        values = [abs(float(v)) for v in values]
    except (TypeError, ValueError):
        logger.warning("Non-numeric values in treemap_chart spec, skipping: %s", spec)
        return None
    total = sum(values)
    if total <= 0:
        logger.warning("Treemap values sum to zero, skipping: %s", spec)
        return None

    paired = sorted(zip(values, labels), key=lambda p: -p[0])
    rects = _treemap_rects(paired, 0, 0, 1, 1, horizontal=True)
    colors = [PIE_PALETTE[i % len(PIE_PALETTE)] for i in range(len(paired))]
    color_by_label = {lbl: colors[i] for i, (_, lbl) in enumerate(paired)}

    fig, ax = plt.subplots(figsize=(6, 4.2), dpi=140)
    for rx, ry, rw, rh, (v, lbl) in rects:
        ax.add_patch(Rectangle((rx, ry), rw, rh, facecolor=color_by_label[lbl], edgecolor="white", linewidth=1.5))
        if rw > 0.08 and rh > 0.06:
            ax.text(rx + rw / 2, ry + rh / 2, f"{lbl}\n{v / total * 100:.0f}%", ha="center", va="center", fontsize=8.5, color="white", fontweight="bold")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.invert_yaxis()
    ax.axis("off")
    ax.set_title(_wrap_title(title, width=42), fontsize=14, fontweight="bold", loc="left", color="#1a1a1a")
    fig.patch.set_facecolor("white")

    fig.tight_layout()
    return _save_fig(fig)


def _grouped_values_spec(spec, min_values=4):
    groups = spec.get("groups") or []
    if not groups or any(len(g.get("values") or []) < min_values for g in groups):
        return None, None
    try:
        data = [[float(v) for v in g["values"]] for g in groups]
    except (TypeError, ValueError, KeyError):
        return None, None
    names = [g.get("name") or f"Group {i + 1}" for i, g in enumerate(groups)]
    return names, data


def generate_box_plot(spec):
    if not spec:
        return None

    title = spec.get("title") or ""
    unit = spec.get("unit") or ""
    names, data = _grouped_values_spec(spec)
    if names is None:
        logger.warning("Malformed box_plot spec, skipping: %s", spec)
        return None

    fig, ax = plt.subplots(figsize=(6, 3.8), dpi=140)
    bp = ax.boxplot(data, tick_labels=names, patch_artist=True, medianprops={"color": "#1a1a1a"})
    for patch in bp["boxes"]:
        patch.set_facecolor(BOX_FILL)
        patch.set_edgecolor(BLUE)

    if unit:
        ax.set_ylabel(unit, fontsize=9, color="#666666")
    ax.set_title(_wrap_title(title, width=42), fontsize=14, fontweight="bold", loc="left", color="#1a1a1a")

    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="x", labelsize=9, colors="#444444")
    ax.tick_params(axis="y", labelsize=8, colors="#666666")
    if len(names) > 4:
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    fig.patch.set_facecolor("white")

    fig.tight_layout()
    return _save_fig(fig)


def generate_violin_plot(spec):
    if not spec:
        return None

    title = spec.get("title") or ""
    unit = spec.get("unit") or ""
    names, data = _grouped_values_spec(spec)
    if names is None:
        logger.warning("Malformed violin_plot spec, skipping: %s", spec)
        return None

    fig, ax = plt.subplots(figsize=(6, 3.8), dpi=140)
    parts = ax.violinplot(data, showmedians=True)
    for body in parts["bodies"]:
        body.set_facecolor(BLUE)
        body.set_edgecolor(BLUE)
        body.set_alpha(0.55)
    for key in ("cmedians", "cmins", "cmaxes", "cbars"):
        if key in parts:
            parts[key].set_color(GRAY)

    ax.set_xticks(np.arange(1, len(names) + 1))
    ax.set_xticklabels(names)
    if unit:
        ax.set_ylabel(unit, fontsize=9, color="#666666")
    ax.set_title(_wrap_title(title, width=42), fontsize=14, fontweight="bold", loc="left", color="#1a1a1a")

    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="x", labelsize=9, colors="#444444")
    ax.tick_params(axis="y", labelsize=8, colors="#666666")
    if len(names) > 4:
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    fig.patch.set_facecolor("white")

    fig.tight_layout()
    return _save_fig(fig)


def generate_scatter_chart(spec):
    if not spec:
        return None

    title = spec.get("title") or ""
    x_label = spec.get("x_label") or ""
    y_label = spec.get("y_label") or ""
    x_values = spec.get("x_values") or []
    y_values = spec.get("y_values") or []

    if len(x_values) < 3 or len(x_values) != len(y_values):
        logger.warning("Malformed scatter_chart spec, skipping: %s", spec)
        return None
    try:
        x_values = [float(v) for v in x_values]
        y_values = [float(v) for v in y_values]
    except (TypeError, ValueError):
        logger.warning("Non-numeric values in scatter_chart spec, skipping: %s", spec)
        return None

    fig, ax = plt.subplots(figsize=(6, 3.8), dpi=140)
    ax.scatter(x_values, y_values, color=BLUE, s=45, alpha=0.75, edgecolor="white", linewidth=0.6)

    if x_label:
        ax.set_xlabel(x_label, fontsize=9, color="#666666")
    if y_label:
        ax.set_ylabel(y_label, fontsize=9, color="#666666")
    ax.set_title(_wrap_title(title, width=42), fontsize=14, fontweight="bold", loc="left", color="#1a1a1a")

    ax.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="x", labelsize=8, colors="#666666")
    ax.tick_params(axis="y", labelsize=8, colors="#666666")
    fig.patch.set_facecolor("white")

    fig.tight_layout()
    return _save_fig(fig)


def generate_bubble_chart(spec):
    if not spec:
        return None

    title = spec.get("title") or ""
    x_label = spec.get("x_label") or ""
    y_label = spec.get("y_label") or ""
    x_values = spec.get("x_values") or []
    y_values = spec.get("y_values") or []
    sizes = spec.get("sizes") or []
    labels = spec.get("labels") or []

    if len(x_values) < 2 or len(x_values) != len(y_values) or len(x_values) != len(sizes):
        logger.warning("Malformed bubble_chart spec, skipping: %s", spec)
        return None
    try:
        x_values = [float(v) for v in x_values]
        y_values = [float(v) for v in y_values]
        sizes = [abs(float(v)) for v in sizes]
    except (TypeError, ValueError):
        logger.warning("Non-numeric values in bubble_chart spec, skipping: %s", spec)
        return None

    max_size = max(sizes) or 1.0
    scaled = [80 + 900 * (s / max_size) for s in sizes]
    colors = [PIE_PALETTE[i % len(PIE_PALETTE)] for i in range(len(x_values))]

    fig, ax = plt.subplots(figsize=(6, 4), dpi=140)
    ax.scatter(x_values, y_values, s=scaled, color=colors, alpha=0.6, edgecolor="white", linewidth=1)

    if labels and len(labels) == len(x_values):
        for lx, ly, lbl in zip(x_values, y_values, labels):
            ax.annotate(lbl, (lx, ly), fontsize=8, ha="center", va="center", color="#1a1a1a")

    if x_label:
        ax.set_xlabel(x_label, fontsize=9, color="#666666")
    if y_label:
        ax.set_ylabel(y_label, fontsize=9, color="#666666")
    ax.set_title(_wrap_title(title, width=42), fontsize=14, fontweight="bold", loc="left", color="#1a1a1a")

    ax.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="x", labelsize=8, colors="#666666")
    ax.tick_params(axis="y", labelsize=8, colors="#666666")
    fig.patch.set_facecolor("white")

    fig.tight_layout()
    return _save_fig(fig)


def generate_correlation_matrix_chart(spec):
    if not spec:
        return None

    title = spec.get("title") or ""
    labels = spec.get("labels") or []
    matrix = spec.get("matrix") or []

    n = len(labels)
    if n < 2 or len(matrix) != n or any(len(row) != n for row in matrix):
        logger.warning("Malformed correlation_matrix_chart spec, skipping: %s", spec)
        return None
    try:
        matrix = np.array([[float(v) for v in row] for row in matrix])
    except (TypeError, ValueError):
        logger.warning("Non-numeric values in correlation_matrix_chart spec, skipping: %s", spec)
        return None

    fig, ax = plt.subplots(figsize=(5.5, 5), dpi=140)
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=-1, vmax=1)

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8.5)
    ax.set_yticklabels(labels, fontsize=8.5)

    for i in range(n):
        for j in range(n):
            v = matrix[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color="white" if abs(v) > 0.55 else "#1a1a1a")

    ax.set_title(_wrap_title(title, width=42), fontsize=14, fontweight="bold", loc="left", color="#1a1a1a")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.patch.set_facecolor("white")

    fig.tight_layout()
    return _save_fig(fig)


def generate_regression_chart(spec):
    if not spec:
        return None

    title = spec.get("title") or ""
    x_label = spec.get("x_label") or ""
    y_label = spec.get("y_label") or ""
    x_values = spec.get("x_values") or []
    y_values = spec.get("y_values") or []

    if len(x_values) < 4 or len(x_values) != len(y_values):
        logger.warning("Malformed regression_chart spec, skipping: %s", spec)
        return None
    try:
        x_arr = np.array([float(v) for v in x_values])
        y_arr = np.array([float(v) for v in y_values])
    except (TypeError, ValueError):
        logger.warning("Non-numeric values in regression_chart spec, skipping: %s", spec)
        return None

    fig, ax = plt.subplots(figsize=(6, 3.8), dpi=140)
    ax.scatter(x_arr, y_arr, color=BLUE, s=40, alpha=0.7, edgecolor="white", linewidth=0.5)

    try:
        slope, intercept = np.polyfit(x_arr, y_arr, 1)
        x_smooth = np.linspace(x_arr.min(), x_arr.max(), 100)
        y_smooth = slope * x_smooth + intercept
        r = np.corrcoef(x_arr, y_arr)[0, 1]
        ax.plot(x_smooth, y_smooth, "--", color=AMBER, linewidth=1.8, label=f"r = {r:.2f}")
        ax.legend(loc="best", fontsize=9, frameon=False)
    except Exception as exc:
        logger.warning("Regression fit failed: %s", exc)

    if x_label:
        ax.set_xlabel(x_label, fontsize=9, color="#666666")
    if y_label:
        ax.set_ylabel(y_label, fontsize=9, color="#666666")
    ax.set_title(_wrap_title(title, width=42), fontsize=14, fontweight="bold", loc="left", color="#1a1a1a")

    ax.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="x", labelsize=8, colors="#666666")
    ax.tick_params(axis="y", labelsize=8, colors="#666666")
    fig.patch.set_facecolor("white")

    fig.tight_layout()
    return _save_fig(fig)


def generate_term_structure_chart(spec):
    if not spec:
        return None

    title = spec.get("title") or ""
    labels = spec.get("labels") or []
    values = spec.get("values") or []
    compare_values = spec.get("compare_values") or None
    compare_label = spec.get("compare_label") or "Prior period"
    unit = spec.get("unit") or ""

    if not labels or not values or len(labels) != len(values) or len(labels) < 2:
        logger.warning("Malformed term_structure_chart spec, skipping: %s", spec)
        return None
    try:
        values = [float(v) for v in values]
        if compare_values is not None:
            compare_values = [float(v) for v in compare_values] if len(compare_values) == len(labels) else None
    except (TypeError, ValueError):
        logger.warning("Non-numeric values in term_structure_chart spec, skipping: %s", spec)
        return None

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6, 3.8), dpi=140)
    if compare_values:
        ax.plot(x, compare_values, "o--", color=GRAY, linewidth=1.4, markersize=5, label=compare_label)
    ax.plot(x, values, "o-", color=BLUE, linewidth=2, markersize=6, label="Current")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    if unit:
        ax.set_ylabel(unit, fontsize=9, color="#666666")
    ax.set_title(_wrap_title(title, width=42), fontsize=14, fontweight="bold", loc="left", color="#1a1a1a")
    if compare_values:
        ax.legend(loc="best", fontsize=8.5, frameon=False)

    ax.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="x", labelsize=9, colors="#444444")
    ax.tick_params(axis="y", labelsize=8, colors="#666666")
    fig.patch.set_facecolor("white")

    fig.tight_layout()
    return _save_fig(fig)


def generate_spread_chart(spec):
    if not spec:
        return None

    title = spec.get("title") or ""
    labels = spec.get("labels") or []
    values = spec.get("values") or []
    unit = spec.get("unit") or ""

    if not labels or not values or len(labels) != len(values) or len(labels) < 3:
        logger.warning("Malformed spread_chart spec, skipping: %s", spec)
        return None
    try:
        values = [float(v) for v in values]
    except (TypeError, ValueError):
        logger.warning("Non-numeric values in spread_chart spec, skipping: %s", spec)
        return None

    x = np.arange(len(values))
    colors = [GREEN if v >= 0 else RED for v in values]

    fig, ax = plt.subplots(figsize=(6, 3.5), dpi=140)
    ax.fill_between(x, values, 0, color=BLUE, alpha=0.12)
    ax.plot(x, values, color=BLUE, linewidth=1.8)
    ax.scatter(x, values, color=colors, s=18, zorder=3)
    ax.axhline(0, color="#999999", linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    if unit:
        ax.set_ylabel(unit, fontsize=9, color="#666666")
    ax.set_title(_wrap_title(title, width=42), fontsize=14, fontweight="bold", loc="left", color="#1a1a1a")

    ax.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="x", labelsize=8, colors="#666666")
    ax.tick_params(axis="y", labelsize=8, colors="#666666")
    if len(labels) > 5:
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    fig.patch.set_facecolor("white")

    fig.tight_layout()
    return _save_fig(fig)


def generate_zscore_chart(spec):
    if not spec:
        return None

    title = spec.get("title") or ""
    labels = spec.get("labels") or []
    values = spec.get("values") or []
    unit = spec.get("unit") or ""

    if not labels or not values or len(labels) != len(values) or len(labels) < 5:
        logger.warning("Malformed zscore_chart spec, skipping: %s", spec)
        return None
    try:
        values = np.array([float(v) for v in values])
    except (TypeError, ValueError):
        logger.warning("Non-numeric values in zscore_chart spec, skipping: %s", spec)
        return None

    mean, std = values.mean(), values.std()
    if std == 0:
        logger.warning("Zero variance in zscore_chart spec, skipping: %s", spec)
        return None
    z = (values - mean) / std
    x = np.arange(len(values))
    colors = [GREEN if v >= 0 else RED for v in z]

    fig, ax = plt.subplots(figsize=(6, 3.5), dpi=140)
    ax.bar(x, z, color=colors, width=0.6)
    for level, style in ((1, "--"), (2, ":")):
        ax.axhline(level, color=GRAY, linewidth=0.8, linestyle=style)
        ax.axhline(-level, color=GRAY, linewidth=0.8, linestyle=style)
    ax.axhline(0, color="#999999", linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(f"Z-score{f' ({unit})' if unit else ''}", fontsize=9, color="#666666")
    ax.set_title(_wrap_title(title, width=42), fontsize=14, fontweight="bold", loc="left", color="#1a1a1a")

    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="x", labelsize=8, colors="#666666")
    ax.tick_params(axis="y", labelsize=8, colors="#666666")
    if len(labels) > 5:
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    fig.patch.set_facecolor("white")

    fig.tight_layout()
    return _save_fig(fig)


def generate_cumulative_flow_chart(spec):
    if not spec:
        return None

    title = spec.get("title") or ""
    labels = spec.get("labels") or []
    values = spec.get("values") or []
    unit = spec.get("unit") or ""

    if not labels or not values or len(labels) != len(values) or len(labels) < 2:
        logger.warning("Malformed cumulative_flow_chart spec, skipping: %s", spec)
        return None
    try:
        values = [float(v) for v in values]
    except (TypeError, ValueError):
        logger.warning("Non-numeric values in cumulative_flow_chart spec, skipping: %s", spec)
        return None

    cumulative = np.cumsum(values)
    x = np.arange(len(labels))
    color = GREEN if cumulative[-1] >= 0 else RED

    fig, ax = plt.subplots(figsize=(6, 3.5), dpi=140)
    ax.fill_between(x, cumulative, 0, color=color, alpha=0.15)
    ax.plot(x, cumulative, color=color, linewidth=2)
    ax.axhline(0, color="#999999", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(f"Cumulative{f' ({unit})' if unit else ''}", fontsize=9, color="#666666")
    ax.set_title(_wrap_title(title, width=42), fontsize=14, fontweight="bold", loc="left", color="#1a1a1a")
    ax.text(
        0.99, 0.05, f"Net: {cumulative[-1]:+.1f}{unit}", transform=ax.transAxes,
        ha="right", va="bottom", fontsize=9.5, color=color, fontweight="bold",
    )

    ax.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="x", labelsize=8, colors="#666666")
    ax.tick_params(axis="y", labelsize=8, colors="#666666")
    if len(labels) > 5:
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
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
    if visual_type == "candlestick_chart":
        return generate_candlestick_chart(result.get("ticker"), label=label)
    if visual_type == "renko_chart":
        return generate_renko_chart(result.get("ticker"), label=label)
    if visual_type == "pnf_chart":
        return generate_pnf_chart(result.get("ticker"), label=label)
    if visual_type == "real_world_image":
        return fetch_wikipedia_image(result.get("image_query"), label=label)
    if visual_type == "ohlc_chart":
        return generate_ohlc_chart(result.get("ticker"), label=label)
    if visual_type == "heikin_ashi_chart":
        return generate_heikin_ashi_chart(result.get("ticker"), label=label)
    if visual_type == "kagi_chart":
        return generate_kagi_chart(result.get("ticker"), label=label)
    if visual_type == "area_chart":
        return generate_area_chart(result.get("ticker"), label=label)
    if visual_type == "volume_chart":
        return generate_volume_chart(result.get("ticker"), label=label)
    if visual_type == "volume_profile_chart":
        return generate_volume_profile_chart(result.get("ticker"), label=label)
    if visual_type == "yield_curve_chart":
        return generate_yield_curve_chart(label=label)
    if visual_type == "seasonality_chart":
        return generate_seasonality_chart(result.get("ticker"), label=label)
    if visual_type == "dumbbell_chart":
        return generate_dumbbell_chart(result.get("dumbbell_chart"))
    if visual_type == "grouped_bar_chart":
        return generate_grouped_bar_chart(result.get("grouped_bar_chart"))
    if visual_type == "stacked_bar_chart":
        return generate_stacked_bar_chart(result.get("stacked_bar_chart"))
    if visual_type == "waterfall_chart":
        return generate_waterfall_chart(result.get("waterfall_chart"))
    if visual_type == "slope_chart":
        return generate_slope_chart(result.get("slope_chart"))
    if visual_type == "bullet_chart":
        return generate_bullet_chart(result.get("bullet_chart"))
    if visual_type == "donut_chart":
        return generate_donut_chart(result.get("donut_chart"))
    if visual_type == "treemap_chart":
        return generate_treemap_chart(result.get("treemap_chart"))
    if visual_type == "box_plot":
        return generate_box_plot(result.get("box_plot"))
    if visual_type == "violin_plot":
        return generate_violin_plot(result.get("violin_plot"))
    if visual_type == "scatter_chart":
        return generate_scatter_chart(result.get("scatter_chart"))
    if visual_type == "bubble_chart":
        return generate_bubble_chart(result.get("bubble_chart"))
    if visual_type == "correlation_matrix_chart":
        return generate_correlation_matrix_chart(result.get("correlation_matrix_chart"))
    if visual_type == "regression_chart":
        return generate_regression_chart(result.get("regression_chart"))
    if visual_type == "term_structure_chart":
        return generate_term_structure_chart(result.get("term_structure_chart"))
    if visual_type == "spread_chart":
        return generate_spread_chart(result.get("spread_chart"))
    if visual_type == "zscore_chart":
        return generate_zscore_chart(result.get("zscore_chart"))
    if visual_type == "cumulative_flow_chart":
        return generate_cumulative_flow_chart(result.get("cumulative_flow_chart"))
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
