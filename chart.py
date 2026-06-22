import io
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import yfinance as yf

logger = logging.getLogger("marketpulse.chart")

GREEN = "#16a34a"
RED = "#dc2626"
GRID_COLOR = "#e5e7eb"


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


def generate_chart(ticker, label=None):
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

    ax.set_title(f"{label or ticker}  {arrow} {pct_change:+.2f}%", color=color, fontsize=14, fontweight="bold", loc="left")
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
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf.read()
