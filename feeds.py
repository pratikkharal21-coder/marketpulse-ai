import logging
import time
from datetime import datetime, timezone

import feedparser

logger = logging.getLogger("marketpulse.feeds")

FEEDS = {
    "markets": [
        "https://finance.yahoo.com/news/rssindex",
        "https://feeds.content.dowjones.io/public/rss/mw_topstories",
        "https://www.investing.com/rss/news.rss",
    ],
    "macro": [
        "https://www.cnbc.com/id/20910258/device/rss/rss.html",
        "https://www.fxstreet.com/rss/news",
    ],
    "crypto": [
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "https://cointelegraph.com/rss",
    ],
    "tech_ai": [
        "https://techcrunch.com/category/artificial-intelligence/feed/",
        "https://www.theverge.com/rss/index.xml",
    ],
}


def _entry_timestamp(entry):
    for key in ("published_parsed", "updated_parsed"):
        value = entry.get(key)
        if value:
            return datetime.fromtimestamp(time.mktime(value), tz=timezone.utc)
    return None


def fetch_recent_items(lookback_hours):
    cutoff = datetime.now(timezone.utc).timestamp() - lookback_hours * 3600
    items = []
    for category, urls in FEEDS.items():
        for url in urls:
            try:
                parsed = feedparser.parse(url)
            except Exception as exc:
                logger.warning("Failed to fetch feed %s: %s", url, exc)
                continue

            if parsed.bozo and not parsed.entries:
                logger.warning("Feed %s returned no usable entries (bozo): %s", url, parsed.get("bozo_exception"))
                continue

            for entry in parsed.entries:
                published = _entry_timestamp(entry)
                if published and published.timestamp() < cutoff:
                    continue

                link = entry.get("link", "")
                title = entry.get("title", "").strip()
                if not title or not link:
                    continue

                items.append(
                    {
                        "title": title,
                        "summary": entry.get("summary", "").strip(),
                        "link": link,
                        "source": category,
                        "published": published.isoformat() if published else None,
                    }
                )
    logger.info("Fetched %d raw items across %d feeds", len(items), sum(len(u) for u in FEEDS.values()))
    return items
