import logging
import re
import time
from datetime import datetime, timezone

import feedparser

logger = logging.getLogger("marketpulse.feeds")

# benzinga.com/feed was dropped from FEEDS below after every single entry it returned (10/10,
# checked live) turned out to be templated SEO filler -- "<Token> Price Prediction: 2025, 2026,
# 2030" for whatever crypto token, regenerated daily with a fresh pubDate despite being
# evergreen, not-actually-new content. That's a distinct failure mode from the investing.com
# date-parsing bug above: the timestamp is genuinely fresh, so LOOKBACK_HOURS filtering can't
# catch it -- only the content itself gives it away. Kept as a title-pattern filter (not just a
# removed feed) so the same template spam gets caught if another feed starts running it, or if
# Benzinga's feed later mixes real news back in alongside the filler.
_TEMPLATE_SPAM_TITLE_RE = re.compile(r"Price Prediction:?\s*20\d\d,\s*20\d\d", re.IGNORECASE)

# investing.com's forex/commodities feeds emit pubDate as "Sep 22, 2026 21:05 GMT" -- not
# RFC 822 (no weekday, no seconds) -- which feedparser's date parser can't handle, so
# published_parsed/updated_parsed come back None. Previously that made _entry_timestamp return
# None, which the lookback_hours filter in fetch_recent_items treats as "always include"
# (`if published and ...`), so every entry from these feeds bypassed freshness filtering
# entirely -- 100% of the "fx" category, silently. This is the manual fallback for that format.
_INVESTING_COM_DATE_FORMAT = "%b %d, %Y %H:%M %Z"

FEEDS = {
    "markets": [
        "https://finance.yahoo.com/news/rssindex",
        "https://feeds.content.dowjones.io/public/rss/mw_topstories",
        "https://www.investing.com/rss/news.rss",
        "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
        "https://feeds.bbci.co.uk/news/business/rss.xml",
        "http://rss.cnn.com/rss/money_markets.rss",
        "https://seekingalpha.com/market_currents.xml",
        "https://feeds.feedburner.com/zerohedge/feed",
        "https://www.ft.com/rss/markets",
        "https://spotgamma.com/feed/",
    ],
    "macro": [
        "https://www.cnbc.com/id/20910258/device/rss/rss.html",
        "https://www.fxstreet.com/rss/news",
        "https://www.economist.com/finance-and-economics/rss.xml",
        "https://www.federalreserve.gov/feeds/press_all.xml",
        "https://www.ecb.europa.eu/rss/press.html",
        "http://rss.cnn.com/rss/money_news_economy.rss",
        "https://feeds.a.dj.com/rss/RSSWorldNews.xml",
        "https://www.bankofengland.co.uk/rss/publications",
        "https://apps.bea.gov/rss/rss.xml",
        "https://www.ons.gov.uk/releasecalendar?rss",
        "https://www.treasurydirect.gov/TA_WS/securities/announced/rss",
    ],
    "fx": [
        "https://www.investing.com/rss/forex.rss",
    ],
    "commodities": [
        "https://oilprice.com/rss/main",
        "https://www.investing.com/rss/commodities.rss",
        "https://www.eia.gov/rss/todayinenergy.xml",
        "https://www.freightos.com/feed/",
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
    # feedparser couldn't parse either field into a struct_time -- try the raw string against
    # the one non-standard format we've actually seen in production (investing.com) before
    # giving up. Anything else still falls through to None (treated as "always include" by the
    # caller), same as before.
    for key in ("published", "updated"):
        raw = entry.get(key)
        if not raw:
            continue
        try:
            return datetime.strptime(raw, _INVESTING_COM_DATE_FORMAT).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def fetch_recent_items(lookback_hours):
    cutoff = datetime.now(timezone.utc).timestamp() - lookback_hours * 3600
    items = []
    category_counts = {category: 0 for category in FEEDS}
    dead_feeds = []

    for category, urls in FEEDS.items():
        for url in urls:
            try:
                parsed = feedparser.parse(url)
            except Exception as exc:
                logger.warning("Failed to fetch feed %s: %s", url, exc)
                dead_feeds.append(url)
                continue

            if parsed.bozo and not parsed.entries:
                logger.warning("Feed %s returned no usable entries (bozo): %s", url, parsed.get("bozo_exception"))
                dead_feeds.append(url)
                continue

            feed_item_count = 0
            for entry in parsed.entries:
                published = _entry_timestamp(entry)
                if published and published.timestamp() < cutoff:
                    continue

                link = entry.get("link", "")
                title = entry.get("title", "").strip()
                if not title or not link:
                    continue
                if _TEMPLATE_SPAM_TITLE_RE.search(title):
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
                feed_item_count += 1

            category_counts[category] += feed_item_count
            if feed_item_count == 0 and parsed.entries:
                # The feed parsed fine and had entries, they just all fell outside the
                # lookback window -- normal for a quiet/low-frequency source, not a failure.
                logger.debug("Feed %s: 0 items within the %dh lookback (%d total entries)", url, lookback_hours, len(parsed.entries))
            else:
                logger.debug("Feed %s: %d item(s) within the %dh lookback", url, feed_item_count, lookback_hours)

    total_feeds = sum(len(u) for u in FEEDS.values())
    logger.info(
        "Fetched %d raw item(s) across %d feed(s) in %d categories: %s",
        len(items), total_feeds, len(FEEDS),
        ", ".join(f"{cat}={count}" for cat, count in category_counts.items()),
    )
    if dead_feeds:
        logger.warning("%d/%d feed(s) returned nothing usable this run: %s", len(dead_feeds), total_feeds, dead_feeds)
    return items
