import logging
from datetime import datetime, timezone

import config
import feeds
import generate
import longform
import mailer
import report
import state
import triage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.FileHandler(config.LOG_PATH), logging.StreamHandler()],
)
logger = logging.getLogger("marketpulse.main")


def run():
    logger.info("=== MarketPulse run starting ===")

    st = state.load()
    st = state.prune(st)

    raw_items = feeds.fetch_recent_items(config.LOOKBACK_HOURS)
    new_items = state.filter_unseen(st, raw_items)

    survivors = triage.triage(new_items)
    if not survivors:
        logger.info("No stories survived triage. Sending empty-digest notice and exiting.")
        report_html, inline_images = report.render([], 0, [])
        mailer.send(f"MarketPulse AI — no high-impact stories ({_now_str()})", report_html, inline_images)
        state.save(st)
        return

    used_hooks = []
    threads = generate.generate_short_threads(survivors, used_hooks)
    deep_dives = longform.generate_top_longform(survivors, used_hooks)

    report_html, inline_images = report.render(threads, len(survivors), deep_dives)
    subject = f"MarketPulse AI — {len(threads)} threads"
    if deep_dives:
        subject += f" + {len(deep_dives)} deep dive{'s' if len(deep_dives) != 1 else ''}"
    mailer.send(f"{subject} ({_now_str()})", report_html, inline_images)

    st = state.mark_sent(st, survivors)
    state.save(st)

    logger.info(
        "=== MarketPulse run complete: %d thread(s) + %d deep dive(s) sent ===",
        len(threads),
        len(deep_dives),
    )


def _now_str():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


if __name__ == "__main__":
    run()
