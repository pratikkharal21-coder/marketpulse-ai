import logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

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

LONDON_TZ = ZoneInfo("Europe/London")

# (slot name, London-local hour, London-local minute, framing note for the generator prompts)
# Schedule: midday and afternoon fire every day; close is weekdays-only (see
# .github/workflows/marketpulse.yml) -- but framing here is purely about time-of-day, so a
# manual/off-schedule run on a weekend still gets sensible "close" framing if triggered at 9pm.
RUN_SLOTS = [
    (
        "midday",
        14,
        0,
        "This is the midday digest, ahead of the US market session. Frame coverage "
        "forward-looking: set the agenda for the session ahead -- what to watch and why it "
        "matters today, not a recap of what already happened.",
    ),
    (
        "afternoon",
        18,
        0,
        "This is the afternoon digest, mid-way through the US trading session. Frame coverage "
        "reactively: focus on what is moving right now, in real time, and why.",
    ),
    (
        "close",
        21,
        0,
        "This is the US market close digest. Frame coverage as a recap: summarize what happened "
        "today and what it sets up for tomorrow's session.",
    ),
]


def get_run_slot(now=None):
    """Derive which of the 3 daily slots (midday/afternoon/close) this run corresponds to,
    based on current London-local time (auto-adjusts for BST/GMT). Workflow_dispatch carries
    no parameters, so this is inferred from the clock rather than passed in by the trigger.
    Falls back to whichever slot is nearest in time-of-day, so manual/off-schedule runs still
    get a sensible framing instead of erroring."""
    now = now or datetime.now(timezone.utc)
    london_now = now.astimezone(LONDON_TZ)
    current_minutes = london_now.hour * 60 + london_now.minute

    def cyclic_distance(a, b):
        diff = abs(a - b) % (24 * 60)
        return min(diff, 24 * 60 - diff)

    name, hour, minute, framing = min(
        RUN_SLOTS, key=lambda slot: cyclic_distance(current_minutes, slot[1] * 60 + slot[2])
    )
    return name, framing


def run():
    logger.info("=== MarketPulse run starting ===")

    slot_name, slot_framing = get_run_slot()
    logger.info("Run slot: %s", slot_name)

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
    threads = generate.generate_short_threads(survivors, used_hooks, slot_framing)
    deep_dives = longform.generate_top_longform(survivors, used_hooks, slot_framing)

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
