import logging
import traceback
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
import verify

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
        logger.info(
            "No stories survived triage (%d raw items, %d unseen). Sending empty-digest notice "
            "and exiting -- this is expected on a genuinely quiet news window, not necessarily a bug.",
            len(raw_items), len(new_items),
        )
        report_html, inline_images = report.render([], 0, [])
        mailer.send(f"MarketPulse AI — no high-impact stories ({_now_str()})", report_html, inline_images)
        state.save(st)
        return

    used_hooks = []
    # Seeded with the last few posts' visual types across PRIOR runs too, not just this run's
    # batch, so "don't repeat the same visual type back-to-back" holds across scheduled runs.
    used_visuals = state.get_recent_visuals(st)

    # Deep dives get first pick of the truly top-ranked stories; short threads then draw from
    # the remaining candidates so the same story never covers both a quick take AND a deep dive
    # in the same digest. Both generators backfill from the full candidate pool on their own
    # (see generate.py/longform.py) rather than being capped to a fixed top-N slice, so one
    # blocked or failed candidate no longer silently shrinks the day's output.
    deep_dives, deep_dive_links = longform.generate_top_longform(survivors, used_hooks, slot_framing, used_visuals)
    threads, thread_links = generate.generate_short_threads(
        survivors, used_hooks, slot_framing, used_visuals, exclude_links=deep_dive_links
    )

    published_links = thread_links | deep_dive_links
    logger.info(
        "Output summary: %d triage survivor(s) -> %d published (%d thread(s) + %d deep dive(s)). "
        "%d candidate(s) unused this run (available for a future run if still within the "
        "lookback window).",
        len(survivors), len(published_links), len(threads), len(deep_dives),
        len(survivors) - len(published_links),
    )
    if len(threads) < 3:
        logger.info(
            "Thread count below the usual 3-5 target -- check the 'skipped' breakdown above in "
            "the Short threads / Deep dives log lines to tell a quiet news day (few triage "
            "survivors) from a quality-filtering day (many blocked_* / generation_error) apart."
        )

    verify.log_provenance(threads)
    verify.log_provenance(deep_dives)

    # Reorders (never adds/removes) so the most screenshot-worthy items lead the digest --
    # composite of the model's own self-reported engagement/significance/relevance scores.
    if config.ENGAGEMENT_SCORING_ENABLED:
        threads = verify.rank_by_engagement(threads)
        deep_dives = verify.rank_by_engagement(deep_dives)

    report_html, inline_images = report.render(threads, len(survivors), deep_dives)
    subject = f"MarketPulse AI — {len(threads)} threads"
    if deep_dives:
        subject += f" + {len(deep_dives)} deep dive{'s' if len(deep_dives) != 1 else ''}"
    mailer.send(f"{subject} ({_now_str()})", report_html, inline_images)

    # Only stories that actually made it into this email are marked "seen" -- a story that was
    # merely a candidate (unused because higher-ranked stories filled the quota) or that failed
    # generation / got blocked by a verification check stays eligible for a future run, instead
    # of being permanently discarded for a transient or borderline failure.
    published_stories = [s for s in survivors if s["link"] in published_links]
    st = state.mark_sent(st, published_stories)
    st = state.save_recent_visuals(st, used_visuals)
    state.save(st)

    logger.info(
        "=== MarketPulse run complete: %d thread(s) + %d deep dive(s) sent ===",
        len(threads),
        len(deep_dives),
    )


def _now_str():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _notify_failure(exc):
    """Best-effort: if the run crashes, try to at least tell the user why instead of silently
    producing nothing. If the failure IS the email send itself, this will also fail -- that's
    fine, it's a best-effort backstop, not a guarantee, and the real error is already logged."""
    try:
        mailer.send(
            f"MarketPulse AI — run failed ({_now_str()})",
            f"<p>The scheduled run crashed before it could finish:</p><pre>{type(exc).__name__}: {exc}</pre>"
            f"<p>Check marketpulse.log / the GitHub Actions run for the full traceback.</p>",
        )
    except Exception as notify_exc:
        logger.error("Also failed to send the failure notification email: %s", notify_exc)


if __name__ == "__main__":
    try:
        run()
    except Exception as exc:
        logger.error("=== MarketPulse run crashed: %s ===\n%s", exc, traceback.format_exc())
        _notify_failure(exc)
        raise
