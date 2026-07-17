import logging

import config
import verify
from ai_client import QuotaExhaustedError, call_for_json
from chart import resolve_visual
from persona import (
    CONTENT_QUALITY_GUIDELINES,
    ENGAGEMENT_GUIDELINES,
    HOOK_SHAPE_TAXONOMY,
    PERSONA,
    VALUE_GUIDELINES,
    VISUAL_GUIDELINES,
    VISUAL_TYPES,
)
from regen import maybe_regenerate

logger = logging.getLogger("marketpulse.generate")

# Built from persona.VISUAL_TYPES (the same single source of truth VISUAL_GUIDELINES's prose
# menu is built from) so the JSON schema below can never drift out of sync with which types are
# actually offered -- a type removed/added there is automatically removed/added here too.
_VISUAL_TYPE_ENUM_JSON = "|".join(f'"{t}"' for t in VISUAL_TYPES) + '|"none"'

SYSTEM_PROMPT = (
    PERSONA
    + """

Given one news story, write a short, ready-to-post X/Twitter thread of 3-5 tweets covering it. \
The thread should:
- Tweet 1: the single most surprising or important data point in the story, stated as a specific \
claim (not a question, not a flat headline restatement). See the hook shape taxonomy below for \
how to pick and vary the framing.
- Middle tweet(s): immediate market context and the "so what" — what this means for the relevant \
markets or sectors right now, and why (the transmission mechanism, not an opinion on what happens \
next).
- Close with a "What to watch next:" line naming 3-5 dated catalysts (the next data release, CB \
meeting, earnings date, expiry, or technical level) relevant to this story, followed directly by \
a question that invites the reader to reply with their own take.

"""
    + HOOK_SHAPE_TAXONOMY
    + "\n\n"
    + VALUE_GUIDELINES
    + "\n\n"
    + CONTENT_QUALITY_GUIDELINES
    + "\n\n"
    + ENGAGEMENT_GUIDELINES
    + "\n\n"
    + VISUAL_GUIDELINES
    + """

Number each tweet by prefixing its text with "N/TOTAL " (e.g. "1/4 ..."). Each tweet, including \
the number prefix, must be under 280 characters. Ground every claim in the specific facts of the \
story — no generic filler.

Also provide a small engagement kit for after the thread is posted:
- "seed_replies": 2-3 short follow-up replies the user could post under their own thread to keep \
the conversation going (e.g. a clarifying stat, a counterpoint, a follow-up question). Each must \
be a complete, ready-to-post reply under 280 characters.
- "quote_angle": one short line (under 200 characters) suggesting a distinct take someone could \
use to quote-post this thread later — a different angle than the thread itself, not a summary of it.

Respond with ONLY a JSON object of this shape, no prose, no markdown fences:
{"thread": ["1/4 ...", "2/4 ...", ...], \
"hook_shape": "number_led"|"contrarian"|"stakes"|"question"|"surprising_fact", \
"visual_type": """ + _VISUAL_TYPE_ENUM_JSON + """, \
"visual_confidence": 0-10 (how confidently the chosen visual_type's DATA SHAPE fits this story; a low score drops the visual even if visual_type is set), \
"ticker": "AAPL" or null (used by price_chart, candlestick_chart, renko_chart, pnf_chart, ohlc_chart, heikin_ashi_chart, kagi_chart, area_chart, volume_chart, volume_profile_chart, seasonality_chart, moving_average_chart, bollinger_bands_chart, rsi_chart, macd_chart, drawdown_chart, historical_volatility_chart, company_revenue_chart; for cot_positioning_chart, ticker MUST be exactly one of GC=F, SI=F, HG=F, CL=F, BZ=F, NG=F, EURUSD=X, GBPUSD=X, USDJPY=X, ^GSPC, ^IXIC, ^DJI, ^VIX -- any other ticker returns no chart), \
"fred_series": "CPIAUCSL" or null (used ONLY by fred_series_chart -- MUST be exactly one of the series IDs listed in the fred_series_chart guidance above; any other ID returns no chart), \
"bar_chart": {"title": "...", "labels": [...], "values": [...], "unit": "...", "orientation": "vertical"|"horizontal"} or null, \
"dumbbell_chart": {"title": "...", "labels": [...], "start_values": [...], "end_values": [...], "start_label": "...", "end_label": "...", "unit": "..."} or null, \
"grouped_bar_chart": {"title": "...", "labels": [...], "series": [{"name": "...", "values": [...]}, ...], "unit": "..."} or null, \
"stacked_bar_chart": {"title": "...", "labels": [...], "series": [{"name": "...", "values": [...]}, ...], "unit": "..."} or null, \
"waterfall_chart": {"title": "...", "labels": [...], "values": [...], "unit": "..."} or null, \
"slope_chart": {"title": "...", "labels": [...], "start_values": [...], "end_values": [...], "start_label": "...", "end_label": "...", "unit": "..."} or null, \
"bullet_chart": {"title": "...", "value": 0, "target": 0, "ranges": [0, 0, 0], "unit": "..."} or null, \
"pie_chart": {"title": "...", "labels": [...], "values": [...]} or null, \
"donut_chart": {"title": "...", "labels": [...], "values": [...]} or null, \
"treemap_chart": {"title": "...", "labels": [...], "values": [...]} or null, \
"histogram": {"title": "...", "values": [...], "unit": "..."} or null, \
"box_plot": {"title": "...", "groups": [{"name": "...", "values": [...]}, ...], "unit": "..."} or null, \
"violin_plot": {"title": "...", "groups": [{"name": "...", "values": [...]}, ...], "unit": "..."} or null, \
"term_structure_chart": {"title": "...", "labels": [...], "values": [...], "compare_values": [...], "compare_label": "...", "unit": "..."} or null, \
"spread_chart": {"title": "...", "labels": [...], "values": [...], "unit": "..."} or null, \
"zscore_chart": {"title": "...", "labels": [...], "values": [...], "unit": "..."} or null, \
"cumulative_flow_chart": {"title": "...", "labels": [...], "values": [...], "unit": "..."} or null, \
"custom_stat_visual": {"title": "...", "stats": [{"label": "...", "value": 0, "unit": "..."}, ...]} or null, \
"flowchart": {"steps": [...]} or null, \
"image_query": "..." or null, \
"seed_replies": ["...", "..."], "quote_angle": "...", \
"relevance": 0-10, "expected_engagement": 0-10, "market_significance": 0-10, "confidence": 0-10}"""
)


def generate_short_thread(story, used_hooks=None, slot_framing=None, used_visuals=None):
    used_hooks = used_hooks if used_hooks is not None else []
    used_visuals = used_visuals if used_visuals is not None else []
    hook_note = (
        f"\n\nHook shapes already used so far in this batch: {used_hooks}. Avoid reusing any "
        f"shape that already appears twice in that list; prefer an unused shape if the story "
        f"allows it."
        if used_hooks
        else ""
    )
    visual_note = (
        f"\n\nVisual types already used in this batch: {used_visuals}. "
        f"For variety, strongly prefer a visual type NOT in this list if the story reasonably supports it. "
        f"Only repeat a type if it is genuinely the only fit for this specific story."
        if used_visuals
        else ""
    )
    slot_note = f"\n\nRun context: {slot_framing}" if slot_framing else ""

    user_content = (
        f"Story source category: {story['source']}\n"
        f"Headline: {story['title']}\n"
        f"Summary: {story['summary'][:600]}\n"
        f"Link: {story['link']}\n"
        f"Triage notes: relevance={story.get('relevance')}, impact={story.get('impact')}, "
        f"reason={story.get('triage_reason')}"
        f"{hook_note}"
        f"{visual_note}"
        f"{slot_note}"
    )

    try:
        result = call_for_json(config.GENERATE_MODEL, SYSTEM_PROMPT, user_content, max_tokens=1024)
        if not isinstance(result, dict):
            raise ValueError(f"Expected a JSON object, got {type(result).__name__}")

        if result.get("thread"):
            result = maybe_regenerate(config.GENERATE_MODEL, SYSTEM_PROMPT, user_content, result, max_tokens=1024)

        thread = [t for t in result.get("thread", []) if t]
        if not thread:
            logger.warning("Story '%s': model produced no usable thread text", story["title"])
            return None, "empty_thread"

        ok, reason = verify.check_thread_completeness(thread)
        if not ok:
            logger.warning("Blocked story '%s': %s", story["title"], reason)
            return None, "blocked_incomplete_thread"

        # Grounding text must match exactly what the model was shown -- not the full (possibly
        # longer) feed summary -- otherwise a "grounded" number could really just be a number
        # that happens to appear later in the raw feed text the model never saw.
        grounding_story = {**story, "summary": story["summary"][:600]}

        ok, reason = verify.check_causal_claims(thread, grounding_story)
        if not ok:
            logger.warning("Blocked story '%s': %s", story["title"], reason)
            return None, "blocked_causal_claim"

        if config.CONTENT_ENGINE_ENABLED:
            ok, reason = verify.check_banned_filler(thread)
            if not ok:
                logger.warning("Blocked story '%s': %s", story["title"], reason)
                return None, "blocked_banned_filler"

            ok, reason = verify.check_hashtag_discipline(thread)
            if not ok:
                logger.warning("Blocked story '%s': %s", story["title"], reason)
                return None, "blocked_hashtag_discipline"

        result, visual_warnings = verify.select_visual(result, grounding_story, thread, used_visuals)

        chart_stats = {}
        chart_image = resolve_visual(result, label=story["title"], stats_out=chart_stats, source=story.get("source"))

        ok, reason = verify.verify_ticker_direction(thread, chart_stats)
        if not ok:
            logger.warning("Blocked story '%s': %s", story["title"], reason)
            return None, "blocked_direction_mismatch"

        advisory_warnings = verify.check_bare_numbers(thread) + verify.check_verb_intensity(thread, chart_stats)
        provenance = verify.build_provenance(story, chart_stats, visual_warnings, advisory_warnings)

        seed_replies = result.get("seed_replies") or []
        if not isinstance(seed_replies, list):
            seed_replies = []
        seed_replies = [r for r in seed_replies if r]

        return {
            "thread": thread,
            "hook_shape": result.get("hook_shape"),
            "visual_type": result.get("visual_type") or "none",
            "chart_image": chart_image,
            "seed_replies": seed_replies,
            "quote_angle": result.get("quote_angle") or None,
            "relevance": result.get("relevance", 0),
            "expected_engagement": result.get("expected_engagement", 0),
            "market_significance": result.get("market_significance", 0),
            "confidence": result.get("confidence", 0),
            "story_title": story["title"],
            "story_link": story["link"],
            "story_source": story["source"],
            "provenance": provenance,
        }, None
    except QuotaExhaustedError:
        return None, "quota_exhausted"
    except Exception as exc:
        logger.error("Short thread generation failed for story '%s': %s", story["title"], exc)
        return None, "generation_error"


def generate_short_threads(stories, used_hooks=None, slot_framing=None, used_visuals=None, exclude_links=None):
    """Attempts candidates in triage-ranked order and BACKFILLS from the rest of `stories` (up
    to config.MAX_STORIES_ANALYZED candidates, not just the top MAX_SHORT_THREADS) whenever one
    is skipped, blocked, or fails to generate -- so one bad candidate doesn't silently shrink
    the day's output when other viable stories were available. Stops once MAX_SHORT_THREADS
    threads have been produced, or candidates run out. Returns (threads, used_links) where
    used_links is the set of story links that ended up published, for seen-tracking and to
    keep deep dives from covering the same story twice."""
    if used_hooks is None:
        used_hooks = []
    if used_visuals is None:
        used_visuals = []
    exclude_links = exclude_links or set()

    threads = []
    used_links = set()
    reason_counts = {}
    attempted = 0

    for story in stories:
        if len(threads) >= config.MAX_SHORT_THREADS:
            break
        if attempted >= config.MAX_GENERATION_ATTEMPTS:
            logger.info(
                "Short threads: stopping backfill at MAX_GENERATION_ATTEMPTS (%d) to leave "
                "quota headroom for later runs today.", config.MAX_GENERATION_ATTEMPTS,
            )
            break
        if story["link"] in exclude_links:
            continue

        attempted += 1
        thread, reason = generate_short_thread(story, used_hooks, slot_framing, used_visuals)
        if thread:
            threads.append(thread)
            used_links.add(story["link"])
            if thread.get("hook_shape"):
                used_hooks.append(thread["hook_shape"])
            vt = thread.get("visual_type")
            if vt and vt != "none":
                used_visuals.append(vt)
        else:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            if reason == "quota_exhausted":
                logger.error(
                    "Stopping short-thread generation early: Groq daily quota exhausted, "
                    "%d remaining candidate(s) not attempted this run.",
                    len(stories) - attempted,
                )
                break

    if reason_counts:
        breakdown = ", ".join(f"{count} {reason}" for reason, count in sorted(reason_counts.items()))
        logger.info(
            "Short threads: %d/%d candidates tried -> %d published (skipped: %s)",
            attempted, len(stories), len(threads), breakdown,
        )
    else:
        logger.info("Short threads: %d/%d candidates tried -> %d published, no skips", attempted, len(stories), len(threads))
    return threads, used_links
