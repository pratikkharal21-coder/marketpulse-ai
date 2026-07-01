import logging

import config
from ai_client import call_for_json
from chart import resolve_visual
from persona import ENGAGEMENT_GUIDELINES, HOOK_SHAPE_TAXONOMY, PERSONA, VALUE_GUIDELINES, VISUAL_GUIDELINES
from regen import maybe_regenerate

logger = logging.getLogger("marketpulse.generate")

SYSTEM_PROMPT = (
    PERSONA
    + """

Given one news story, write a short, ready-to-post X/Twitter thread of 3-5 tweets covering it. \
The thread should:
- Open with a hook tied to the core fact — not a flat headline restatement. See the hook shape \
taxonomy below for how to pick and vary it.
- Add one tweet of immediate market context — what it means for the relevant markets or sectors \
right now.
- Include at least one concrete number or data point from the story.
- Close with a "What to watch next:" line naming the next relevant data point, event, or catalyst, \
followed directly by a question that invites the reader to reply with their own take.

"""
    + HOOK_SHAPE_TAXONOMY
    + "\n\n"
    + VALUE_GUIDELINES
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
"visual_type": "price_chart"|"candlestick_chart"|"renko_chart"|"pnf_chart"|"bar_chart"|"histogram"|"pie_chart"|"trend_chart"|"flowchart"|"real_world_image"|"none", \
"ticker": "AAPL" or null, \
"bar_chart": {"title": "...", "labels": [...], "values": [...], "unit": "..."} or null, \
"histogram": {"title": "...", "values": [...], "unit": "..."} or null, \
"pie_chart": {"title": "...", "labels": [...], "values": [...]} or null, \
"trend_chart": {"title": "...", "labels": [...], "values": [...], "fit": "linear"|"cubic", "unit": "..."} or null, \
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
            return None

        chart_image = resolve_visual(result, label=story["title"])

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
        }
    except Exception as exc:
        logger.error("Short thread generation failed for story '%s': %s", story["title"], exc)
        return None


def generate_short_threads(stories, used_hooks=None, slot_framing=None, used_visuals=None):
    if used_hooks is None:
        used_hooks = []
    if used_visuals is None:
        used_visuals = []
    threads = []
    for story in stories[: config.MAX_SHORT_THREADS]:
        thread = generate_short_thread(story, used_hooks, slot_framing, used_visuals)
        if thread:
            threads.append(thread)
            if thread.get("hook_shape"):
                used_hooks.append(thread["hook_shape"])
            vt = thread.get("visual_type")
            if vt and vt != "none":
                used_visuals.append(vt)
    logger.info("Generated %d short thread(s) from %d candidate stories", len(threads), len(stories))
    return threads
