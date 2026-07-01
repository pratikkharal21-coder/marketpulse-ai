import logging

import config
from ai_client import call_for_json
from chart import resolve_visual
from persona import ENGAGEMENT_GUIDELINES, HOOK_SHAPE_TAXONOMY, PERSONA, VALUE_GUIDELINES, VISUAL_GUIDELINES
from regen import maybe_regenerate

logger = logging.getLogger("marketpulse.longform")

SYSTEM_PROMPT = (
    PERSONA
    + """

Write a deep-dive X/Twitter thread (8-10 tweets) on one major financial/market story — going far \
beyond a quick take. Structure:
1. Hook tweet — this is the single most important tweet for getting someone to open the thread —
make it count. See the hook shape taxonomy below for how to pick and vary it.
2. Context — why this matters right now, including at least one historical comparison (a past \
similar event, cycle, or precedent) to anchor the analysis.
3. Concrete data — specific numbers and recent developments, and where relevant, probabilities \
(e.g. odds implied by markets, futures pricing, or stated by officials/analysts).
4. Market scenarios — lay out at least two plausible scenarios (a base case and a risk case) and \
what each would mean for equities, bonds, currencies, commodities, or sectors as relevant.
5. Balanced insight — explicitly cover both the upside case and the key risk; never take a side on \
which will happen.
6. If the story involves politics, policy, elections, or geopolitics, discuss it ONLY through its \
financial/market impact. Do not state or imply any political opinion — stay strictly neutral and \
non-partisan.
7. Close with a "What to watch next:" line naming specific upcoming dates, data releases, or \
catalysts to monitor, followed directly by a question asking readers which scenario (from step 4) \
they find more likely, or what they're watching that wasn't covered.

Where the story supports it, structure ONE tweet in this thread as a saveable reference: a short \
numbered framework (e.g. "3 things that determine X: 1) ... 2) ... 3) ..."), a dated "what to \
watch" checklist of specific upcoming dates/events, or an if-X-then-Y scenario map (e.g. "If CPI \
prints above 3.0% Thursday: Y. If below: Z."). The reason someone would want to save this tweet \
must be implicit in the specific dates, levels, or conditions it contains — never write the \
literal words "bookmark this," "save this," or any equivalent. If the story doesn't naturally \
support a structure like this, skip it entirely; not every deep dive needs one.

"""
    + HOOK_SHAPE_TAXONOMY
    + "\n\n"
    + VALUE_GUIDELINES
    + "\n\n"
    + ENGAGEMENT_GUIDELINES
    + "\n\n"
    + VISUAL_GUIDELINES
    + """

Number each tweet by prefixing its text with "N/TOTAL " (e.g. "1/9 ..."). Each tweet, including \
the number prefix, must be under 280 characters. Ground every claim in the specific facts of the \
story — no generic filler.

Also provide a small engagement kit for after the thread is posted:
- "seed_replies": 2-3 short follow-up replies the user could post under their own thread to keep \
the conversation going (e.g. a clarifying stat, a counterpoint, a follow-up question). Each must \
be a complete, ready-to-post reply under 280 characters.
- "quote_angle": one short line (under 200 characters) suggesting a distinct take someone could \
use to quote-post this thread later — a different angle than the thread itself, not a summary of it.

Respond with ONLY a JSON object of this shape, no prose, no markdown fences:
{"thread": ["1/9 ...", "2/9 ...", ...], \
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


def generate_longform(story, used_hooks=None, slot_framing=None, used_visuals=None):
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
        result = call_for_json(config.GENERATE_MODEL, SYSTEM_PROMPT, user_content, max_tokens=2048)
        if not isinstance(result, dict):
            raise ValueError(f"Expected a JSON object, got {type(result).__name__}")

        if result.get("thread"):
            result = maybe_regenerate(config.GENERATE_MODEL, SYSTEM_PROMPT, user_content, result, max_tokens=2048)

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
        logger.error("Long-form generation failed for story '%s': %s", story["title"], exc)
        return None


def generate_top_longform(stories, used_hooks=None, slot_framing=None, used_visuals=None):
    if used_hooks is None:
        used_hooks = []
    if used_visuals is None:
        used_visuals = []
    items = []
    for story in stories[: config.MAX_LONGFORM_STORIES]:
        longform = generate_longform(story, used_hooks, slot_framing, used_visuals)
        if longform:
            items.append(longform)
            if longform.get("hook_shape"):
                used_hooks.append(longform["hook_shape"])
            vt = longform.get("visual_type")
            if vt and vt != "none":
                used_visuals.append(vt)
    logger.info("Generated %d deep-dive thread(s) from %d candidate stories", len(items), len(stories))
    return items
