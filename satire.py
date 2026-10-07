import logging
import re
from urllib.parse import urlparse

import config
import verify
from ai_client import QuotaExhaustedError, call_for_json
from satire_image import generate_satire_image

logger = logging.getLogger("marketpulse.satire")

_SOURCE_LABELS = {
    "theonion.com": "The Onion",
    "thedailymash.co.uk": "The Daily Mash",
}

# Satire outlets (The Onion especially) sometimes run a real person's actual death/tragedy in a
# straight, non-fictional voice rather than a fabricated joke premise -- confirmed in testing: a
# genuine recent death ("Dennis Hastert Dies") was picked up from FEEDS["humor"] and the model
# wrote a mocking joke about the real, named person despite the system prompt explicitly
# forbidding it. The prompt instruction alone isn't trustworthy enough to rely on for this --
# these topics are filtered out before a generation call is even made. Deliberately broad and
# conservative: skipping a borderline-serious headline costs nothing (satire.py always has
# another candidate to backfill with); letting one through costs a lot.
_SENSITIVE_TOPIC_RE = re.compile(
    r"\b(dies|dead|death|died|killed|kills|murder\w*|suicide|shooting|shot|stabbed|funeral|"
    r"mourns?|mourning|obituary|assassinat\w*|terminally ill)\b",
    re.IGNORECASE,
)

# Cheap heuristic for "this headline centers a real, specifically-named individual": a
# consecutive run of 2-3 Title-Case words (e.g. "Dennis Hastert", "Elon Musk"). Used as a
# backstop AFTER generation too -- if the same name the headline named shows up in the model's
# own output, the joke named a real person, which the system prompt forbids but can't be trusted
# to self-enforce (same incident as above).
_NAME_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b")


def _is_sensitive_topic(story):
    return bool(_SENSITIVE_TOPIC_RE.search(f"{story['title']} {story['summary'][:200]}"))


def _names_real_individual_from_headline(thread_lines, story):
    headline_names = set(_NAME_RE.findall(story["title"]))
    if not headline_names:
        return False
    full_text = " ".join(thread_lines)
    return any(name in full_text for name in headline_names)

# Deliberately NOT built from persona.PERSONA's "neutral, data-driven, professional" framing --
# same reasoning as meme.py. The input headline here is already fictional (from a satire/comedy
# publication, not a real news source), so there is no real-world fact to protect; the only
# hard rule is that the joke must never be presented as if it were real market news.
SYSTEM_PROMPT = (
    "You write ONE short, funny X/Twitter take reacting to a headline from a known satire/comedy "
    "publication (e.g. The Onion, The Daily Mash) -- the headline itself is already a joke, not "
    "real news. This is pure comic relief for a finance/markets audience, clearly separate from "
    "this account's serious market analysis.\n\n"
    "Rules:\n"
    "- 1-2 tweets, each under 280 characters, numbered \"N/TOTAL \" like every other thread.\n"
    "- Riff on the headline -- add a punchline or hot take a trader/investor would find funny. "
    "Never write it like a real market-moving news brief, and never imply the premise is real.\n"
    "- Mock the absurd premise/situation, never a named real individual.\n"
    "- Zero hashtags. At most one emoji, only if it lands.\n\n"
    "Also describe a companion illustration, in the style of a bold finance-meme poster:\n"
    "- \"image_caption\": short, punchy bold text for the image (under 60 characters) -- a "
    "punchline, not a restatement of the headline.\n"
    "- \"image_stat\": a short invented stat/number callout for the image in the same joke spirit "
    "(e.g. \"-28%\", \"$1.33M/DAY\"), consistent with the joke -- since the premise is already "
    "fictional this can be a playful invented number. Null if no number fits naturally.\n"
    "- \"image_scene\": one short plain-English description (under 150 characters) of the "
    "illustrated scene -- GENERIC invented characters only, never the real name or likeness of "
    "any actual person (no real politicians, CEOs, celebrities); if the headline references a "
    "real person, describe a generic stand-in role instead (e.g. \"a generic suited politician\", "
    "\"a generic exhausted trader\").\n\n"
    "Respond with ONLY JSON, no prose, no markdown fences: "
    '{"thread": ["1/2 ...", "2/2 ..."], "image_caption": "...", "image_stat": "..." or null, '
    '"image_scene": "..."}'
)


def _source_label(link):
    netloc = urlparse(link).netloc.removeprefix("www.")
    return _SOURCE_LABELS.get(netloc, netloc)


def generate_satire_take(story, slot_framing=None):
    """One joke take on a satire/comedy headline from FEEDS["humor"] -- additive to every other
    section, never a replacement. Reuses the same structural hard-blocks as generate.py/meme.py
    (numbering completeness, hashtag discipline) since those check structure, not tone or
    real-world grounding -- check_causal_claims is deliberately skipped here (unlike meme.py),
    since the "story" being grounded against is itself fictional. Returns (item_dict_or_None,
    reason_or_None)."""
    if _is_sensitive_topic(story):
        logger.info(
            "Satire take skipped for '%s': sensitive real-world topic (death/tragedy keyword), "
            "not safe for comedic framing regardless of source", story["title"],
        )
        return None, "skipped_sensitive_topic"

    slot_note = f"\n\nRun context: {slot_framing}" if slot_framing else ""
    user_content = (
        f"Satire source: {story['source']}\n"
        f"Headline: {story['title']}\n"
        f"Summary: {story['summary'][:400]}\n"
        f"Link: {story['link']}"
        f"{slot_note}"
    )

    try:
        result = call_for_json(config.GENERATE_MODEL, SYSTEM_PROMPT, user_content, max_tokens=512)
        if not isinstance(result, dict):
            raise ValueError(f"Expected a JSON object, got {type(result).__name__}")

        thread = [t for t in result.get("thread", []) if t]
        if not thread:
            logger.warning("Satire take for '%s': model produced no usable thread text", story["title"])
            return None, "empty_thread"

        ok, reason = verify.check_thread_completeness(thread)
        if not ok:
            logger.warning("Blocked satire take '%s': %s", story["title"], reason)
            return None, "blocked_incomplete_thread"

        ok, reason = verify.check_hashtag_discipline(thread)
        if not ok:
            logger.warning("Blocked satire take '%s': %s", story["title"], reason)
            return None, "blocked_hashtag_discipline"

        if _names_real_individual_from_headline(thread, story):
            logger.warning(
                "Blocked satire take '%s': generated thread names a real individual from the "
                "headline, despite the system prompt forbidding it", story["title"],
            )
            return None, "blocked_named_individual"

        image_caption = (result.get("image_caption") or story["title"])[:60]
        image_stat = result.get("image_stat") or None
        image_scene = result.get("image_scene") or "a generic office worker reacting to surprising news"

        chart_stats = {}
        chart_image = generate_satire_image(
            image_scene, image_caption, image_stat, headline=story["title"], stats_out=chart_stats,
        )

        return {
            "thread": thread,
            "chart_image": chart_image,
            "story_title": story["title"],
            "story_link": story["link"],
            "story_source": story["source"],
            "source_label": _source_label(story["link"]),
        }, None
    except QuotaExhaustedError:
        return None, "quota_exhausted"
    except Exception as exc:
        logger.error("Satire take generation failed for story '%s': %s", story["title"], exc)
        return None, "generation_error"


def generate_satire_digest(stories, slot_framing=None, max_items=None):
    """Attempts candidates in order (already recency-sorted by the caller) and backfills a
    little beyond max_items if an early candidate is skipped/blocked, same spirit as
    generate.py's backfill but on a much smaller scale -- this is a bonus section, not the main
    event. Returns (items, used_links)."""
    max_items = config.MAX_SATIRE_ITEMS if max_items is None else max_items
    max_attempts = max_items + 2

    items = []
    used_links = set()
    reason_counts = {}
    attempted = 0

    for story in stories:
        if len(items) >= max_items:
            break
        if attempted >= max_attempts:
            break
        attempted += 1

        item, reason = generate_satire_take(story, slot_framing)
        if item:
            items.append(item)
            used_links.add(story["link"])
        else:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            if reason == "quota_exhausted":
                logger.error("Satire digest: stopping early, Groq daily quota exhausted")
                break

    if reason_counts:
        breakdown = ", ".join(f"{count} {reason}" for reason, count in sorted(reason_counts.items()))
        logger.info(
            "Satire digest: %d/%d candidate(s) tried -> %d published (skipped: %s)",
            attempted, len(stories), len(items), breakdown,
        )
    else:
        logger.info("Satire digest: %d/%d candidate(s) tried -> %d published", attempted, len(stories), len(items))
    return items, used_links
