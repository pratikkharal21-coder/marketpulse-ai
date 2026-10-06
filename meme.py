import logging

import config
import verify
from ai_client import QuotaExhaustedError, call_for_json
from memeart import MEME_MOODS, generate_meme_image
from persona import PERSONA

logger = logging.getLogger("marketpulse.meme")

_MOOD_ENUM = "|".join(f'"{m}"' for m in MEME_MOODS)

# Deliberately NOT built from persona.PERSONA's "neutral, data-driven, professional" framing --
# this is the one place in the pipeline that's supposed to break that tone on purpose. Still
# shares the same anti-fabrication rule as everything else (see check_causal_claims /
# check_thread_completeness reuse below): the JOKE can exaggerate, the FACTS it's grounded in
# can't be invented.
SYSTEM_PROMPT = (
    "You write ONE short, sarcastic, mocking X/Twitter post reacting to a real financial "
    "markets story -- the comic-relief counterpart to this account's usual neutral analyst "
    "threads. Channel dark, self-deprecating FinTwit/WSB humor: relatable trader frustration, "
    "gallows humor about losses, deadpan mockery of absurd market behavior. The goal is a "
    "laugh and a retweet from someone who had a rough trading day, not another news brief.\n\n"
    "Rules:\n"
    "- 1-3 tweets, each under 280 characters, numbered \"N/TOTAL \" like every other thread.\n"
    "- Every joke must be grounded in a real fact from the story below -- comic exaggeration "
    "about reactions/feelings is fine (\"my portfolio\", \"my therapist\"), inventing a market "
    "FACT (a number, an event) that isn't in the story is not, same rule as everywhere else in "
    "this pipeline.\n"
    "- Mock the market/situation/collective trader psychology, never a named individual.\n"
    "- Zero hashtags. At most one emoji, only if it lands.\n\n"
    "Also pick one \"mood\" for the accompanying cartoon, from exactly one of: " + _MOOD_ENUM + ". "
    "And write a short \"caption\" (under 60 characters) for that cartoon, in the same voice "
    "(e.g. \"me checking my portfolio\" style) -- it sits ABOVE the cartoon, so it should stand "
    "alone without needing the tweets for context.\n\n"
    "Respond with ONLY JSON, no prose, no markdown fences: "
    '{"thread": ["1/2 ...", "2/2 ..."], "mood": "' + MEME_MOODS[0] + '", "caption": "...", '
    '"relevance": 0-10}'
)


def generate_meme_thread(story, slot_framing=None):
    """One extra, sarcastic/mocking take on `story` -- additive, not a replacement for the
    normal short threads/deep dives. Reuses the same structural/grounding hard-blocks as
    generate.py (numbering completeness, causal-claim grounding, hashtag discipline) since
    those check FACTS, not tone -- the only thing skipped is persona.py's neutral-tone
    CONTENT_QUALITY_GUIDELINES, which would actively fight the sarcastic voice this is for.
    Returns (item_dict_or_None, reason_or_None)."""
    slot_note = f"\n\nRun context: {slot_framing}" if slot_framing else ""
    user_content = (
        f"Story source category: {story['source']}\n"
        f"Headline: {story['title']}\n"
        f"Summary: {story['summary'][:600]}\n"
        f"Link: {story['link']}"
        f"{slot_note}"
    )

    try:
        result = call_for_json(config.GENERATE_MODEL, SYSTEM_PROMPT, user_content, max_tokens=512)
        if not isinstance(result, dict):
            raise ValueError(f"Expected a JSON object, got {type(result).__name__}")

        thread = [t for t in result.get("thread", []) if t]
        if not thread:
            logger.warning("Meme take for '%s': model produced no usable thread text", story["title"])
            return None, "empty_thread"

        ok, reason = verify.check_thread_completeness(thread)
        if not ok:
            logger.warning("Blocked meme take '%s': %s", story["title"], reason)
            return None, "blocked_incomplete_thread"

        grounding_story = {**story, "summary": story["summary"][:600]}
        ok, reason = verify.check_causal_claims(thread, grounding_story)
        if not ok:
            logger.warning("Blocked meme take '%s': %s", story["title"], reason)
            return None, "blocked_causal_claim"

        ok, reason = verify.check_hashtag_discipline(thread)
        if not ok:
            logger.warning("Blocked meme take '%s': %s", story["title"], reason)
            return None, "blocked_hashtag_discipline"

        mood = result.get("mood")
        if mood not in MEME_MOODS:
            logger.info("Meme take '%s': unrecognized/missing mood '%s', defaulting to 'confused'", story["title"], mood)
            mood = "confused"
        caption = (result.get("caption") or story["title"])[:60]

        chart_stats = {}
        chart_image = generate_meme_image(mood, caption, stats_out=chart_stats)

        return {
            "thread": thread,
            "mood": mood,
            "caption": caption,
            "chart_image": chart_image,
            "relevance": result.get("relevance", 0),
            "story_title": story["title"],
            "story_link": story["link"],
            "story_source": story["source"],
        }, None
    except QuotaExhaustedError:
        return None, "quota_exhausted"
    except Exception as exc:
        logger.error("Meme take generation failed for story '%s': %s", story["title"], exc)
        return None, "generation_error"
