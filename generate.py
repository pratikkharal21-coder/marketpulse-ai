import logging

import config
from ai_client import call_for_json
from chart import generate_chart
from persona import ENGAGEMENT_GUIDELINES, PERSONA, TICKER_REFERENCE, VALUE_GUIDELINES

logger = logging.getLogger("marketpulse.generate")

SYSTEM_PROMPT = (
    PERSONA
    + """

Given one news story, write a short, ready-to-post X/Twitter thread of 3-5 tweets covering it. \
The thread should:
- Open with a hook tied to the core fact — a specific number, a sharp contrast, or a direct \
question — not a flat headline restatement.
- Add one tweet of immediate market context — what it means for the relevant markets or sectors \
right now.
- Include at least one concrete number or data point from the story.
- Close with a "What to watch next:" line naming the next relevant data point, event, or catalyst, \
followed directly by a question that invites the reader to reply with their own take.

"""
    + VALUE_GUIDELINES
    + "\n\n"
    + ENGAGEMENT_GUIDELINES
    + "\n\n"
    + TICKER_REFERENCE
    + """

Number each tweet by prefixing its text with "N/TOTAL " (e.g. "1/4 ..."). Each tweet, including \
the number prefix, must be under 280 characters. Ground every claim in the specific facts of the \
story — no generic filler.

Respond with ONLY a JSON object of this shape, no prose, no markdown fences:
{"thread": ["1/4 ...", "2/4 ...", ...], "ticker": "AAPL" or null, "relevance": 0-10, \
"expected_engagement": 0-10, "market_significance": 0-10, "confidence": 0-10}"""
)


def generate_short_thread(story):
    user_content = (
        f"Story source category: {story['source']}\n"
        f"Headline: {story['title']}\n"
        f"Summary: {story['summary'][:600]}\n"
        f"Link: {story['link']}\n"
        f"Triage notes: relevance={story.get('relevance')}, impact={story.get('impact')}, "
        f"reason={story.get('triage_reason')}"
    )

    try:
        result = call_for_json(config.GENERATE_MODEL, SYSTEM_PROMPT, user_content, max_tokens=1024)
        if not isinstance(result, dict):
            raise ValueError(f"Expected a JSON object, got {type(result).__name__}")

        thread = [t for t in result.get("thread", []) if t]
        if not thread:
            return None

        ticker = result.get("ticker") or None
        chart_image = generate_chart(ticker, label=story["title"]) if ticker else None

        return {
            "thread": thread,
            "ticker": ticker,
            "chart_image": chart_image,
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


def generate_short_threads(stories):
    threads = []
    for story in stories[: config.MAX_SHORT_THREADS]:
        thread = generate_short_thread(story)
        if thread:
            threads.append(thread)
    logger.info("Generated %d short thread(s) from %d candidate stories", len(threads), len(stories))
    return threads
