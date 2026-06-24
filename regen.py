import logging

import config
from ai_client import call_for_json

logger = logging.getLogger("marketpulse.regen")

CRITIQUE_INSTRUCTION = (
    "This draft scored low on engagement. The first tweet is the highest-leverage line. "
    "Rewrite the thread with a sharper pattern-interrupt hook; keep the factual content and "
    "neutrality intact. Do not invent new numbers."
)


def maybe_regenerate(model, system_prompt, user_content, result, max_tokens):
    """If `result`'s expected_engagement is below REGEN_THRESHOLD, make one retry call that
    critiques the weak draft and asks for a sharper rewrite. Returns whichever of the two
    scored higher on expected_engagement. At most one retry — never recurses."""
    score = result.get("expected_engagement", 10)
    if score >= config.REGEN_THRESHOLD:
        return result

    weak_thread = "\n".join(result.get("thread", []))
    retry_user_content = (
        f"{user_content}\n\n"
        f"Your previous draft (expected_engagement={score}/10):\n{weak_thread}\n\n"
        f"{CRITIQUE_INSTRUCTION}"
    )

    try:
        retry_result = call_for_json(model, system_prompt, retry_user_content, max_tokens=max_tokens)
    except Exception as exc:
        logger.warning("Regeneration retry failed, keeping original draft: %s", exc)
        return result

    if not isinstance(retry_result, dict) or not retry_result.get("thread"):
        logger.warning("Regeneration retry returned a malformed result, keeping original draft")
        return result

    retry_score = retry_result.get("expected_engagement", 0)
    if retry_score > score:
        logger.info("Regeneration improved expected_engagement %s -> %s", score, retry_score)
        return retry_result

    logger.info("Regeneration did not improve score (%s -> %s), keeping original draft", score, retry_score)
    return result
