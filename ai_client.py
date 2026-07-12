import json
import logging
import re
import time

from groq import Groq, RateLimitError

import config

logger = logging.getLogger("marketpulse.ai")

client = Groq(api_key=config.GROQ_API_KEY)

_JSON_BLOCK = re.compile(r"\{.*\}|\[.*\]", re.DOTALL)

RETRY_ATTEMPTS = 2
RETRY_DELAY_SECONDS = 3

# Groq's error text for a daily-quota breach says "tokens per day" / "(TPD)" and quotes a
# multi-hour retry-after; a per-minute rate limit is transient (seconds) and worth retrying.
_DAILY_QUOTA_MARKERS = ("tokens per day", "(tpd)", "requests per day", "(rpd)")


class QuotaExhaustedError(Exception):
    """Raised when Groq reports the account's daily token/request quota is used up. Retrying
    immediately is pointless -- the account-wide quota won't refill for the rest of the day --
    so callers should stop attempting further generations this run instead of burning more
    candidates against a guaranteed failure."""


def _call_once(model, system, user_content, max_tokens):
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system + "\n\nRespond with a JSON object."},
            {"role": "user", "content": user_content},
        ],
    )
    text = response.choices[0].message.content
    match = _JSON_BLOCK.search(text)
    if not match:
        raise ValueError(f"No JSON found in model response: {text[:300]}")
    return json.loads(match.group(0))


def call_for_json(model, system, user_content, max_tokens=2048):
    """A transient Groq API hiccup (rate limit, timeout, brief 5xx) previously killed whatever
    story/thread was being generated at that moment outright -- with no retry, a single blip
    could cost a candidate for the entire run. Retries with a short fixed delay before giving up
    and re-raising to the caller (which still handles a persistent failure the same as before).

    A daily-quota breach (429 "tokens per day"/"requests per day") is different: it won't clear
    up in the next few seconds, so it's raised immediately as QuotaExhaustedError instead of
    being retried -- callers use this to stop trying further candidates for the rest of the run."""
    last_exc = None
    for attempt in range(1, RETRY_ATTEMPTS + 2):
        try:
            return _call_once(model, system, user_content, max_tokens)
        except RateLimitError as exc:
            message = str(exc).lower()
            if any(marker in message for marker in _DAILY_QUOTA_MARKERS):
                logger.error("Groq daily quota exhausted, not retrying: %s", exc)
                raise QuotaExhaustedError(str(exc)) from exc
            last_exc = exc
            if attempt <= RETRY_ATTEMPTS:
                logger.warning(
                    "Groq call failed (attempt %d/%d), retrying in %ds: %s",
                    attempt, RETRY_ATTEMPTS + 1, RETRY_DELAY_SECONDS, exc,
                )
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                logger.error("Groq call failed after %d attempt(s), giving up: %s", attempt, exc)
        except Exception as exc:
            last_exc = exc
            if attempt <= RETRY_ATTEMPTS:
                logger.warning(
                    "Groq call failed (attempt %d/%d), retrying in %ds: %s",
                    attempt, RETRY_ATTEMPTS + 1, RETRY_DELAY_SECONDS, exc,
                )
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                logger.error("Groq call failed after %d attempt(s), giving up: %s", attempt, exc)
    raise last_exc
