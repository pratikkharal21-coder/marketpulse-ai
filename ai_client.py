import json
import logging
import time

from google import genai
from google.genai import types as genai_types
from groq import Groq, RateLimitError

import config

logger = logging.getLogger("marketpulse.ai")

client = Groq(api_key=config.GROQ_API_KEY)
gemini_client = genai.Client(api_key=config.GEMINI_API_KEY) if config.GEMINI_API_KEY else None

_json_decoder = json.JSONDecoder()

RETRY_ATTEMPTS = 2
RETRY_DELAY_SECONDS = 3

# Groq's error text for a daily-quota breach says "tokens per day" / "(TPD)" and quotes a
# multi-hour retry-after; a per-minute rate limit is transient (seconds) and worth retrying.
_DAILY_QUOTA_MARKERS = ("tokens per day", "(tpd)", "requests per day", "(rpd)")

# Once Groq's daily quota is confirmed exhausted, every further call in this process would hit
# the same wall -- skip straight to the Gemini fallback instead of re-trying a doomed Groq call
# each time. Resets naturally since each GitHub Actions run is a fresh process.
_groq_daily_quota_exhausted = False


class QuotaExhaustedError(Exception):
    """Raised when Groq's daily quota is exhausted AND the Gemini fallback (if configured) also
    failed or isn't set up. Retrying immediately is pointless -- the account-wide Groq quota
    won't refill for the rest of the day -- so callers should stop attempting further
    generations this run instead of burning more candidates against a guaranteed failure."""


def _extract_json(text):
    """Finds the first '{' or '[' and decodes exactly one JSON value from there, ignoring
    anything after it. Gemini's JSON mode occasionally emits trailing data after a complete,
    valid object (seen in testing -- not a truncation, a second fragment tacked on after a
    clean close brace); a naive json.loads(whole_text) or greedy-regex-then-loads both choke on
    that, even though the actual answer is intact and perfectly parseable."""
    positions = [i for i in (text.find("{"), text.find("[")) if i != -1]
    if not positions:
        raise ValueError(f"No JSON found in model response: {text[:300]}")
    try:
        obj, _ = _json_decoder.raw_decode(text, min(positions))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON in model response: {text[:300]}") from exc
    return obj


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
    return _extract_json(response.choices[0].message.content)


def _call_gemini_once(system, user_content, max_tokens):
    # thinking_budget=0 disables Gemini's default extended-reasoning mode -- left on, it spends
    # an unpredictable chunk of max_output_tokens "thinking" before writing the actual JSON,
    # which was truncating output in testing. We don't need chain-of-thought for this task.
    response = gemini_client.models.generate_content(
        model=config.GEMINI_FALLBACK_MODEL,
        contents=user_content,
        config=genai_types.GenerateContentConfig(
            system_instruction=system + "\n\nRespond with a JSON object.",
            response_mime_type="application/json",
            max_output_tokens=max_tokens,
            thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return _extract_json(response.text)


def _fallback_to_gemini(system, user_content, max_tokens, groq_exc):
    if gemini_client is None:
        raise QuotaExhaustedError(str(groq_exc) if groq_exc else "Groq daily quota exhausted") from groq_exc

    last_exc = groq_exc
    for attempt in range(1, RETRY_ATTEMPTS + 2):
        try:
            result = _call_gemini_once(system, user_content, max_tokens)
            logger.info("Gemini fallback (%s) succeeded.", config.GEMINI_FALLBACK_MODEL)
            return result
        except Exception as exc:
            last_exc = exc
            if attempt <= RETRY_ATTEMPTS:
                logger.warning(
                    "Gemini fallback call failed (attempt %d/%d), retrying in %ds: %s",
                    attempt, RETRY_ATTEMPTS + 1, RETRY_DELAY_SECONDS, exc,
                )
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                logger.error("Gemini fallback failed after %d attempt(s), giving up: %s", attempt, exc)
    raise QuotaExhaustedError(str(groq_exc) if groq_exc else str(last_exc)) from last_exc


def call_for_json(model, system, user_content, max_tokens=2048):
    """A transient Groq API hiccup (rate limit, timeout, brief 5xx) previously killed whatever
    story/thread was being generated at that moment outright -- with no retry, a single blip
    could cost a candidate for the entire run. Retries with a short fixed delay before giving up
    and re-raising to the caller (which still handles a persistent failure the same as before).

    A daily-quota breach (429 "tokens per day"/"requests per day") is different: it won't clear
    up in the next few seconds. Instead of giving up outright, this falls back to Gemini (if
    GEMINI_API_KEY is configured) so the run can keep producing output; only if Gemini also
    fails (or isn't configured) does this raise QuotaExhaustedError, which callers use to stop
    trying further candidates for the rest of the run."""
    global _groq_daily_quota_exhausted

    if _groq_daily_quota_exhausted:
        return _fallback_to_gemini(system, user_content, max_tokens, None)

    last_exc = None
    for attempt in range(1, RETRY_ATTEMPTS + 2):
        try:
            return _call_once(model, system, user_content, max_tokens)
        except RateLimitError as exc:
            message = str(exc).lower()
            if any(marker in message for marker in _DAILY_QUOTA_MARKERS):
                logger.error("Groq daily quota exhausted: %s", exc)
                _groq_daily_quota_exhausted = True
                return _fallback_to_gemini(system, user_content, max_tokens, exc)
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
