import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env", encoding="utf-8-sig")


def _require(name):
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _flag(name, default=True):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in ("0", "false", "no", "off")


GROQ_API_KEY = _require("GROQ_API_KEY")
# Optional: when set, ai_client falls back to Gemini for a generation call that hits Groq's
# daily token quota, instead of giving up for the rest of the run. Free at aistudio.google.com.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_FALLBACK_MODEL = os.environ.get("GEMINI_FALLBACK_MODEL", "gemini-flash-lite-latest")
GMAIL_ADDRESS = _require("GMAIL_ADDRESS")
GMAIL_APP_PASSWORD = _require("GMAIL_APP_PASSWORD")
RECIPIENT_EMAIL = _require("RECIPIENT_EMAIL")

TRIAGE_MODEL = os.environ.get("TRIAGE_MODEL", "llama-3.1-8b-instant")
GENERATE_MODEL = os.environ.get("GENERATE_MODEL", "llama-3.3-70b-versatile")
TRIAGE_RELEVANCE_THRESHOLD = int(os.environ.get("TRIAGE_RELEVANCE_THRESHOLD", "6"))
MAX_STORIES_ANALYZED = int(os.environ.get("MAX_STORIES_ANALYZED", "10"))
MAX_SHORT_THREADS = int(os.environ.get("MAX_SHORT_THREADS", "5"))
MAX_LONGFORM_STORIES = int(os.environ.get("MAX_LONGFORM_STORIES", "2"))
# Separate from MAX_STORIES_ANALYZED: caps how many candidates each backfill loop (short
# threads, deep dives) will actually spend a generation call on per run. Each call costs
# ~7.5-9K Groq tokens against a 100K/day account-wide quota shared across all 3 daily runs --
# letting backfill run all the way to MAX_STORIES_ANALYZED (10) in both stages lets one run's
# worth of attempts (up to 20 calls) burn the entire day's budget by itself, starving later
# runs. Lower default trades a little resilience (fewer backfill attempts if early candidates
# get blocked) for leaving quota headroom for the rest of the day.
MAX_GENERATION_ATTEMPTS = int(os.environ.get("MAX_GENERATION_ATTEMPTS", "6"))
LOOKBACK_HOURS = int(os.environ.get("LOOKBACK_HOURS", "8"))
REGEN_THRESHOLD = int(os.environ.get("REGEN_THRESHOLD", "6"))

# Per-module toggles so each upgrade can be disabled independently without touching code.
# All default on; set e.g. CONTENT_ENGINE_ENABLED=false in .env to turn one off.
CONTENT_ENGINE_ENABLED = _flag("CONTENT_ENGINE_ENABLED", True)
ENGAGEMENT_SCORING_ENABLED = _flag("ENGAGEMENT_SCORING_ENABLED", True)

BASE_DIR = Path(__file__).parent
STATE_PATH = BASE_DIR / "state.json"
LOG_PATH = BASE_DIR / "marketpulse.log"
PROVENANCE_LOG_PATH = BASE_DIR / "provenance_log.jsonl"
