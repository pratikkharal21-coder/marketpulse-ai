import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env", encoding="utf-8-sig")


def _require(name):
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


GROQ_API_KEY = _require("GROQ_API_KEY")
GMAIL_ADDRESS = _require("GMAIL_ADDRESS")
GMAIL_APP_PASSWORD = _require("GMAIL_APP_PASSWORD")
RECIPIENT_EMAIL = _require("RECIPIENT_EMAIL")

TRIAGE_MODEL = os.environ.get("TRIAGE_MODEL", "llama-3.1-8b-instant")
GENERATE_MODEL = os.environ.get("GENERATE_MODEL", "llama-3.3-70b-versatile")
TRIAGE_RELEVANCE_THRESHOLD = int(os.environ.get("TRIAGE_RELEVANCE_THRESHOLD", "6"))
MAX_STORIES_ANALYZED = int(os.environ.get("MAX_STORIES_ANALYZED", "10"))
MAX_SHORT_THREADS = int(os.environ.get("MAX_SHORT_THREADS", "5"))
MAX_LONGFORM_STORIES = int(os.environ.get("MAX_LONGFORM_STORIES", "2"))
LOOKBACK_HOURS = int(os.environ.get("LOOKBACK_HOURS", "8"))
REGEN_THRESHOLD = int(os.environ.get("REGEN_THRESHOLD", "6"))

BASE_DIR = Path(__file__).parent
STATE_PATH = BASE_DIR / "state.json"
LOG_PATH = BASE_DIR / "marketpulse.log"
