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
# Optional: free API key from fredaccount.stlouisfed.org, powers fred_series_chart (real macro
# data -- CPI, unemployment, Fed funds rate, ...). Without it, that visual type is unavailable
# and the model isn't offered it (see generate.py/longform.py's SYSTEM_PROMPT construction).
FRED_API_KEY = os.environ.get("FRED_API_KEY")
GMAIL_ADDRESS = _require("GMAIL_ADDRESS")
GMAIL_APP_PASSWORD = _require("GMAIL_APP_PASSWORD")
RECIPIENT_EMAIL = _require("RECIPIENT_EMAIL")

# llama-3.1-8b-instant / llama-3.3-70b-versatile were retired from this Groq account (confirmed
# via /openai/v1/models -- both now 404 model_not_found), which silently zeroed every triage
# batch and produced only empty "no high-impact stories" emails for days. qwen/qwen3.8-27b is
# the remaining sizable non-reasoning general chat model on the free tier (the other survivors --
# openai/gpt-oss-* -- burn an unpredictable chunk of max_tokens on hidden reasoning before
# writing JSON, which risks the same truncation-on-budget failure the Gemini fallback already
# works around with thinking_budget=0; qwen3.8-27b showed no such overhead in testing).
TRIAGE_MODEL = os.environ.get("TRIAGE_MODEL", "qwen/qwen3.8-27b")
GENERATE_MODEL = os.environ.get("GENERATE_MODEL", "qwen/qwen3.8-27b")
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

# Optional: auto-post the top-ranked thread(s) to X instead of only emailing drafts. Needs an
# X Developer account's OAuth 1.0a credentials (developer.x.com -> your app -> "Keys and
# tokens"). Auto-enables once all four are present; X_AUTO_POST_ENABLED=false forces it off
# even with keys configured. Left unset (the default), nothing changes -- posting is opt-in.
X_API_KEY = os.environ.get("X_API_KEY")
X_API_SECRET = os.environ.get("X_API_SECRET")
X_ACCESS_TOKEN = os.environ.get("X_ACCESS_TOKEN")
X_ACCESS_TOKEN_SECRET = os.environ.get("X_ACCESS_TOKEN_SECRET")
X_AUTO_POST_ENABLED = _flag("X_AUTO_POST_ENABLED", True) and all(
    (X_API_KEY, X_API_SECRET, X_ACCESS_TOKEN, X_ACCESS_TOKEN_SECRET)
)
# X's free API tier caps writes at 500 posts/month account-wide, and EVERY tweet in a thread is
# a separate post -- a single 10-tweet deep dive is 10 posts, not 1. Default cap sits below 500
# as a safety margin (clock skew between our monthly bucket and X's own billing-cycle reset,
# plus this pipeline running 3x/day means overshooting by even one run's worth of threads is
# easy to do right at the boundary). Raise it if your actual X plan allows more.
X_MONTHLY_POST_CAP = int(os.environ.get("X_MONTHLY_POST_CAP", "480"))
# How many of this run's (already engagement-ranked) threads to actually post to X. Everything
# generated still gets emailed regardless -- this only throttles the auto-posted subset, so the
# free tier's monthly cap survives 3 runs/day without exhausting itself in one run. At the
# default of 1 thread/run (avg ~4 tweets) x 3 runs/day, that's ~360 posts/month even before the
# cap kicks in -- comfortable headroom for occasional longer deep-dive posts.
X_MAX_THREADS_PER_RUN = int(os.environ.get("X_MAX_THREADS_PER_RUN", "1"))
# Delay between consecutive tweets in a posted thread, so replies land in order and the account
# doesn't look automated/bursty to X's own abuse heuristics.
X_POST_DELAY_SECONDS = int(os.environ.get("X_POST_DELAY_SECONDS", "20"))

BASE_DIR = Path(__file__).parent
STATE_PATH = BASE_DIR / "state.json"
LOG_PATH = BASE_DIR / "marketpulse.log"
PROVENANCE_LOG_PATH = BASE_DIR / "provenance_log.jsonl"
