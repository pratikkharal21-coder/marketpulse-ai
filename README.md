# MarketPulse AI

Private, single-user pipeline: scans free RSS feeds (markets, macro, crypto, tech/AI) → filters
for impact with Llama 3.1 8B (via Groq) → drafts ready-to-post X content in 5 styles with Llama
3.3 70B (via Groq) → emails you the best ones. No paid news/financial/X/AI APIs required — runs
entirely on Groq's free API tier plus your own Gmail account.

## 1. Install

```powershell
cd "marketpulse"
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

## 2. Get credentials

- **Groq API key (free)**: console.groq.com/keys → sign up → "Create API Key". No credit card
  required for the free tier.
- **Gmail App Password**: Google Account → Security → enable 2-Step Verification → App
  Passwords → generate one for "Mail". This is a 16-character password, not your normal Gmail
  password.

## 3. Configure

```powershell
copy .env.example .env
```

Edit `.env` and fill in `GROQ_API_KEY`, `GMAIL_ADDRESS`, `GMAIL_APP_PASSWORD`,
`RECIPIENT_EMAIL` (can be the same Gmail address). Optional tuning vars are documented in the
file with sensible defaults.

## 4. Run it once manually

```powershell
.venv\Scripts\python main.py
```

Check `marketpulse.log` for what happened, and check your inbox. Run it a second time
immediately after — you should get a "no high-impact stories" email (or far fewer drafts)
because `state.json` now remembers what was already sent.

## 5. Schedule it (3x/day)

This registers 3 Windows Task Scheduler jobs that run the script unattended. Run these yourself
in an elevated or normal PowerShell — adjust the paths if your project lives somewhere else:

```powershell
$python = "C:\Users\prati\OneDrive\Desktop\social media company\marketpulse\.venv\Scripts\python.exe"
$script = "C:\Users\prati\OneDrive\Desktop\social media company\marketpulse\main.py"

schtasks /create /tn "MarketPulseAI_Morning" /tr "`"$python`" `"$script`"" /sc daily /st 07:00 /f
schtasks /create /tn "MarketPulseAI_Midday"  /tr "`"$python`" `"$script`"" /sc daily /st 13:00 /f
schtasks /create /tn "MarketPulseAI_Evening" /tr "`"$python`" `"$script`"" /sc daily /st 19:00 /f
```

To remove them later:

```powershell
schtasks /delete /tn "MarketPulseAI_Morning" /f
schtasks /delete /tn "MarketPulseAI_Midday" /f
schtasks /delete /tn "MarketPulseAI_Evening" /f
```

To change the times, just re-run the `/create` commands with different `/st` values (the `/f`
flag overwrites the existing task).

## How it works

1. `feeds.py` pulls headlines from the last `LOOKBACK_HOURS` across curated RSS feeds.
2. `state.py` drops anything already sent in a previous run.
3. `triage.py` makes one Groq call (Llama 3.1 8B) to score every remaining headline on
   relevance/impact; anything below `TRIAGE_RELEVANCE_THRESHOLD` is dropped.
4. `generate.py` makes one Groq call (Llama 3.3 70B) per surviving story (capped at
   `MAX_STORIES_ANALYZED`) to produce 5 styled, ready-to-post drafts with scores.
5. The top `MAX_DRAFTS_IN_EMAIL` drafts across all stories are ranked and emailed via Gmail SMTP.

## Tuning / troubleshooting

- Feeds occasionally move or rate-limit — edit the URL lists in `feeds.py` if one starts failing.
  A single failing feed is logged and skipped, it won't break the run.
- If every run says "no high-impact stories," lower `TRIAGE_RELEVANCE_THRESHOLD` in `.env`.
- If drafts feel generic, that's a triage/summary quality issue — RSS summaries are often thin;
  you can lengthen what's passed to the model in `generate.py`'s `user_content`.
- Cost: **$0**. Usage is roughly 1 triage call + up to 10 generation calls per run, 3 runs/day
  (~33 calls/day) — comfortably inside Groq's free-tier rate limits. Check
  console.groq.com/settings/limits if you ever hit a 429.
- If a model name in `.env` stops working (Groq periodically retires older open models), check
  the current free-tier model list at console.groq.com/docs/models and update `TRIAGE_MODEL` /
  `GENERATE_MODEL` accordingly.
