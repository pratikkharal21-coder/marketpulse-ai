# MarketPulse AI

A private, single-user pipeline that watches financial/market news, macro data, and crypto/FX/
commodities feeds, then uses an LLM to turn the highest-impact stories into ready-to-post X
(Twitter) threads — with charts — and emails them to one person, three times a day. Runs entirely
in the cloud on GitHub Actions' free tier, independent of any local machine, triggered by a
precise external scheduler. Total running cost: **$0/month**.

## Architecture at a glance

```
GitHub Actions schedule: trigger (3x/day, fixed UTC cron — see "Production schedule")
        ▼
GitHub Actions (ubuntu-latest, free tier)
        │
        ▼
  feeds.py ──fetch──▶ state.py ──dedupe──▶ triage.py ──score (Groq qwen/qwen3.8-27b)──▶
        │                                                                              │
        ▼                                                                              ▼
  generate.py (5 short threads)  +  longform.py (2 deep dives)   ← both call Groq qwen/qwen3.8-27b
        │                                  │
        ▼                                  ▼
   chart.py (price/bar/histogram/pie/trend/flowchart, via yfinance + matplotlib)
        │
        ▼
  report.py (Jinja2 HTML + inline image CIDs) ──▶ mailer.py (Gmail SMTP) ──▶ recipient inbox
        │
        ▼
  state.json committed back to the repo (dedupe memory for next run)
```

## Why it's built this way (context for review)

This started as a local Python script and was migrated to GitHub Actions specifically so it
keeps running even when the owner's PC is off. That migration drove several downstream
decisions worth knowing about before reviewing the code:

- **No paid APIs anywhere.** News comes from public RSS feeds (no key needed). The LLM is Groq's
  free tier (Llama models, OpenAI-compatible API). Email is Gmail SMTP with an App Password.
  This was an explicit constraint, not an oversight — it shows up as rate-limit handling
  (`triage.py` batches into groups of 12 to stay under Groq's free-tier TPM cap) and as the
  retry/backoff already built into the `groq` SDK.
- **Timing runs on GitHub's own `schedule:` cron, currently.** A precise external scheduler
  (cron-job.org, calling the GitHub REST API's `dispatches` endpoint with a fine-grained PAT) was
  planned and documented but never actually set up, so native `schedule:` cron is what's live —
  see `.github/workflows/marketpulse.yml` and "Production schedule" below for the real, current
  tradeoffs (1-3hr possible drift, no DST awareness).
- **State persistence is git-based, not a database.** `state.json` (a flat map of seen-story
  hashes → timestamps, pruned after 7 days) is committed back to the repo by the workflow itself
  after each run. This was chosen over `actions/cache` because cache is explicitly best-effort/
  evictable, and silent dedupe failures would mean duplicate emails. The tradeoff: concurrent
  runs can race on the `git push` — see the retry-with-rebase loop in the workflow's last step,
  added after a real race condition produced a false "failed" run (the email had already sent
  fine; only the housekeeping commit collided).
- **Chart generation is fully self-hosted, not TradingView.** TradingView screenshots would
  require either a paid API or scraping (against their ToS). Instead `chart.py` pulls real price
  data via `yfinance` (free, no key) and renders with `matplotlib` (headless `Agg` backend, since
  there's no display in CI). Six chart types exist — see below.
- **The model picks its own visuals and tickers.** Rather than hardcoding "this category gets
  this chart," the LLM is given a Yahoo Finance ticker-symbol cheat sheet and a description of
  each chart type, and decides per-story whether a visual helps and which type fits — including
  setting `"none"` when it doesn't. This is a deliberate quality bet (less predictable, but avoids
  forcing irrelevant charts onto stories that don't need one).

## Repo layout

| File | Responsibility |
|---|---|
| `main.py` | Orchestrates one end-to-end run (the entry point GitHub Actions calls). |
| `config.py` | Loads `.env` / environment variables; all tunable knobs live here. |
| `feeds.py` | `FEEDS` dict (category → RSS URLs) + fetch/parse logic. Failures are per-feed and non-fatal. |
| `state.py` | Load/save/prune `state.json`; dedupes stories already sent. |
| `triage.py` | Batches headlines to Groq (qwen/qwen3.8-27b) for relevance/impact scoring; drops low-value stories. |
| `generate.py` | Per-story call to Groq (qwen/qwen3.8-27b) producing one short thread (3-5 tweets). |
| `longform.py` | Same model, deeper prompt: one 8-10 tweet "deep dive" thread per top story (historical context, scenarios, risk). |
| `persona.py` | Shared prompt fragments: tone/neutrality rules, X-engagement craft rules, "no bare recaps" rule, and the visual-selection guidelines + ticker cheat sheet. Both `generate.py` and `longform.py` compose their system prompts from these. |
| `chart.py` | Renders all 6 visual types to PNG bytes; `resolve_visual()` dispatches on the model's `visual_type` field. |
| `ai_client.py` | Thin wrapper around the Groq SDK: forces JSON-object responses, parses them. |
| `report.py` | Jinja2 HTML template; assigns each chart a Content-ID for inline embedding. |
| `mailer.py` | Gmail SMTP send, `multipart/related` with inline images. |
| `poster.py` | Optional: posts the run's top-ranked thread(s) to X as a real reply chain via `tweepy`, respecting a monthly post budget. A no-op unless X credentials are configured -- see "Auto-posting to X" below. |
| `.github/workflows/marketpulse.yml` | The only thing GitHub's scheduler runs. `workflow_dispatch`-only (see above). |

## Data flow per run

1. **`feeds.py`** fetches every RSS feed in `FEEDS` (6 categories: markets, macro, fx,
   commodities, crypto, tech_ai — 30 feeds total), filtered to the last `LOOKBACK_HOURS`.
2. **`state.py`** drops anything already sent in a previous run (hash = story URL).
3. **`triage.py`** scores every remaining headline 0-10 on relevance/impact via Groq, in batches
   of 12 (tuned to stay under the free-tier token-per-minute cap). Survivors above
   `TRIAGE_RELEVANCE_THRESHOLD` are kept, sorted, capped at `MAX_STORIES_ANALYZED`.
4. **`generate.py`** takes the top `MAX_SHORT_THREADS` survivors and asks Groq (qwen/qwen3.8-27b)
   for one short thread each, plus an optional visual spec.
5. **`longform.py`** takes the top `MAX_LONGFORM_STORIES` (overlaps with step 4 by design — the
   single most important story often deserves both a quick take and a deep dive) and asks for a
   longer, more analytical thread.
6. **`chart.py`** renders whichever visual each story's JSON response specified (or none).
7. **`poster.py`** (optional, off by default) posts the top `X_MAX_THREADS_PER_RUN` thread(s)
   across both groups to X for real, as a genuine reply chain, before the email goes out.
9. **`report.py`** renders the HTML email and collects inline images by Content-ID.
10. **`mailer.py`** sends it via Gmail SMTP.
11. **`state.py`** marks everything sent; the workflow commits `state.json` back to the repo.

## The six visual types (`chart.py`)

| Type | When the model picks it | Notes |
|---|---|---|
| `price_chart` | One tradable instrument is central to the story | Pulls real OHLC data via `yfinance`; green/red by direction |
| `bar_chart` | Comparison across a few categories at one point in time | Auto-detects "level" vs "delta" data (delta gets ± signs and red/green; level gets neutral blue) |
| `histogram` | Distribution of many similar values (e.g. analyst targets) | |
| `pie_chart` | Composition / share of a whole | |
| `trend_chart` | A metric over several periods, where shape matters | Plots a `numpy.polyfit` overlay — linear (degree 1) or cubic (degree 3) — so acceleration is visually obvious |
| `flowchart` | A short cause-effect chain | Vertical boxes + arrows, rendered with `matplotlib.patches.FancyBboxPatch` |

All chart titles wrap to 2 lines with ellipsis truncation (a real bug we hit: long headlines
used as chart titles were getting clipped off the canvas) and saves use `bbox_inches="tight"` as
a second line of defense.

## Auto-posting to X

By default MarketPulse only emails ready-to-post thread drafts -- nothing gets posted anywhere
on its own, and growing an X account off it means manually copying threads over yourself.
`poster.py` adds real auto-posting as an opt-in on top of that; email still always sends
regardless.

**Setup:** create an X Developer app at developer.x.com with **Read and Write** permissions,
generate/regenerate the access token *after* setting that permission (an access token generated
before is stuck read-only), and set four secrets -- locally in `.env`, in the cloud as GitHub
Actions repo secrets (`X_API_KEY`, `X_API_SECRET`, `X_ACCESS_TOKEN`, `X_ACCESS_TOKEN_SECRET`).
Posting turns on automatically once all four are present; `X_AUTO_POST_ENABLED=false` forces it
back off without removing the keys.

**Budget, not best-effort:** X's free API tier caps writes at 500 posts/month account-wide, and
every tweet in a thread is its own post -- a 10-tweet deep dive is 10 posts, not 1. `state.py`
tracks posts made this UTC calendar month (`x_post_budget` in `state.json`, resets automatically
each month) against `X_MONTHLY_POST_CAP` (default 480, a safety margin below the real 500 cap).
Each run posts up to `X_MAX_THREADS_PER_RUN` threads (default 1), chosen from that run's threads
+ deep dives ranked together by the same engagement composite used for email ordering, skipping
any candidate whose tweet count wouldn't fit the remaining budget rather than posting it
partially. At the defaults (1 thread/run, ~4 tweets average, 3 runs/day) that's roughly
360 posts/month even before the cap engages -- headroom for the occasional longer deep dive.

**What actually posts:** the chosen thread's tweets go out as a real reply chain (each replies
to the previous one, in the same "N/TOTAL" order the model already writes them in), with the
story's chart image, if any, attached to the first tweet via the v1.1 media upload endpoint. A
failed upload just drops the image rather than blocking the tweet. `seed_replies` (the
follow-up reply suggestions in each item) are NOT auto-posted -- persona.py wrote those framed
as "replies the user could post," so they stay a manual, editorial choice rather than going out
unreviewed.

**Failure mode:** if posting fails partway through a thread (rate limit, transient API error),
`poster.py` stops rather than retrying from tweet 1 -- a partial thread on X is fixable by hand;
a duplicated opening tweet from a naive retry isn't. The run still emails everything normally
either way; a posting failure never blocks or delays the email.

## Known rough edges (good places to look for improvement)

- **Groq free-tier rate limits cause visible retry noise.** `triage.py`'s batch size (12) and
  `generate.py`/`longform.py`'s per-story calls were tuned empirically against 429s, not from a
  documented limit. Logs show frequent `429 → retry` cycles (handled gracefully by the SDK, but a
  run can take 2-3 minutes because of it).
- **No automated tests.** Everything has been verified by running the real pipeline against live
  feeds/APIs and reading logs/output. There's no test suite, no CI lint/test step in the
  workflow — just the one job that runs the actual program.
- **No type hints / dataclasses.** Story/thread/draft objects are plain dicts threaded through
  every function. Works, but `mypy` would have a field day.
- **`chart.py`'s ticker normalization is minimal** (`.strip().replace("/", "")`) — it's caught
  one real malformed-ticker case (`NZD/USD=X` from the model) but isn't a general validator.
- **The model occasionally returns malformed JSON shapes** (e.g. a bare array instead of the
  requested object) — `generate.py`/`longform.py` now validate `isinstance(result, dict)` after a
  production incident where this crashed the whole run. Worth asking whether this class of
  problem is fully closed off or just patched for the one shape we saw.
- **Bar chart label offset math** (`chart.py`, the `offset = max(...)` line in
  `generate_bar_chart`) is a heuristic, not a principled calculation — works for the values we've
  tested, may misplace labels for very large or very small magnitudes.
- **No automated tests around the email-state race condition fix** — the retry-with-rebase loop
  in the workflow was added reactively after observing the failure; it hasn't been deliberately
  load-tested with genuinely concurrent triggers.
- **A silent model-retirement produced days of empty digests.** Groq retired `llama-3.1-8b-instant`
  and `llama-3.3-70b-versatile` from this account (they now 404 as `model_not_found`); since
  `triage._triage_batch` catches per-batch exceptions and just skips the batch instead of failing
  the run, every triage call quietly failed, 0 stories ever survived, and the workflow "succeeded"
  while only ever sending the empty-digest notice — for several days before anyone noticed. Fixed
  by moving to `qwen/qwen3.8-27b` (see `config.py`), but the underlying gap remains: nothing pages
  on triage's survivor count going to 0 for multiple consecutive runs, which is exactly what a
  silent upstream deprecation looks like.

## Local development

```powershell
cd marketpulse
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
copy .env.example .env   # fill in GROQ_API_KEY, GMAIL_ADDRESS, GMAIL_APP_PASSWORD, RECIPIENT_EMAIL
.venv\Scripts\python main.py
```

`.env` is gitignored; secrets live only there locally and as encrypted GitHub Actions secrets in
the cloud (`GROQ_API_KEY`, `GMAIL_ADDRESS`, `GMAIL_APP_PASSWORD`, `RECIPIENT_EMAIL`). Tunable
knobs (model names, thresholds, counts) are documented with defaults in `.env.example`.

## Production schedule

The originally-planned cron-job.org setup (see below) was never actually completed, so timing
is currently driven by GitHub Actions' own `schedule:` trigger in
`.github/workflows/marketpulse.yml` — fixed UTC cron expressions tuned for `Europe/London`
during BST, no DST awareness, and GitHub's shared-runner queue can run it 1-3 hours late:

- ~14:00 — midday digest, every day
- ~18:00 — afternoon digest, every day
- ~21:00 — close digest, **weekdays only** (no Saturday/Sunday)

Times will drift ~1hr late once GMT/winter starts (late October) unless the cron expressions are
updated. Revisit `cron-job.org` for precise, DST-aware timing if that's ever worth the ~10 minutes
of setup — it would replace the `schedule:` block above with `workflow_dispatch`-only timing
driven externally, POSTing to the GitHub Actions dispatch endpoint at the exact minute.

## Cost

$0/month: Groq free tier (~1 triage call + ~7 generation calls per run × 3 runs/day, well inside
free-tier limits), GitHub Actions free tier (a few minutes of `ubuntu-latest` per run, free tier
covers 2,000 min/month for private repos), Gmail SMTP (free), cron-job.org (free tier).
