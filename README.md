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
| `meme.py` | Optional (on by default): generates one extra sarcastic/mocking "meme take" thread per run, reacting to the day's top-ranked story. |
| `memeart.py` | Renders the meme take's cartoon illustration -- original, hand-drawn-in-code icons (not scraped/copied meme templates), same free/keyless/self-hosted approach as `chart.py`. |
| `satire.py` | Optional (on by default): builds a separate "lighter side" digest from `FEEDS["humor"]` (satire/comedy sources) -- bypasses `triage.py` entirely since a fictional headline has no real financial relevance to score. |
| `illustration.py` | Shared Pollinations.ai fetch + PIL caption/stat/badge compositing used by both `satire_image.py` and `news_image.py` -- one implementation, two different disclosure badges. |
| `satire_image.py` | Thin wrapper around `illustration.py` for satire items: burns in a red "SATIRE — NOT REAL NEWS" badge. |
| `news_image.py` | Thin wrapper around `illustration.py` for the `editorial_illustration` visual type (see `chart.py`): burns in a neutral "AI ILLUSTRATION — NOT A REAL PHOTO" badge instead. |
| `.github/workflows/marketpulse.yml` | The only thing GitHub's scheduler runs. `workflow_dispatch`-only (see above). |

## Data flow per run

1. **`feeds.py`** fetches every RSS feed in `FEEDS` (7 categories: markets, macro, fx,
   commodities, crypto, tech_ai, humor — 32 feeds total), filtered to the last `LOOKBACK_HOURS`.
2. **`state.py`** drops anything already sent in a previous run (hash = story URL). The `humor`
   category is then split off from everything else -- it skips step 3 entirely and goes straight
   to `satire.py` (step 8a below) instead.
3. **`triage.py`** scores every remaining (non-`humor`) headline 0-10 on relevance/impact via
   Groq, in batches of 12 (tuned to stay under the free-tier token-per-minute cap). Survivors
   above `TRIAGE_RELEVANCE_THRESHOLD` are kept, sorted, capped at `MAX_STORIES_ANALYZED`.
4. **`generate.py`** takes the top `MAX_SHORT_THREADS` survivors and asks Groq (qwen/qwen3.8-27b)
   for one short thread each, plus an optional visual spec.
5. **`longform.py`** takes the top `MAX_LONGFORM_STORIES` (overlaps with step 4 by design — the
   single most important story often deserves both a quick take and a deep dive) and asks for a
   longer, more analytical thread.
6. **`chart.py`** renders whichever visual each story's JSON response specified (or none).
7. **`poster.py`** (optional, off by default) posts the top `X_MAX_THREADS_PER_RUN` thread(s)
   across both groups to X for real, as a genuine reply chain, before the email goes out.
8. **`meme.py`** (optional, on by default) generates one extra mocking thread reacting to the
   run's top-ranked story, with a cartoon illustration from `memeart.py`.
8a. **`satire.py`** (optional, on by default) takes the most recent `FEEDS["humor"]` items
    (step 2) and generates up to `MAX_SATIRE_ITEMS` short joke takes, each with its own
    illustration from `satire_image.py`.
9. **`report.py`** renders the HTML email and collects inline images by Content-ID.
10. **`mailer.py`** sends it via Gmail SMTP.
11. **`state.py`** marks everything sent (including published satire items); the workflow commits
    `state.json` back to the repo.

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

## Meme take

One extra thread per run (on by default, `MEME_MODE_ENABLED=false` to turn off): a sarcastic,
mocking reaction to whichever story won the run's top engagement-ranked spot -- the FinTwit/WSB
"comic relief" counterpart to the neutral-analyst threads everywhere else in the digest. It's
additive, never a replacement -- the serious threads/deep dives are unaffected either way.

**Tone:** `meme.py` deliberately does NOT build its system prompt from `persona.py`'s neutral,
professional voice -- that's the one thing intentionally different. It still reuses the same
hard anti-fabrication blocks as every other thread (`verify.check_causal_claims`,
`check_thread_completeness`, `check_hashtag_discipline`): the joke can exaggerate a reaction
("my portfolio", "my therapist"), but every market FACT it references still has to trace back
to the real story, same as everywhere else in this pipeline. No hashtags, no mocking named
individuals -- only the market/situation/collective trader psychology.

**Visual:** `memeart.py` draws a small, fixed set of original cartoon icons (crying, panic,
rocket, diamond hands, confused, facepalm, cool) with matplotlib primitives -- circles, polygons,
simple shapes, in the same flat-color palette `chart.py` already uses. These are NOT scraped or
copied meme templates (no Wojak, Pepe, "this is fine" dog, or similar) -- this project's existing
rule against scraping/reusing someone else's protected image (see the TradingView note above)
applies here too, so every illustration is drawn from scratch, free and keyless, same as every
chart. The model picks which mood fits the story's tone and writes a short caption; an
unrecognized mood falls back to "confused" rather than posting with no image at all.

## Real-news illustrations (`editorial_illustration`)

One more entry in `chart.py`'s visual-type menu (see "The six visual types" above -- that table
predates several rounds of additions and is no longer literally six), for serious threads/deep
dives, not just the satire digest: `editorial_illustration` gives the model a cartoon-style
illustrated-scene option for stories that are dramatic/qualitative (a policy shock, a
geopolitical clash, a sentiment shift) rather than a clean numeric series -- a bold caption and
an optional single stat callout over a generated scene, instead of forcing an awkward bar/pie/
trend chart onto a story that doesn't actually have multi-value data. The model picks it the same
way it picks every other visual type (data-shape fit, self-reported confidence), nothing forces
it.

**Grounding:** registered in `verify.py`'s `SPEC_DRIVEN_FIELDS`, so it gets the exact same
anti-fabrication pipeline as bar_chart/pie_chart/etc. for free -- its one optional `stat_value`
must trace back to a real number in the story's own text (verified in testing: an invented stat
gets fully suppressed to `"none"` rather than published), and its `title` must share a grounded
term with the story's subject.

**Real people:** real news routinely centers a real named person (a Fed chair, a CEO, a head of
state) -- completely normal for the thread text, but the image prompt is separately instructed to
depict only a generic stand-in, and `news_image.py` hard-blocks (skips the image entirely) if a
real name from the story's own headline leaks into the generated scene description anyway, same
rule and same reasoning as the satire digest below.

**Why the disclosure badge matters here specifically:** unlike the satire digest, these threads
can get auto-posted to a real X account via `poster.py`. A photorealistic-looking AI scene of a
real event needs a clear, conspicuous "this is illustrative, not a photo" label to stay on the
right side of platform synthetic-media policies -- `news_image.py` burns in "AI ILLUSTRATION —
NOT A REAL PHOTO" for exactly this reason, not just as a style choice.

## Satire digest

A separate "lighter side" section (on by default, `SATIRE_DIGEST_ENABLED=false` to turn off):
up to `MAX_SATIRE_ITEMS` short joke takes per run, built from `FEEDS["humor"]` (currently The
Onion and The Daily Mash's business feed) instead of the real market feeds. Additive, never a
replacement -- and rendered in its own clearly-labeled email section, never mixed into the
serious threads/deep dives.

**Why it skips triage:** `triage.py`'s model scores real financial relevance/impact, which is
meaningless against an already-fictional headline. `main.py` splits the `humor` category off
`new_items` before triage ever runs, and `satire.py` picks straight from the most recent
candidates instead.

**Tone and grounding:** like `meme.py`, `satire.py` does NOT build its prompt from `persona.py`'s
neutral voice. Unlike `meme.py`, it also skips `verify.check_causal_claims` -- that check exists
to catch a claim not actually supported by a *real* source story, which doesn't apply when the
source headline is already satire. The structural hard-blocks that aren't about real-world
grounding (`check_thread_completeness`, `check_hashtag_discipline`) still apply.

**A real safety gap, found in testing, not theoretical:** a live run picked up a genuine Onion
headline, "Dennis Hastert Dies" -- a real, named public figure's real death -- and despite the
system prompt explicitly forbidding it ("never a named real individual"), the model wrote a
mocking joke about him by name. The prompt instruction alone was not trustworthy enough to rely
on. `satire.py` now runs two hard, deterministic filters instead of trusting the model to
self-police: `_is_sensitive_topic()` skips any headline matching death/tragedy keywords
*before* a generation call is even made, and `_names_real_individual_from_headline()` blocks the
generated thread afterward if a capitalized name from the headline shows up in the model's own
output anyway. Both are deliberately conservative -- skipping a borderline-serious headline costs
nothing (there's always another satire candidate to backfill with); letting one through costs a
lot.

**Visual:** `satire_image.py` (a thin wrapper around the shared `illustration.py`) fetches a
free, keyless background illustration from Pollinations.ai (prompted for a generic, invented
scene only -- explicitly never a real or recognizable person, even when the source headline
names one), then draws the bold poster-style caption and an optional stat callout on top with
PIL. This split exists because diffusion image models can't reliably spell words -- asking
Pollinations itself to render the caption came back as garbled nonsense in testing, so every
piece of readable text in the final image is drawn separately, the same self-rendered-text
approach `chart.py`/`memeart.py` already use everywhere else. Every image also gets a burned-in
"SATIRE — NOT REAL NEWS" badge, since it could be screenshotted and shared out of the context of
this email. `illustration.py` also hard-blocks the image (skips it, text still ships) if a real
name from the headline leaks into the generated scene description -- the same incident described
above ("A real safety gap") showed the system prompt's instruction alone isn't trustworthy
enough on its own, and that gap applied just as much to the separately-generated image scene as
it did to the thread text.

**A real, not just theoretical, free-tier gap:** Pollinations' anonymous/keyless tier has a real
rate/budget limit that returned `402 Payment Required` multiple times during testing, even at
low volume (2 requests, 20 seconds apart) -- looser than its documented "~1 request/15s" framing
suggests. `generate_satire_image()` treats this as a soft failure: the item still ships with its
joke text, just without an image, same contract as every other optional visual in this pipeline
(a missing `chart_image` is not an error).

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
covers 2,000 min/month for private repos), Gmail SMTP (free), cron-job.org (free tier),
Pollinations.ai image generation (free, keyless -- no account needed, see its real-world rate
limit caveat under "Satire digest" above).
