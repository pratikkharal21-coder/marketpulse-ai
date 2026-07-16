import json
import logging
import re
from datetime import datetime, timezone

import yfinance as yf

import config

logger = logging.getLogger("marketpulse.verify")

# Every visual_type that has its own JSON spec field the model fills in directly (as opposed
# to the ticker-driven types, where chart.py fetches real numbers from yfinance). None of
# these have a live free data source wired in -- the only "already-coded free source" that
# can ground their numbers is the story's own title/summary text pulled from feeds.py.
SPEC_DRIVEN_FIELDS = (
    "bar_chart", "dumbbell_chart", "grouped_bar_chart", "stacked_bar_chart", "waterfall_chart",
    "slope_chart", "bullet_chart", "pie_chart", "donut_chart", "treemap_chart", "histogram",
    "box_plot", "violin_plot", "scatter_chart", "bubble_chart", "correlation_matrix_chart",
    "regression_chart", "trend_chart", "term_structure_chart", "spread_chart", "zscore_chart",
    "cumulative_flow_chart", "custom_stat_visual",
)

# Retired from the persona.py menu (2026-07-15): 8 days of real production logs showed these
# four picked often but NEVER once with genuinely grounded data -- scatter_chart alone
# accounted for 14 fully-fabricated attempts (every single one 100% ungrounded, often
# suspiciously round sequences like [50, 60, 70]) across just 8 runs, with zero successes. A
# scatter/bubble/regression/correlation-matrix chart needs 3+ real paired observations, which
# short news article text essentially never states -- the model kept spending its one visual
# attempt on a type that was always going to be rejected. Kept here (not deleted from
# SPEC_DRIVEN_FIELDS/chart.py) as a hard, unconditional block so the existing grounding
# machinery still applies as a second layer if the model ever emits one of these anyway despite
# not being offered in the prompt -- an LLM doesn't always respect a stated enum perfectly.
RETIRED_VISUAL_TYPES = frozenset({
    "scatter_chart", "bubble_chart", "regression_chart", "correlation_matrix_chart",
})

TICKER_DRIVEN_TYPES = (
    "price_chart", "candlestick_chart", "renko_chart", "pnf_chart", "ohlc_chart",
    "heikin_ashi_chart", "kagi_chart", "area_chart", "volume_chart", "volume_profile_chart",
    "seasonality_chart", "moving_average_chart", "bollinger_bands_chart", "rsi_chart",
    "macd_chart", "drawdown_chart", "historical_volatility_chart", "cot_positioning_chart",
    # company_revenue_chart is ticker-driven like the rest -- SEC EDGAR revenue for a US-listed
    # company, keyed off the same stock ticker -- so it inherits ground_ticker_subject for free.
    "company_revenue_chart",
    # Same reasoning: crypto_market_cap_chart is keyed off the same BTC-USD/ETH-USD-style
    # ticker convention as price_chart, so it also inherits ground_ticker_subject for free.
    "crypto_market_cap_chart",
)

# fred_series_chart is macro-driven, not ticker-driven -- a FRED series ID (e.g. "CPIAUCSL")
# isn't a Yahoo Finance symbol, so it needs its own grounding path (see
# ground_fred_series_subject) rather than reusing ground_ticker_subject's yfinance lookup.
MACRO_SERIES_DRIVEN_TYPES = ("fred_series_chart",)

# Every visual_type mapped to the data "shape(s)" it's actually suited to represent. Used by
# check_shape_match to catch a visual type being force-fit onto a story whose data plainly
# isn't that shape (e.g. a multi-period trend line for a single two-number comparison).
# "single_stat", "photo_subject", and "process" are treated as near-universally plausible (see
# classify_story_shape) since almost any story can support a single-number callout, a subject
# photo, or a narrative cause-effect chart -- this keeps the check conservative, only firing on
# a clear mismatch rather than a marginal one.
VISUAL_TYPE_SHAPES = {
    "price_chart": {"single_stat", "multi_period_trend"},
    # candlestick/ohlc/heikin_ashi are the natural pick for a plain "X moved to $Y" story too --
    # a real chart of recent daily candles is informative even when the story itself doesn't
    # explicitly use trend language, same reasoning as price_chart/area_chart above. renko/pnf/
    # kagi stay multi_period_trend-only -- persona.py deliberately scopes those to be rare,
    # explicitly-technical-framing-only picks, and widening their shape match would work against
    # that.
    "candlestick_chart": {"single_stat", "multi_period_trend"},
    "renko_chart": {"multi_period_trend"},
    "pnf_chart": {"multi_period_trend"},
    "ohlc_chart": {"single_stat", "multi_period_trend"},
    "heikin_ashi_chart": {"single_stat", "multi_period_trend"},
    "kagi_chart": {"multi_period_trend"},
    "area_chart": {"single_stat", "multi_period_trend"},
    "volume_chart": {"multi_period_trend", "technical_reading"},
    "volume_profile_chart": {"multi_period_trend", "technical_reading"},
    "yield_curve_chart": {"macro_curve"},
    "seasonality_chart": {"multi_period_trend", "technical_reading"},
    "moving_average_chart": {"multi_period_trend", "technical_reading"},
    "bollinger_bands_chart": {"multi_period_trend", "technical_reading"},
    "rsi_chart": {"multi_period_trend", "technical_reading"},
    "macd_chart": {"multi_period_trend", "technical_reading"},
    "drawdown_chart": {"multi_period_trend", "technical_reading"},
    "historical_volatility_chart": {"multi_period_trend", "technical_reading"},
    # Includes "single_stat" like price_chart/area_chart -- a positioning story ("funds turn net
    # short gold") is a single-snapshot claim even though the chart itself shows weekly history,
    # and COT-specific language ("net short", "flipped", "specs piled in") isn't covered by the
    # generic multi_period_trend/flow_over_time keyword signals.
    "cot_positioning_chart": {"multi_period_trend", "flow_over_time", "single_stat"},
    "fred_series_chart": {"single_stat", "multi_period_trend"},
    "company_revenue_chart": {"single_stat", "multi_period_trend"},
    "crypto_market_cap_chart": {"single_stat", "multi_period_trend"},
    "bar_chart": {"two_point_comparison", "ranked_list"},
    "dumbbell_chart": {"two_point_comparison", "ranked_list"},
    "grouped_bar_chart": {"ranked_list", "two_point_comparison"},
    "stacked_bar_chart": {"composition", "ranked_list"},
    "waterfall_chart": {"breakdown", "ranked_list"},
    "slope_chart": {"two_point_comparison", "ranked_list"},
    "bullet_chart": {"single_stat", "two_point_comparison"},
    "pie_chart": {"composition"},
    "donut_chart": {"composition"},
    "treemap_chart": {"composition"},
    "histogram": {"distribution"},
    "box_plot": {"distribution"},
    "violin_plot": {"distribution"},
    "scatter_chart": {"correlation"},
    "bubble_chart": {"correlation"},
    "correlation_matrix_chart": {"correlation"},
    "regression_chart": {"correlation"},
    "trend_chart": {"multi_period_trend"},
    "term_structure_chart": {"macro_curve"},
    "spread_chart": {"multi_period_trend", "macro_curve"},
    "zscore_chart": {"ranked_list", "distribution"},
    "cumulative_flow_chart": {"flow_over_time", "multi_period_trend"},
    "flowchart": {"process"},
    "real_world_image": {"photo_subject"},
    "custom_stat_visual": {"single_stat", "two_point_comparison"},
}

# Shapes considered plausible for virtually any story -- a mismatch is only meaningful when the
# chosen type ALSO doesn't match any shape actually detected in the story text.
_UNIVERSAL_SHAPES = frozenset({"single_stat", "photo_subject", "process"})

# "three-week high", "52-week low", "10-day rally" etc. -- a numbered time span, digit or
# spelled-out, that the plain keyword list in _SHAPE_SIGNALS can't express as a fixed substring.
_NUMBERED_PERIOD_RE = re.compile(
    r"\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifty[\s-]two)"
    r"[\s-](?:day|week|month|quarter|year)s?\b",
    re.IGNORECASE,
)

_SHAPE_SIGNALS = {
    "multi_period_trend": (
        # Bare "week"/"month"/"quarter" are deliberately excluded -- they're substrings of
        # "quarterly"/"monthly" and appear in almost any earnings story regardless of whether
        # a multi-period trend is actually being described; these phrase-level signals are
        # more specific to an actual sequence-over-time being discussed.
        "trend", "since", "year-to-date", "ytd", "days in a row", "consecutive", "streak",
        "over the past", "over the last", "this week", "this month", "recent weeks",
        "recent months", "in recent", "week-over-week", "month-over-month", "several weeks",
        "several months", "several quarters", "past few",
    ),
    # Technical-analysis vocabulary that names a specific computed reading (RSI, MACD, moving
    # average, drawdown, realized volatility) -- the story naming the indicator IS the shape
    # signal here, distinct from generic "over time" phrasing in multi_period_trend.
    "technical_reading": (
        "rsi", "overbought", "oversold", "relative strength index", "macd", "bullish crossover",
        "bearish crossover", "momentum crossover", "moving average", "golden cross", "death cross",
        "bollinger", "volatility squeeze", "trading range", "below its high", "below its highs",
        "off its high", "off its highs", "off highs", "down from its peak", "down from its high",
        "below its peak", "below its all-time high", "drawdown", "underwater", "calm trading",
        "calmest", "choppy", "choppiness", "turbulent", "quiet trading", "volatile trading",
        "realized volatility", "implied volatility", "historically", "seasonal", "seasonality",
        "tends to rally", "tends to fall", "seasonal pattern", "volume spike", "trading volume",
        "heavy volume", "heavy trading", "unusual volume", "shares traded",
    ),
    "ranked_list": (
        "top ", "biggest", "largest", "leading", "worst", "best-performing", "best performing",
        "ranked", "among the", "list of", "highest", "lowest",
    ),
    "two_point_comparison": (
        " vs ", " vs. ", "versus", "compared to", "compared with", "from ", "before and after",
        "up from", "down from", "rose from", "fell from",
    ),
    "composition": (
        "share of", "percent of", "% of", "makes up", "portion", "mix", "breakdown", "weighting",
        "weightings", "comprised of", "consists of",
    ),
    "distribution": (
        "range of", "spread", "distribution", "varied between", "average of", "median",
        "spread of",
    ),
    "correlation": (
        "correlat", "relationship between", "tied to", "linked to", "in line with", "tracks",
        "moves with",
    ),
    "macro_curve": (
        "yield curve", "term structure", "tenor", "maturities", "2s10s", "curve",
    ),
    "flow_over_time": (
        "inflows", "outflows", "fund flows", "cumulative", "net flows", "net long", "net short",
        "positioning", "speculators", "specs ", "managed money", "hedge funds", "flipped",
        "piled into", "unwound",
    ),
    "breakdown": (
        "bridge", "breakdown", "broken down", "made up of", "contributors to", "drivers of",
    ),
}

_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")

_STOPWORDS = frozenset("""
    the a an and or but if then than that this these those of to in on at for from by with
    about into over after before under above below between out up down off again further
    once is are was were be been being have has had do does did will would could should
    may might must can shall not no nor so as it its it's their his her they them he she we
    you your our us i my mine yours theirs ours which who whom what when where why how all
    each every both few more most other some such only own same too very just also more than
    now here there while during through against toward towards per via amid among within
    without upon
""".split())

BASIS_KEYWORDS = (
    "today", "yesterday", "week", "month", "quarter", "year", "ytd", "year-to-date",
    "vs", "versus", "compared", "prior", "previous", "consensus", "estimate", "estimates",
    "expected", "forecast", "all-time", "record", "since", "close", "open", "session",
    "q1", "q2", "q3", "q4", "fy", "annualized", "yoy", "mom", "qoq", "guidance", "target",
)

CAUSAL_RE = re.compile(
    r"\b(?:because of|due to|caused by|triggered by|driven by|as a result of|thanks to|following|"
    r"amid|after)\b\s+(.{3,60})",
    re.IGNORECASE,
)

_DIRECTION_UP = re.compile(r"\b(up|rose|rise|rising|rallie[ds]?|surge[ds]?|jump(?:ed|s)?|climb(?:ed|s)?|gain(?:ed|s)?|soar(?:ed|s)?)\b", re.IGNORECASE)
_DIRECTION_DOWN = re.compile(r"\b(down|fell|fall(?:ing|s)?|drop(?:ped|s)?|slump(?:ed|s)?|sink(?:s)?|sank|sunk|plunge[ds]?|slide[ds]?|tumbl(?:e[ds]?|ing))\b", re.IGNORECASE)
_PCT_NEAR_DIRECTION = re.compile(r"[+-]?\d[\d,]*\.?\d*\s*%")

# Verbs that imply a large, dramatic move -- flagged (not blocked) when the real move is small,
# per "prefer precise, neutral language over dramatic verbs when the data doesn't support it."
_DRAMATIC_VERB_RE = re.compile(
    r"\b(surge[ds]?|soar(?:ed|s)?|plunge[ds]?|crash(?:ed|es)?|rocket(?:ed|s)?|tank(?:ed|s)?|"
    r"collapse[ds]?|explode[ds]?|skyrocket(?:ed|s)?)\b",
    re.IGNORECASE,
)
DRAMATIC_VERB_THRESHOLD_PCT = 2.0

# Unconditionally banned filler phrases -- generic hedge-words that could apply to any story on
# any day and add no information. "eyes on"/"in focus" get a narrower rule below since they're
# only filler when NOT anchored to a specific level/event.
_UNCONDITIONAL_FILLER_RE = re.compile(
    r"\b(amid|as investors digest|uncertainty looms)\b", re.IGNORECASE,
)
_CONDITIONAL_FILLER_RE = re.compile(r"\b(eyes on|in focus)\b", re.IGNORECASE)
_HASHTAG_RE = re.compile(r"#\w+")

_PERIOD_TOKEN_RE = re.compile(r"\bQ[1-4]\b|\bFY\s?\d{2,4}\b|\b20\d{2}\b", re.IGNORECASE)

_QUARTER_WORDS = {
    "q1": ("q1", "first quarter", "1st quarter"),
    "q2": ("q2", "second quarter", "2nd quarter"),
    "q3": ("q3", "third quarter", "3rd quarter"),
    "q4": ("q4", "fourth quarter", "4th quarter"),
}

# Generic finance/chart vocabulary that would trivially "match" between almost any two
# finance stories -- excluded when checking whether a chart's own title actually names the
# same subject as the story, so a title can't pass on a word like "revenue" alone while
# naming the wrong company entirely.
_GENERIC_CHART_VOCAB = frozenset("""
    revenue earnings growth market markets sector trend trends comparison overview
    performance outlook results result chart summary breakdown analysis quarter quarterly
    year yearly price prices rate rates target targets estimate estimates actual before
    after change changes mix share shares total net income sales guidance forecast data
    value values current prior previous period metric metrics figures numbers company
    companies stock stocks profit profits loss losses margin margins vs versus overview
""".split())


def _tokenize(text):
    words = re.findall(r"[a-zA-Z][a-zA-Z\-']{2,}", (text or "").lower())
    # Strip a trailing possessive ("nvidia's" -> "nvidia", "investors'" -> "investors") so the
    # same entity mentioned possessively in one place (a chart title) and plainly in another
    # (the story text) still overlaps -- every grounding check in this module (title, ticker,
    # causal claims, image queries, FRED series) is built on this function, so an unstripped
    # possessive silently produces a false "shares no grounded terms" mismatch anywhere the
    # source material happens to use 's.
    normalized = []
    for w in words:
        if w.endswith("'s"):
            w = w[:-2]
        elif w.endswith("'"):
            w = w[:-1]
        normalized.append(w)
    return {w for w in normalized if w not in _STOPWORDS}


def _extract_text_numbers(text):
    found = []
    for m in _NUMBER_RE.finditer(text or ""):
        raw = m.group(0).replace(",", "")
        try:
            found.append(float(raw))
        except ValueError:
            continue
    return found


def _collect_spec_numbers(obj):
    """Recursively pulls every numeric leaf out of a chart spec, ignoring strings/labels."""
    nums = []
    if isinstance(obj, bool):
        return nums
    if isinstance(obj, (int, float)):
        nums.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            nums.extend(_collect_spec_numbers(v))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            nums.extend(_collect_spec_numbers(v))
    return nums


def _is_grounded(value, text_numbers, rel_tol=0.02, abs_tol=0.05):
    for t in text_numbers:
        if abs(value - t) <= max(abs_tol, abs(t) * rel_tol):
            return True
    return False


def _collect_spec_strings(obj):
    """Recursively pulls every string leaf out of a chart spec (titles, labels, names)."""
    strs = []
    if isinstance(obj, str):
        strs.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            strs.extend(_collect_spec_strings(v))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            strs.extend(_collect_spec_strings(v))
    return strs


def _period_grounded(token, story_text_lower):
    t = token.lower().strip()
    if t in _QUARTER_WORDS:
        return any(v in story_text_lower for v in _QUARTER_WORDS[t])
    return t in story_text_lower


def _suppress(result, story, warning, field=None):
    logger.info("Story '%s': %s", story.get("title", "?"), warning)
    if field:
        result[field] = None
    result["visual_type"] = "none"
    return result, warning


def _field_for_visual_type(visual_type):
    """Which key in `result` holds a given visual_type's content, so suppressing it can null
    the right thing instead of leaving stale data sitting unused in the response."""
    if visual_type in SPEC_DRIVEN_FIELDS or visual_type == "flowchart":
        return visual_type
    if visual_type in TICKER_DRIVEN_TYPES:
        return "ticker"
    if visual_type in MACRO_SERIES_DRIVEN_TYPES:
        return "fred_series"
    if visual_type == "real_world_image":
        return "image_query"
    return None


def _paired_labels_values(spec):
    """Common {"labels": [...], "values": [...]} shape shared by several spec-driven types."""
    return list(zip(spec.get("labels") or [], spec.get("values") or []))


# Types whose spec reduces cleanly to a plain labels[]/values[] pairing -- a stat card can
# always represent "this label had this value" regardless of what the original chart framing
# was (a bar comparison, a time series, a slice of a pie), since the number itself is what
# matters for accuracy, not the chart geometry it was originally destined for.
_LABELS_VALUES_TYPES = frozenset({
    "bar_chart", "histogram", "pie_chart", "donut_chart", "treemap_chart", "waterfall_chart",
    "trend_chart", "spread_chart", "cumulative_flow_chart", "term_structure_chart", "zscore_chart",
})


def _build_downgrade_stats(visual_type, spec, text_numbers, max_stats=3):
    """When a spec-driven visual is rejected for containing fabricated (ungrounded) numbers,
    salvage whichever (label, value) pairs in the model's OWN spec ARE genuinely grounded in
    the story's own text, reusing its own labels/title rather than writing new ones from
    scratch -- so a story with one real number and one invented comparison point still gets an
    accurate single-stat visual instead of nothing. Returns a list of
    {"label", "value", "unit"} dicts, capped at max_stats; empty if nothing in the spec is
    salvageable or the type's shape doesn't reduce to a stat card at all (scatter/bubble/
    regression/correlation_matrix/box_plot/violin_plot -- these are inherently about a
    multi-point relationship or distribution, not a handful of standalone numbers)."""
    if not isinstance(spec, dict):
        return []
    unit = spec.get("unit", "") or ""

    triples = []
    if visual_type in _LABELS_VALUES_TYPES:
        for label, value in _paired_labels_values(spec):
            triples.append((label, value, unit))
    elif visual_type in ("dumbbell_chart", "slope_chart"):
        labels = spec.get("labels") or []
        start_label = spec.get("start_label") or "Before"
        end_label = spec.get("end_label") or "After"
        for label, value in zip(labels, spec.get("start_values") or []):
            triples.append((f"{label} ({start_label})", value, unit))
        for label, value in zip(labels, spec.get("end_values") or []):
            triples.append((f"{label} ({end_label})", value, unit))
    elif visual_type in ("grouped_bar_chart", "stacked_bar_chart"):
        labels = spec.get("labels") or []
        for series in spec.get("series") or []:
            name = series.get("name", "") if isinstance(series, dict) else ""
            for label, value in zip(labels, (series or {}).get("values") or []):
                triples.append((f"{name} {label}".strip(), value, unit))
    elif visual_type == "bullet_chart":
        if "value" in spec:
            triples.append(("Actual", spec.get("value"), unit))
        if "target" in spec:
            triples.append(("Target", spec.get("target"), unit))
    elif visual_type == "custom_stat_visual":
        for stat in spec.get("stats") or []:
            if isinstance(stat, dict) and "value" in stat:
                triples.append((stat.get("label", ""), stat.get("value"), stat.get("unit", unit)))

    stats = []
    for label, value, item_unit in triples:
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if not _is_grounded(value, text_numbers):
            continue
        stats.append({"label": str(label)[:40], "value": value, "unit": item_unit})
        if len(stats) >= max_stats:
            break
    return stats


def ground_visual_spec(result, story):
    """For any spec-driven visual_type (one where the model invents the chart's own content,
    as opposed to a ticker-driven type where chart.py fetches real data), require:
      1. every number in the spec to trace back to the story's own title/summary text,
      2. the chart's own title to be about the same subject as the story (catches a chart
         titled for a different company/entity entirely), and
      3. any quarter/year mentioned in the spec to be a period the story actually discusses
         (catches e.g. a Q1 chart attached to a Q2 story).
    The story's title/summary text is the only free, already-coded source of ground truth for
    an arbitrary story's specific facts. A title/period mismatch fully suppresses the visual
    (down to "none"); a spec containing some fabricated numbers among otherwise-real ones is
    downgraded to a custom_stat_visual built only from the values that ARE grounded (see
    _build_downgrade_stats) rather than losing the visual entirely over a partially-invented
    spec -- never publishes with invented or mismatched content either way. Returns
    (result, warning_or_None); result may be mutated."""
    visual_type = result.get("visual_type") or "none"
    if visual_type == "flowchart":
        return _ground_flowchart(result, story)
    if visual_type not in SPEC_DRIVEN_FIELDS:
        return result, None

    spec = result.get(visual_type)
    if not spec:
        return result, None

    story_text = f"{story.get('title', '')} {story.get('summary', '')}"
    story_tokens = _tokenize(story_text)
    story_text_lower = story_text.lower()

    spec_numbers = _collect_spec_numbers(spec)
    text_numbers = _extract_text_numbers(story_text)
    ungrounded_numbers = [n for n in spec_numbers if not _is_grounded(n, text_numbers)]
    if ungrounded_numbers:
        downgraded_stats = _build_downgrade_stats(visual_type, spec, text_numbers)
        if downgraded_stats:
            title = spec.get("title") if isinstance(spec, dict) else None
            if visual_type != "custom_stat_visual":
                result[visual_type] = None
            result["custom_stat_visual"] = {
                "title": title or story.get("title", "")[:60],
                "stats": downgraded_stats,
            }
            result["visual_type"] = "custom_stat_visual"
            warning = (
                f"visual '{visual_type}' downgraded to custom_stat_visual: kept "
                f"{len(downgraded_stats)} grounded value(s), dropped "
                f"{len(ungrounded_numbers)} fabricated/ungrounded value(s) "
                f"(e.g. {ungrounded_numbers[:3]})"
            )
            logger.info("Story '%s': %s", story.get("title", "?"), warning)
            return result, warning
        return _suppress(
            result, story,
            f"visual '{visual_type}' suppressed: {len(ungrounded_numbers)}/{len(spec_numbers)} chart "
            f"value(s) not traceable to the story's own text (e.g. {ungrounded_numbers[:3]})",
            field=visual_type,
        )

    title = spec.get("title") if isinstance(spec, dict) else None
    if title:
        title_tokens = _tokenize(title) - _GENERIC_CHART_VOCAB
        if title_tokens and not (title_tokens & story_tokens):
            return _suppress(
                result, story,
                f"visual '{visual_type}' suppressed: title '{title}' shares no grounded terms with the story",
                field=visual_type,
            )

    for s in _collect_spec_strings(spec):
        for m in _PERIOD_TOKEN_RE.finditer(s):
            if not _period_grounded(m.group(0), story_text_lower):
                return _suppress(
                    result, story,
                    f"visual '{visual_type}' suppressed: period '{m.group(0)}' in spec is not "
                    f"mentioned anywhere in the story",
                    field=visual_type,
                )

    return result, None


def _ground_flowchart(result, story):
    spec = result.get("flowchart")
    if not spec or not spec.get("steps"):
        return result, None

    story_tokens = _tokenize(f"{story.get('title', '')} {story.get('summary', '')}")
    steps_tokens = _tokenize(" ".join(spec["steps"]))
    if steps_tokens and not (steps_tokens & story_tokens):
        return _suppress(
            result, story,
            "visual 'flowchart' suppressed: steps share no grounded terms with the story",
            field="flowchart",
        )
    return result, None


def _fetch_ticker_name(ticker):
    try:
        info = yf.Ticker(ticker).info
        return info.get("shortName") or info.get("longName") or info.get("displayName")
    except Exception as exc:
        logger.warning("Could not fetch company/instrument name for ticker %s: %s", ticker, exc)
        return None


def ground_ticker_subject(ticker, story):
    """The model supplies a bare ticker symbol for ticker-driven visuals; nothing else
    validates that this ticker is actually the company/instrument the story is about --
    chart.py will happily render a real, accurate chart for the WRONG company if the model
    mixes them up. Resolves the ticker to its real name via yfinance (the same already-coded
    source used to render the chart itself) and requires that name to share a grounded word
    with the story's own text. Falls back to a literal-symbol check (e.g. a story that
    literally writes "Apple (AAPL)") if the name lookup fails or doesn't match on its own.
    Returns (ok, reason_or_None)."""
    if not ticker:
        return True, None

    story_text = f"{story.get('title', '')} {story.get('summary', '')}"
    story_tokens = _tokenize(story_text)

    name = _fetch_ticker_name(ticker)
    if name and (_tokenize(name) & story_tokens):
        return True, None

    bare_symbol = re.sub(r"[\^=].*$", "", ticker).replace("-USD", "").replace("/", "")
    if bare_symbol and bare_symbol.lower() in story_text.lower():
        return True, None

    if name is None:
        return False, f"could not verify ticker '{ticker}' against the story (name lookup failed) -- failing closed"
    return False, f"ticker '{ticker}' resolves to '{name}', which shares no grounded terms with the story"


def ground_fred_series_subject(series_id, story):
    """Same purpose as ground_ticker_subject but for FRED series IDs, which aren't Yahoo
    tickers and so can't reuse the yfinance name lookup -- checks the curated series'
    plain-English name (FRED_SERIES_NAMES, imported from chart.py so there's a single source of
    truth) shares a grounded word with the story. An unrecognized series_id fails closed the
    same way an unmapped ticker does. Returns (ok, reason_or_None)."""
    if not series_id:
        return True, None

    from chart import FRED_SERIES_NAMES

    name = FRED_SERIES_NAMES.get(series_id.strip().upper())
    if name is None:
        return False, f"'{series_id}' is not a recognized FRED series -- failing closed"

    story_tokens = _tokenize(f"{story.get('title', '')} {story.get('summary', '')}")
    if _tokenize(name) & story_tokens:
        return True, None
    return False, f"FRED series '{series_id}' ('{name}') shares no grounded terms with the story"


def ground_image_query(image_query, story):
    """The model supplies a free-text Wikipedia search query for real_world_image visuals;
    require it to share a grounded word with the story before spending a network call on it
    -- an ungrounded query would still return a real photo, just of the wrong subject.
    Returns (ok, reason_or_None)."""
    if not image_query:
        return True, None

    story_tokens = _tokenize(f"{story.get('title', '')} {story.get('summary', '')}")
    query_tokens = _tokenize(image_query)
    if not query_tokens or (query_tokens & story_tokens):
        return True, None
    return False, f"image query '{image_query}' shares no grounded terms with the story"


def check_visual_relevance(result, story):
    """Single entry point: whatever visual_type the model picked, verify its SUBJECT and
    CONTENT are grounded in this specific story before it is ever rendered -- ticker-driven,
    spec-driven, and photo visuals alike, so any future visual type automatically inherits
    the same rule from this one place. Downgrades to "none" on any mismatch (wrong company,
    wrong period, invented numbers, unrelated photo) rather than publishing something
    irrelevant. Returns (result, warning_or_None); result may be mutated."""
    visual_type = result.get("visual_type") or "none"

    if visual_type in RETIRED_VISUAL_TYPES:
        return _suppress(
            result, story,
            f"visual '{visual_type}' suppressed: retired type, never selectable regardless of "
            f"grounding (see RETIRED_VISUAL_TYPES)",
            field=visual_type,
        )

    if visual_type in TICKER_DRIVEN_TYPES:
        ok, reason = ground_ticker_subject(result.get("ticker"), story)
        if not ok:
            return _suppress(result, story, reason, field="ticker")
        return result, None

    if visual_type in MACRO_SERIES_DRIVEN_TYPES:
        ok, reason = ground_fred_series_subject(result.get("fred_series"), story)
        if not ok:
            return _suppress(result, story, reason, field="fred_series")
        return result, None

    if visual_type == "real_world_image":
        ok, reason = ground_image_query(result.get("image_query"), story)
        if not ok:
            return _suppress(result, story, reason, field="image_query")
        return result, None

    return ground_visual_spec(result, story)


VISUAL_CONFIDENCE_THRESHOLD = 6
VARIETY_REPEAT_CONFIDENCE_THRESHOLD = 8
RECENT_VISUALS_WINDOW = 10


def check_visual_confidence(result, story):
    """The model must self-report how confidently its chosen visual_type's DATA SHAPE (not
    just its numbers) actually fits this story, via a "visual_confidence" (0-10) field --
    "if the match... is weak or uncertain, default to no visual rather than guessing." A
    missing score fails closed (treated as 0) rather than assumed fine, since an LLM omitting
    the field is itself a sign the response wasn't produced carefully. Returns
    (result, warning_or_None); result may be mutated."""
    visual_type = result.get("visual_type") or "none"
    if visual_type == "none":
        return result, None

    confidence = result.get("visual_confidence")
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0

    if confidence < VISUAL_CONFIDENCE_THRESHOLD:
        return _suppress(
            result, story,
            f"visual '{visual_type}' suppressed: self-reported visual_confidence "
            f"{confidence} is below the {VISUAL_CONFIDENCE_THRESHOLD} threshold",
            field=_field_for_visual_type(visual_type),
        )
    return result, None


def classify_story_shape(story):
    """Heuristic, deterministic classification of what data "shape(s)" a story's own text
    plausibly supports (multi-period trend, ranked list, composition, ...), used as a second,
    independent signal alongside the model's self-reported confidence -- catches a forced-fit
    even if the model is (wrongly) confident about it. Always includes the universal shapes
    (single_stat/photo_subject/process) so the check stays conservative. Returns a set of
    shape tags."""
    text = f"{story.get('title', '')} {story.get('summary', '')}".lower()
    shapes = set(_UNIVERSAL_SHAPES)
    for shape, signals in _SHAPE_SIGNALS.items():
        if any(sig in text for sig in signals):
            shapes.add(shape)

    if _NUMBERED_PERIOD_RE.search(text):
        shapes.add("multi_period_trend")

    # A bare two-number comparison ("X vs Y", "from A to B") without any of the above signals
    # is still a very common story shape, so detect it directly off the raw text_numbers count.
    if len(_extract_text_numbers(text)) == 2:
        shapes.add("two_point_comparison")
    if len(_extract_text_numbers(text)) >= 3:
        shapes.add("ranked_list")

    return shapes


def check_shape_match(result, story):
    """Cross-checks the chosen visual_type's expected data shape(s) (VISUAL_TYPE_SHAPES)
    against the shapes actually detected in the story (classify_story_shape). Blocks only on
    a CLEAR, total mismatch (zero shape overlap) -- deliberately conservative, since shape
    detection from a short headline/summary is inherently fuzzy and the goal is to catch
    obvious forced fits (e.g. a correlation matrix for a single-number earnings beat), not to
    second-guess every marginal call. Returns (result, warning_or_None); result may be
    mutated."""
    visual_type = result.get("visual_type") or "none"
    expected_shapes = VISUAL_TYPE_SHAPES.get(visual_type)
    if not expected_shapes:
        return result, None

    story_shapes = classify_story_shape(story)
    if expected_shapes & story_shapes:
        return result, None

    return _suppress(
        result, story,
        f"visual '{visual_type}' suppressed: its expected data shape {sorted(expected_shapes)} "
        f"doesn't match anything detected in the story (detected: {sorted(story_shapes)})",
        field=_field_for_visual_type(visual_type),
    )


def check_visual_thread_consistency(result, story, thread_lines):
    """The chart and the tweet text are generated from the same JSON response but nothing
    otherwise confirms they actually agree with each other -- a spec-driven chart could show
    numbers that are individually grounded in the source story yet never actually appear in
    what the thread says, which would still read as a mismatch to anyone comparing the two. At
    least one of the chart's own numbers must be echoed in the thread text. Returns
    (result, warning_or_None); result may be mutated."""
    visual_type = result.get("visual_type") or "none"
    if visual_type not in SPEC_DRIVEN_FIELDS:
        return result, None

    spec = result.get(visual_type)
    if not spec:
        return result, None

    spec_numbers = _collect_spec_numbers(spec)
    if not spec_numbers:
        return result, None

    thread_numbers = _extract_text_numbers(" ".join(thread_lines))
    if any(_is_grounded(n, thread_numbers) for n in spec_numbers):
        return result, None

    return _suppress(
        result, story,
        f"visual '{visual_type}' suppressed: none of its chart values are echoed anywhere "
        f"in the accompanying thread text",
        field=visual_type,
    )


def check_visual_variety(result, story, recent_visuals):
    """Recency-based tiebreaker: if the chosen visual_type is identical to the single most
    recently PUBLISHED one (tracked across runs, not just this batch) and the model's own
    confidence in the pick isn't very high, prefer skipping the visual over repeating it
    back-to-back. Never overrides a genuinely confident match -- variety is only a
    tiebreaker among otherwise-marginal calls, per "must never override relevance or
    accuracy." Returns (result, warning_or_None); result may be mutated."""
    visual_type = result.get("visual_type") or "none"
    if visual_type == "none" or not recent_visuals:
        return result, None

    if recent_visuals[-1] != visual_type:
        return result, None

    try:
        confidence = float(result.get("visual_confidence") or 0)
    except (TypeError, ValueError):
        confidence = 0
    if confidence >= VARIETY_REPEAT_CONFIDENCE_THRESHOLD:
        return result, None

    return _suppress(
        result, story,
        f"visual '{visual_type}' suppressed for variety: repeats the immediately preceding "
        f"post's visual type and confidence ({confidence}) is below the "
        f"{VARIETY_REPEAT_CONFIDENCE_THRESHOLD} repeat threshold",
        field=_field_for_visual_type(visual_type),
    )


def select_visual(result, story, thread_lines, recent_visuals):
    """Single orchestrated entry point for the whole visual-selection pipeline: subject/content
    grounding, self-reported confidence threshold, data-shape match, chart-vs-thread
    consistency, and cross-run variety -- run in sequence, each seeing the previous step's
    (possibly already-suppressed) result. Callers get one call instead of wiring five
    individually, and any check added here in the future is automatically applied everywhere
    this is called. Returns (result, warnings) where warnings is a list (possibly empty)."""
    warnings = []

    result, w = check_visual_relevance(result, story)
    if w:
        warnings.append(w)

    result, w = check_visual_confidence(result, story)
    if w:
        warnings.append(w)

    result, w = check_shape_match(result, story)
    if w:
        warnings.append(w)

    result, w = check_visual_thread_consistency(result, story, thread_lines)
    if w:
        warnings.append(w)

    result, w = check_visual_variety(result, story, recent_visuals)
    if w:
        warnings.append(w)

    return result, warnings


def check_causal_claims(thread_lines, story):
    """Any 'X because of Y' / 'triggered by Y' / 'due to Y' style claim must share at least
    one non-generic word with the story's own title/summary -- otherwise the model is
    asserting a cause that isn't evidenced by the source text at all. Hard-blocks the whole
    piece of content, since a fabricated cause is exactly the kind of misleading claim this
    tool must never publish. Returns (ok, reason_or_None)."""
    story_tokens = _tokenize(f"{story.get('title', '')} {story.get('summary', '')}")
    full_text = " ".join(thread_lines)

    for m in CAUSAL_RE.finditer(full_text):
        clause = m.group(1)
        clause_tokens = _tokenize(clause)
        if clause_tokens and not (clause_tokens & story_tokens):
            reason = f"causal claim '...{m.group(0)[:70]}...' shares no grounded terms with the source story"
            return False, reason

    return True, None


def verify_ticker_direction(thread_lines, chart_stats):
    """When a ticker-driven visual was rendered, chart_stats carries the REAL computed
    direction (pct_change sign) straight from yfinance. Scan the thread for directional
    price language ('surged', 'fell', ...) attached to a percentage and hard-block if the
    thread asserts the opposite direction from what the real data shows -- the single
    clearest, cheapest form of fact-checking available since we already fetched the truth.
    Returns (ok, reason_or_None)."""
    if not chart_stats or chart_stats.get("pct_change") is None:
        return True, None

    real_direction = "up" if chart_stats["pct_change"] >= 0 else "down"
    full_text = " ".join(thread_lines)

    claimed_up = bool(_DIRECTION_UP.search(full_text))
    claimed_down = bool(_DIRECTION_DOWN.search(full_text))

    if claimed_up and claimed_down:
        # Thread discusses both directions (e.g. "up on the week but fell today") -- too
        # ambiguous for a blunt sign check, don't false-positive block it.
        return True, None
    if claimed_up and real_direction == "down":
        return False, f"thread says price moved up, but real data shows {chart_stats['pct_change']:+.2f}%"
    if claimed_down and real_direction == "up":
        return False, f"thread says price moved down, but real data shows {chart_stats['pct_change']:+.2f}%"

    return True, None


def check_verb_intensity(thread_lines, chart_stats):
    """Advisory (non-blocking): flags dramatic verbs ('surged', 'plunged', 'crashed', ...)
    used to describe a move that the real fetched data shows was actually small (< 2 pts).
    Not auto-corrected -- rewriting the model's prose automatically risks its own mistakes,
    so this is surfaced for review rather than silently edited. Returns a list of warnings."""
    if not chart_stats or chart_stats.get("pct_change") is None:
        return []

    magnitude = abs(chart_stats["pct_change"])
    if magnitude >= DRAMATIC_VERB_THRESHOLD_PCT:
        return []

    warnings = []
    for line in thread_lines:
        m = _DRAMATIC_VERB_RE.search(line)
        if m:
            warnings.append(
                f"dramatic verb '{m.group(0)}' used for a {magnitude:.2f}% move "
                f"(below the {DRAMATIC_VERB_THRESHOLD_PCT:.0f}% threshold) in: {line[:80]}"
            )
    return warnings


def check_bare_numbers(thread_lines):
    """Advisory (non-blocking) check: flags percentage/currency figures that have no nearby
    comparison-basis word (vs/YTD/prior/consensus/...) anywhere in their own tweet. This is a
    best-effort heuristic, not a semantic guarantee -- rule-based text matching cannot fully
    verify comparison intent, so these are surfaced for review rather than hard-blocked, to
    avoid silently suppressing legitimate content on false positives. Returns a list of
    warning strings (empty if nothing flagged)."""
    warnings = []
    for line in thread_lines:
        lower = line.lower()
        for m in _PCT_NEAR_DIRECTION.finditer(line):
            window = lower[max(0, m.start() - 40): m.end() + 40]
            if not any(kw in window for kw in BASIS_KEYWORDS):
                warnings.append(f"bare figure '{m.group(0)}' with no visible comparison basis in: {line[:80]}")
    return warnings


_TWEET_PREFIX_RE = re.compile(r"^\d+/\d+\s+")


def check_banned_filler(thread_lines):
    """Hard block on generic hedge-phrases that could apply to any story on any day and add no
    information ('amid', 'as investors digest', 'uncertainty looms'), plus 'eyes on'/'in focus'
    specifically when NOT anchored to a concrete level or date -- deterministic string matching,
    not a fuzzy heuristic, so this is enforced as a hard block rather than an advisory warning.
    Returns (ok, reason_or_None)."""
    for line in thread_lines:
        m = _UNCONDITIONAL_FILLER_RE.search(line)
        if m:
            return False, f"banned filler phrase '{m.group(0)}' in: {line[:80]}"

        m = _CONDITIONAL_FILLER_RE.search(line)
        if m:
            # Strip the leading "N/TOTAL " tweet-number prefix first -- otherwise every tweet
            # trivially "has a number" from its own numbering and this check never fires.
            body = _TWEET_PREFIX_RE.sub("", line)
            if not _extract_text_numbers(body) and not _PERIOD_TOKEN_RE.search(body):
                return False, f"'{m.group(0)}' used with no concrete level/date attached in: {line[:80]}"

    return True, None


_TWEET_NUMBERING_RE = re.compile(r"^(\d+)/(\d+)\s+")


def check_thread_completeness(thread_lines):
    """Hard block: every tweet numbers itself "N/TOTAL" (e.g. "3/4"), and that self-declared
    TOTAL must match how many tweets actually made it into the thread -- otherwise a reader's
    last visible tweet says "3/4" and a 4th that was promised never arrives. Seen in real
    production output; the most likely source is regen.py's "sharper rewrite" retry, which is
    an entirely independent model call that only gets compared on its expected_engagement
    score before being swapped in -- nothing else validates it preserved the original tweet
    count. Deterministic, not a heuristic: every tweet must have a numbering prefix, all of
    them must agree on the same TOTAL, TOTAL must equal len(thread_lines), and the N's must
    cover 1..TOTAL exactly once. Returns (ok, reason_or_None)."""
    if not thread_lines:
        return True, None

    numbers = []
    totals = set()
    for line in thread_lines:
        m = _TWEET_NUMBERING_RE.match(line)
        if not m:
            return False, f"tweet has no 'N/TOTAL' numbering prefix: {line[:80]}"
        numbers.append(int(m.group(1)))
        totals.add(int(m.group(2)))

    if len(totals) > 1:
        return False, f"tweets disagree on the thread's own declared total: {sorted(totals)}"

    declared_total = totals.pop()
    if declared_total != len(thread_lines):
        return False, (
            f"thread declares {declared_total} tweets ('N/{declared_total}') but only "
            f"{len(thread_lines)} were actually produced -- last tweet(s) missing"
        )

    if sorted(numbers) != list(range(1, declared_total + 1)):
        return False, f"tweet numbering isn't sequential 1..{declared_total}: got {numbers}"

    return True, None


def check_hashtag_discipline(thread_lines):
    """Hard block: zero hashtags anywhere in the thread, in any tweet -- X doesn't favor them
    and they read as dated. Deterministic counting, not a heuristic. Returns (ok, reason_or_None)."""
    for line in thread_lines:
        tags = _HASHTAG_RE.findall(line)
        if tags:
            return False, f"no hashtags allowed anywhere in the thread, found {tags} in: {line[:80]}"

    return True, None


def build_provenance(story, chart_stats=None, visual_warnings=None, advisory_warnings=None):
    """A retained-but-not-necessarily-shown record of what free source(s) backed this piece
    of content and when, so any future error can be traced back and audited. `visual_warnings`
    and `advisory_warnings` are each lists (possibly empty/None)."""
    provenance = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "story_source": story.get("source"),
        "story_link": story.get("link"),
        "story_published": story.get("published"),
        "data_sources": ["rss:" + str(story.get("source"))],
        "warnings": [],
    }
    if chart_stats:
        provenance["chart_data"] = chart_stats
        src = chart_stats.get("source")
        if src:
            provenance["data_sources"].append(src)
    if visual_warnings:
        provenance["warnings"].extend(visual_warnings)
    if advisory_warnings:
        provenance["warnings"].extend(advisory_warnings)
    return provenance


def rank_by_engagement(items):
    """Orders published items by a composite score built from the self-reported
    expected_engagement (surprise/shareability), market_significance (magnitude), and relevance
    (audience breadth) fields every item already carries -- a re-ordering of what generation
    already produced, never a reason to include or exclude anything. Note this is a heuristic
    proxy from the model's own self-assessment, not measured real-world X engagement data (no
    free source for that exists) -- it's the same honest limitation as the rest of this
    pipeline's self-reported confidence scores. Returns a new sorted list, highest first."""
    def score(item):
        return (
            item.get("expected_engagement", 0) * 0.4
            + item.get("market_significance", 0) * 0.35
            + item.get("relevance", 0) * 0.25
        )
    return sorted(items, key=score, reverse=True)


def log_provenance(items):
    """Appends one JSON line per published item (thread or deep dive) to an audit log, so any
    future report of misleading content can be traced back to exactly which free source(s)
    backed it and when it was generated. Best-effort: a logging failure must never block the
    actual send."""
    try:
        with open(config.PROVENANCE_LOG_PATH, "a", encoding="utf-8") as f:
            for item in items:
                provenance = item.get("provenance")
                if not provenance:
                    continue
                record = {"story_title": item.get("story_title"), **provenance}
                f.write(json.dumps(record, default=str) + "\n")
    except OSError as exc:
        logger.warning("Could not write provenance log: %s", exc)
