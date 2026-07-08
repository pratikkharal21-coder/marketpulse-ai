import json
import logging
import re
from datetime import datetime, timezone

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
    "cumulative_flow_chart",
)

TICKER_DRIVEN_TYPES = (
    "price_chart", "candlestick_chart", "renko_chart", "pnf_chart", "ohlc_chart",
    "heikin_ashi_chart", "kagi_chart", "area_chart", "volume_chart", "volume_profile_chart",
    "seasonality_chart",
)

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


def _tokenize(text):
    words = re.findall(r"[a-zA-Z][a-zA-Z\-']{2,}", (text or "").lower())
    return {w for w in words if w not in _STOPWORDS}


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


def ground_visual_spec(result, story):
    """For any spec-driven visual_type (one where the model invents the chart's own numbers,
    as opposed to a ticker-driven type where chart.py fetches real data), require every
    number in the spec to be traceable back to the story's own title/summary text -- the
    only free, already-coded source of ground truth for an arbitrary story's specific facts.
    If any number can't be traced, the visual is blocked (downgraded to "none") rather than
    published with invented data. Returns (result, warning_or_None); result may be mutated."""
    visual_type = result.get("visual_type") or "none"
    if visual_type not in SPEC_DRIVEN_FIELDS:
        return result, None

    spec = result.get(visual_type)
    if not spec:
        return result, None

    spec_numbers = _collect_spec_numbers(spec)
    if not spec_numbers:
        return result, None

    text_numbers = _extract_text_numbers(f"{story.get('title', '')} {story.get('summary', '')}")
    ungrounded = [n for n in spec_numbers if not _is_grounded(n, text_numbers)]

    if ungrounded:
        warning = (
            f"visual '{visual_type}' suppressed: {len(ungrounded)}/{len(spec_numbers)} chart "
            f"value(s) not traceable to the story's own text (e.g. {ungrounded[:3]})"
        )
        logger.info("Story '%s': %s", story.get("title", "?"), warning)
        result[visual_type] = None
        result["visual_type"] = "none"
        return result, warning

    return result, None


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


def build_provenance(story, chart_stats=None, spec_warning=None, bare_number_warnings=None):
    """A retained-but-not-necessarily-shown record of what free source(s) backed this piece
    of content and when, so any future error can be traced back and audited."""
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
    if spec_warning:
        provenance["warnings"].append(spec_warning)
    if bare_number_warnings:
        provenance["warnings"].extend(bare_number_warnings)
    return provenance


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
