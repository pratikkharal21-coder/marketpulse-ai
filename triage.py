import logging

import config
from ai_client import call_for_json
from persona import PERSONA

logger = logging.getLogger("marketpulse.triage")

SYSTEM_PROMPT = (
    PERSONA
    + """

You will be given a numbered list of news headlines with short summaries. For each item, score its \
relevance to financial markets — macro data, equities, fixed income, FX, commodities, and company \
earnings — and its potential market impact. Score political, policy, election, or geopolitical \
headlines only on their financial/market impact, never on political significance.

Respond with ONLY a JSON object of this shape, one entry per input item, in the same order:
{"scores": [{"index": 0, "relevance": 0-10, "impact": 0-10, "reason": "one short phrase"}, ...]}

No prose, no markdown fences, just the JSON object."""
)

BATCH_SIZE = 12


def _triage_batch(batch):
    numbered = "\n".join(
        f"{i}. [{item['source']}] {item['title']} — {item['summary'][:120]}"
        for i, item in enumerate(batch)
    )

    try:
        result = call_for_json(config.TRIAGE_MODEL, SYSTEM_PROMPT, numbered, max_tokens=1024)
        scores = result["scores"]
    except Exception as exc:
        logger.error("Triage batch call failed, skipping this batch: %s", exc)
        return []

    scored_items = []
    for entry in scores:
        idx = entry.get("index")
        if idx is None or idx >= len(batch):
            continue
        item = dict(batch[idx])
        item["relevance"] = entry.get("relevance", 0)
        item["impact"] = entry.get("impact", 0)
        item["triage_reason"] = entry.get("reason", "")
        scored_items.append(item)
    return scored_items


def triage(items):
    if not items:
        return []

    scored_items = []
    for i in range(0, len(items), BATCH_SIZE):
        scored_items.extend(_triage_batch(items[i : i + BATCH_SIZE]))

    survivors = [
        item
        for item in scored_items
        if item["relevance"] >= config.TRIAGE_RELEVANCE_THRESHOLD
        or item["impact"] >= config.TRIAGE_RELEVANCE_THRESHOLD
    ]
    survivors.sort(key=lambda x: x["relevance"] + x["impact"], reverse=True)
    logger.info("Triage: %d in -> %d survive threshold", len(items), len(survivors))
    return survivors[: config.MAX_STORIES_ANALYZED]
