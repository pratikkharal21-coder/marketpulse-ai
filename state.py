import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone

import config

logger = logging.getLogger("marketpulse.state")

RETENTION_DAYS = 7


def _hash_item(item):
    return hashlib.sha256(item["link"].encode("utf-8")).hexdigest()


def load():
    if not config.STATE_PATH.exists():
        return {"seen": {}}
    try:
        with open(config.STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read state file, starting fresh: %s", exc)
        return {"seen": {}}


def save(state):
    with open(config.STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def prune(state):
    cutoff = datetime.now(timezone.utc) - timedelta(days=RETENTION_DAYS)
    state["seen"] = {
        h: ts for h, ts in state["seen"].items() if datetime.fromisoformat(ts) > cutoff
    }
    return state


def filter_unseen(state, items):
    unseen = []
    seen_in_batch = set()
    for item in items:
        h = _hash_item(item)
        if h in state["seen"] or h in seen_in_batch:
            continue
        seen_in_batch.add(h)
        unseen.append(item)
    logger.info("%d/%d items are new", len(unseen), len(items))
    return unseen


def mark_sent(state, items):
    now = datetime.now(timezone.utc).isoformat()
    for item in items:
        state["seen"][_hash_item(item)] = now
    return state


RECENT_VISUALS_WINDOW = 10


def get_recent_visuals(state):
    """The visual types used by the last several PUBLISHED posts, across runs -- not just
    within the current run's batch -- so "avoid repeating the same visual type back-to-back"
    holds across separate scheduled runs, not only within one."""
    return list(state.get("recent_visuals", []))[-RECENT_VISUALS_WINDOW:]


def save_recent_visuals(state, used_visuals):
    """`used_visuals` is the seeded history plus everything appended during this run, so this
    replaces (not extends) the stored list to avoid double-counting on the next run."""
    state["recent_visuals"] = [v for v in used_visuals if v and v != "none"][-RECENT_VISUALS_WINDOW:]
    return state
