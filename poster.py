import io
import logging
import time

import tweepy

import config

logger = logging.getLogger("marketpulse.poster")

_client = None
_api_v1 = None


def _get_clients():
    """Lazy singletons -- constructing tweepy's clients does no network I/O itself, but keeping
    this behind a function means importing poster.py never requires X credentials to be set,
    only actually posting does."""
    global _client, _api_v1
    if _client is None:
        _client = tweepy.Client(
            consumer_key=config.X_API_KEY,
            consumer_secret=config.X_API_SECRET,
            access_token=config.X_ACCESS_TOKEN,
            access_token_secret=config.X_ACCESS_TOKEN_SECRET,
        )
        auth = tweepy.OAuth1UserHandler(
            config.X_API_KEY, config.X_API_SECRET, config.X_ACCESS_TOKEN, config.X_ACCESS_TOKEN_SECRET,
        )
        _api_v1 = tweepy.API(auth)
    return _client, _api_v1


def _upload_media(image_bytes):
    """Chart images are attached via the v1.1 media/upload endpoint -- v2 has no media upload
    of its own yet, only media_ids on create_tweet. Returns None (post without an image) rather
    than raising, since a missing chart shouldn't block the tweet it belongs to."""
    if not image_bytes:
        return None
    _, api_v1 = _get_clients()
    try:
        media = api_v1.media_upload(filename="chart.png", file=io.BytesIO(image_bytes))
        return media.media_id
    except Exception as exc:
        logger.warning("Chart upload to X failed, posting without image: %s", exc)
        return None


def post_thread(item):
    """Posts one generated thread's tweets to X as a real reply chain (each tweet replies to the
    previous one, matching how the model already numbers them "N/TOTAL"). The first tweet
    carries the chart image, if any. Stops and returns whatever was posted so far on the first
    failure -- a partial thread on X is recoverable manually; retrying from tweet 1 would double
    -post the opening tweets.

    Returns the number of tweets actually posted (0 if nothing went out)."""
    tweets = item.get("thread") or []
    if not tweets:
        return 0

    client, _ = _get_clients()
    media_id = _upload_media(item.get("chart_image"))

    posted = 0
    reply_to = None
    try:
        for i, tweet_text in enumerate(tweets):
            kwargs = {"text": tweet_text}
            if reply_to:
                kwargs["in_reply_to_tweet_id"] = reply_to
            if i == 0 and media_id:
                kwargs["media_ids"] = [media_id]
            response = client.create_tweet(**kwargs)
            reply_to = response.data["id"]
            posted += 1
            if i < len(tweets) - 1:
                time.sleep(config.X_POST_DELAY_SECONDS)
    except Exception as exc:
        logger.error(
            "Posting to X failed for '%s' after %d/%d tweet(s): %s",
            item.get("story_title"), posted, len(tweets), exc,
        )
    return posted


def post_top_threads(items, budget_remaining):
    """Posts up to config.X_MAX_THREADS_PER_RUN of `items` (already engagement-ranked by the
    caller, highest first) to X, skipping any item whose tweet count wouldn't fit in
    `budget_remaining` rather than posting a partial thread and stranding the rest for next
    month. Tries the next-ranked item instead of stopping outright, since a shorter lower-ranked
    thread fitting the remaining budget is better than posting nothing this run.

    Returns (posted_count, tweets_used) -- posted_count is how many items got a real thread on
    X; tweets_used is the total tweet/post count to charge against the monthly budget."""
    posted_count = 0
    tweets_used = 0

    for item in items:
        if posted_count >= config.X_MAX_THREADS_PER_RUN:
            break
        thread_len = len(item.get("thread") or [])
        if thread_len == 0 or thread_len > budget_remaining - tweets_used:
            continue

        actually_posted = post_thread(item)
        tweets_used += actually_posted
        if actually_posted > 0:
            posted_count += 1
            logger.info(
                "Posted to X: '%s' (%d/%d tweet(s))",
                item.get("story_title"), actually_posted, thread_len,
            )

    return posted_count, tweets_used
