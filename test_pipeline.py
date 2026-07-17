"""Tests for the generation pipeline's backfill/exclusion/seen-tracking behavior -- the fixes
for the output-volume-collapse bug (Part 0.5): candidates must not be capped to a fixed top-N
slice before generation is attempted, so one blocked/failed story doesn't silently shrink the
day's output when other viable candidates existed."""

import unittest
from unittest.mock import MagicMock, patch

import httpx
from groq import RateLimitError

import ai_client
import config
import generate
import longform
import persona
import verify


def _daily_quota_error():
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(429, request=request)
    return RateLimitError(
        "Rate limit reached for model `llama-3.3-70b-versatile` in organization `org_x` on "
        "tokens per day (TPD): Limit 100000, Used 99992, Requested 800. Please try again in "
        "1h25m3.847s. Visit https://console.groq.com/settings/billing to learn more.",
        response=response, body=None,
    )


def _story(n, title=None):
    return {
        "source": "markets",
        "title": title or f"Story {n}: distinct headline with a number 4{n}%",
        "summary": f"Story {n} summary text describing a real market move of 4{n}%.",
        "link": f"http://x.test/{n}",
        "published": "2026-01-01T00:00:00+00:00",
        "relevance": 8, "impact": 7, "triage_reason": "test",
    }


def _ok_response(n):
    return {
        "thread": [f"1/1 Story {n} moved 4{n}% today."],
        "hook_shape": "number_led", "visual_type": "none", "visual_confidence": 0,
        "seed_replies": [], "quote_angle": None,
        "relevance": 8, "expected_engagement": 7, "market_significance": 7, "confidence": 7,
    }


def _empty_response():
    return {
        "thread": [],
        "hook_shape": "number_led", "visual_type": "none", "visual_confidence": 0,
        "seed_replies": [], "quote_angle": None,
        "relevance": 5, "expected_engagement": 5, "market_significance": 5, "confidence": 5,
    }


class ShortThreadsBackfillTests(unittest.TestCase):
    def test_backfills_past_the_target_count_when_early_candidates_fail(self):
        # 7 candidates, first 2 produce empty threads (simulating a block/failure), target is
        # MAX_SHORT_THREADS (5) -- the OLD behavior sliced to stories[:5] up front, so 2
        # failures there would have produced only 3 threads with candidates #6/#7 never tried.
        stories = [_story(i) for i in range(7)]
        responses = [_empty_response(), _empty_response()] + [_ok_response(i) for i in range(2, 7)]

        call_count = {"n": 0}

        def fake_call_for_json(model, system, user_content, max_tokens=1024):
            resp = responses[call_count["n"]]
            call_count["n"] += 1
            return resp

        # This test is about the backfill-past-failures logic specifically, not the separate
        # MAX_GENERATION_ATTEMPTS quota-headroom cap -- raise the cap so 7 attempts can happen.
        with patch("generate.config.MAX_GENERATION_ATTEMPTS", 10):
            with patch("generate.call_for_json", side_effect=fake_call_for_json):
                threads, used_links = generate.generate_short_threads(stories)

        self.assertEqual(len(threads), config.MAX_SHORT_THREADS)
        self.assertEqual(len(used_links), config.MAX_SHORT_THREADS)
        # Candidates #0 and #1 failed and are NOT in used_links; #2-#6 (5 stories) are.
        self.assertNotIn("http://x.test/0", used_links)
        self.assertNotIn("http://x.test/1", used_links)
        for i in range(2, 7):
            self.assertIn(f"http://x.test/{i}", used_links)

    def test_stops_once_target_count_reached_even_with_more_candidates(self):
        stories = [_story(i) for i in range(10)]
        with patch("generate.call_for_json", side_effect=lambda *a, **k: _ok_response(0)):
            threads, used_links = generate.generate_short_threads(stories)
        self.assertEqual(len(threads), config.MAX_SHORT_THREADS)

    def test_exclude_links_are_skipped_entirely(self):
        stories = [_story(i) for i in range(3)]
        exclude = {"http://x.test/0"}
        with patch("generate.call_for_json", side_effect=lambda *a, **k: _ok_response(0)):
            threads, used_links = generate.generate_short_threads(stories, exclude_links=exclude)
        self.assertNotIn("http://x.test/0", used_links)
        self.assertEqual(len(threads), 2)

    def test_all_candidates_exhausted_returns_fewer_than_target_without_erroring(self):
        # A genuinely quiet/low-quality day: fewer usable candidates than the target count.
        # This must be a normal, non-erroring outcome, not padded with anything.
        stories = [_story(i) for i in range(2)]
        with patch("generate.call_for_json", side_effect=lambda *a, **k: _ok_response(0)):
            threads, used_links = generate.generate_short_threads(stories)
        self.assertEqual(len(threads), 2)

    def test_backfill_stops_at_max_generation_attempts_even_with_more_candidates(self):
        # MAX_GENERATION_ATTEMPTS caps how many calls a single run will spend per stage, so
        # one run's backfill can't burn the whole day's shared Groq quota by itself. With the
        # cap lower than the candidate pool and every candidate failing, the loop must stop at
        # the cap, not exhaust all 10 stories.
        stories = [_story(i) for i in range(10)]
        call_count = {"n": 0}

        def fake_call_for_json(*a, **k):
            call_count["n"] += 1
            return _empty_response()

        with patch("generate.config.MAX_GENERATION_ATTEMPTS", 4):
            with patch("generate.call_for_json", side_effect=fake_call_for_json):
                threads, used_links = generate.generate_short_threads(stories)
        self.assertEqual(len(threads), 0)
        self.assertEqual(call_count["n"], 4)

    def test_stops_backfilling_once_daily_quota_is_exhausted(self):
        # Once Groq's daily TPD quota is hit, every subsequent candidate is guaranteed to fail
        # identically -- burning through the rest of the candidate list just wastes run time
        # (and, previously, kept re-triggering the same doomed retry-with-backoff). The loop
        # should stop at the first quota_exhausted result instead of trying all 10 stories.
        stories = [_story(i) for i in range(10)]
        call_count = {"n": 0}

        def fake_call_for_json(model, system, user_content, max_tokens=1024):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return _ok_response(0)
            raise ai_client.QuotaExhaustedError(str(_daily_quota_error()))

        with patch("generate.call_for_json", side_effect=fake_call_for_json):
            threads, used_links = generate.generate_short_threads(stories)

        self.assertEqual(len(threads), 1)
        # Only 2 attempts: one success, one that hit the quota wall and stopped the loop --
        # candidates #2-#9 must never have been tried.
        self.assertEqual(call_count["n"], 2)


class DeepDiveBackfillTests(unittest.TestCase):
    def test_backfills_past_the_target_count_when_early_candidates_fail(self):
        stories = [_story(i) for i in range(5)]
        responses = [_empty_response()] + [_ok_response(i) for i in range(1, 5)]

        call_count = {"n": 0}

        def fake_call_for_json(model, system, user_content, max_tokens=2048):
            resp = responses[call_count["n"]]
            call_count["n"] += 1
            return resp

        with patch("longform.call_for_json", side_effect=fake_call_for_json):
            items, used_links = longform.generate_top_longform(stories)

        self.assertEqual(len(items), config.MAX_LONGFORM_STORIES)
        self.assertNotIn("http://x.test/0", used_links)

    def test_stops_backfilling_once_daily_quota_is_exhausted(self):
        stories = [_story(i) for i in range(10)]
        call_count = {"n": 0}

        def fake_call_for_json(model, system, user_content, max_tokens=2048):
            call_count["n"] += 1
            raise ai_client.QuotaExhaustedError(str(_daily_quota_error()))

        with patch("longform.call_for_json", side_effect=fake_call_for_json):
            items, used_links = longform.generate_top_longform(stories)

        self.assertEqual(len(items), 0)
        self.assertEqual(call_count["n"], 1)


class AIClientRetryTests(unittest.TestCase):
    def setUp(self):
        # The daily-quota tests flip this module-level flag; reset it so one test's outcome
        # can't leak into the next (each real GitHub Actions run is a fresh process, but the
        # test suite runs everything in one).
        ai_client._groq_daily_quota_exhausted = False

    def test_recovers_from_a_transient_failure(self):
        call_count = {"n": 0}

        def flaky(*a, **k):
            call_count["n"] += 1
            if call_count["n"] < 2:
                raise TimeoutError("simulated transient failure")
            resp = MagicMock()
            resp.choices = [MagicMock(message=MagicMock(content='{"ok": true}'))]
            return resp

        with patch("ai_client.client.chat.completions.create", side_effect=flaky):
            with patch("ai_client.time.sleep"):
                result = ai_client.call_for_json("model", "sys", "user")

        self.assertEqual(result, {"ok": True})
        self.assertEqual(call_count["n"], 2)

    def test_gives_up_and_reraises_after_persistent_failure(self):
        call_count = {"n": 0}

        def always_fails(*a, **k):
            call_count["n"] += 1
            raise TimeoutError("persistent failure")

        with patch("ai_client.client.chat.completions.create", side_effect=always_fails):
            with patch("ai_client.time.sleep"):
                with self.assertRaises(TimeoutError):
                    ai_client.call_for_json("model", "sys", "user")

        self.assertEqual(call_count["n"], ai_client.RETRY_ATTEMPTS + 1)

    def test_daily_quota_error_is_not_retried(self):
        # A per-minute rate limit is transient and worth retrying; a daily TPD quota breach
        # won't clear for the rest of the day, so retrying it (even a few times) just wastes
        # time against a guaranteed failure. Must raise immediately, on the first attempt --
        # with no Gemini fallback configured, there's nowhere else to go.
        call_count = {"n": 0}

        def rate_limited(*a, **k):
            call_count["n"] += 1
            raise _daily_quota_error()

        with patch("ai_client.gemini_client", None):
            with patch("ai_client.client.chat.completions.create", side_effect=rate_limited):
                with patch("ai_client.time.sleep") as mock_sleep:
                    with self.assertRaises(ai_client.QuotaExhaustedError):
                        ai_client.call_for_json("model", "sys", "user")

        self.assertEqual(call_count["n"], 1)
        mock_sleep.assert_not_called()

    def test_daily_quota_falls_back_to_gemini_when_configured(self):
        # With a Gemini key configured, a Groq daily-quota breach should transparently hand the
        # same call to Gemini instead of giving up the candidate outright.
        fake_gemini = MagicMock()
        fake_gemini.models.generate_content.return_value = MagicMock(text='{"ok": true}')

        def rate_limited(*a, **k):
            raise _daily_quota_error()

        with patch("ai_client.gemini_client", fake_gemini):
            with patch("ai_client.client.chat.completions.create", side_effect=rate_limited):
                with patch("ai_client.time.sleep"):
                    result = ai_client.call_for_json("model", "sys", "user")

        self.assertEqual(result, {"ok": True})
        fake_gemini.models.generate_content.assert_called_once()
        self.assertEqual(fake_gemini.models.generate_content.call_args.kwargs["model"], config.GEMINI_FALLBACK_MODEL)

    def test_subsequent_calls_skip_groq_once_daily_quota_is_marked_exhausted(self):
        # Avoid re-hitting a Groq call that's guaranteed to fail the same way every time within
        # a run once the daily wall has already been confirmed hit once.
        fake_gemini = MagicMock()
        fake_gemini.models.generate_content.return_value = MagicMock(text='{"ok": true}')
        groq_call = MagicMock(side_effect=lambda *a, **k: (_ for _ in ()).throw(_daily_quota_error()))

        with patch("ai_client.gemini_client", fake_gemini):
            with patch("ai_client.client.chat.completions.create", groq_call):
                with patch("ai_client.time.sleep"):
                    ai_client.call_for_json("model", "sys", "user")
                    self.assertEqual(groq_call.call_count, 1)
                    ai_client.call_for_json("model", "sys", "user")
                    # Groq was never called again for the second request -- straight to Gemini.
                    self.assertEqual(groq_call.call_count, 1)
                    self.assertEqual(fake_gemini.models.generate_content.call_count, 2)

    def test_gemini_fallback_also_failing_raises_quota_exhausted(self):
        fake_gemini = MagicMock()
        fake_gemini.models.generate_content.side_effect = RuntimeError("Gemini also unavailable")

        def rate_limited(*a, **k):
            raise _daily_quota_error()

        with patch("ai_client.gemini_client", fake_gemini):
            with patch("ai_client.client.chat.completions.create", side_effect=rate_limited):
                with patch("ai_client.time.sleep"):
                    with self.assertRaises(ai_client.QuotaExhaustedError):
                        ai_client.call_for_json("model", "sys", "user")

    def test_transient_rate_limit_without_daily_quota_wording_is_still_retried(self):
        request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
        response = httpx.Response(429, request=request)
        transient = RateLimitError(
            "Rate limit reached for model `llama-3.1-8b-instant`: requests per minute (RPM): "
            "Limit 30, Used 30. Please try again in 1.2s.",
            response=response, body=None,
        )
        call_count = {"n": 0}

        def flaky(*a, **k):
            call_count["n"] += 1
            if call_count["n"] < 2:
                raise transient
            resp = MagicMock()
            resp.choices = [MagicMock(message=MagicMock(content='{"ok": true}'))]
            return resp

        with patch("ai_client.client.chat.completions.create", side_effect=flaky):
            with patch("ai_client.time.sleep"):
                result = ai_client.call_for_json("model", "sys", "user")

        self.assertEqual(result, {"ok": True})
        self.assertEqual(call_count["n"], 2)


class VisualSchemaConsistencyTests(unittest.TestCase):
    """persona.VISUAL_TYPES is meant to be the single source of truth for which visual_type
    values are valid -- both the prose menu (VISUAL_GUIDELINES) and the JSON schema in
    generate.py/longform.py are supposed to be built FROM it. This previously drifted silently:
    a type retired from the menu was left in the JSON schema, and two new types (fred_series_
    chart, company_revenue_chart) were fully described in prose but never added to the JSON
    schema at all, making them unreachable no matter how well VISUAL_GUIDELINES described them.
    These tests fail loudly if that ever happens again instead of only surfacing in production
    logs weeks later."""

    def test_every_visual_type_appears_in_the_short_thread_schema(self):
        for visual_type in persona.VISUAL_TYPES:
            with self.subTest(visual_type=visual_type):
                self.assertIn(f'"{visual_type}"', generate.SYSTEM_PROMPT)

    def test_every_visual_type_appears_in_the_deep_dive_schema(self):
        for visual_type in persona.VISUAL_TYPES:
            with self.subTest(visual_type=visual_type):
                self.assertIn(f'"{visual_type}"', longform.SYSTEM_PROMPT)

    def test_retired_types_are_absent_from_both_schemas(self):
        for visual_type in verify.RETIRED_VISUAL_TYPES:
            with self.subTest(visual_type=visual_type):
                self.assertNotIn(f'"{visual_type}"', generate.SYSTEM_PROMPT)
                self.assertNotIn(f'"{visual_type}"', longform.SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
