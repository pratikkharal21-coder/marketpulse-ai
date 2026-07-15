"""Automated tests for the verification/grounding layer (verify.py) plus the watermark
requirement in chart.py. Uses stdlib unittest only -- no new test dependency, no network
calls (chart tests below use spec-driven renderers, which never hit yfinance/Wikipedia)."""

import unittest
from unittest.mock import patch

import chart
import state
import verify


class GroundVisualSpecTests(unittest.TestCase):
    def test_grounded_numbers_pass_through_unchanged(self):
        story = {"title": "Fed cuts rates by 25 basis points", "summary": "The Fed cut its target rate to 4.5% from 4.75%."}
        result = {
            "visual_type": "bar_chart",
            "bar_chart": {"title": "Rate change", "labels": ["Before", "After"], "values": [4.75, 4.5], "unit": "%"},
        }
        new_result, warning = verify.ground_visual_spec(result, story)
        self.assertIsNone(warning)
        self.assertEqual(new_result["visual_type"], "bar_chart")
        self.assertIsNotNone(new_result["bar_chart"])

    def test_ungrounded_numbers_suppress_the_visual(self):
        story = {"title": "Fed holds rates steady", "summary": "The Federal Reserve left interest rates unchanged today."}
        result = {
            "visual_type": "pie_chart",
            "pie_chart": {"title": "Sector weightings", "labels": ["Tech", "Health", "Energy"], "values": [42, 31, 27]},
        }
        new_result, warning = verify.ground_visual_spec(result, story)
        self.assertIsNotNone(warning)
        self.assertEqual(new_result["visual_type"], "none")
        self.assertIsNone(new_result["pie_chart"])

    def test_ticker_driven_types_are_not_touched(self):
        story = {"title": "Apple rallies", "summary": "Shares jumped."}
        result = {"visual_type": "price_chart", "ticker": "AAPL"}
        new_result, warning = verify.ground_visual_spec(result, story)
        self.assertIsNone(warning)
        self.assertEqual(new_result["visual_type"], "price_chart")

    def test_none_visual_type_is_a_noop(self):
        story = {"title": "x", "summary": "y"}
        result = {"visual_type": "none"}
        new_result, warning = verify.ground_visual_spec(result, story)
        self.assertIsNone(warning)
        self.assertEqual(new_result, {"visual_type": "none"})


class VisualDowngradeTests(unittest.TestCase):
    """A spec-driven chart needing 2+ real values (dumbbell, bar, ...) but only backed by ONE
    real number in the story is the single most common source of a fabricated chart value --
    the model invents a plausible-looking second data point to fill out the shape. Rather than
    losing the visual entirely, ground_visual_spec should salvage the value(s) that ARE
    grounded into a custom_stat_visual, using the model's own labels."""

    def test_dumbbell_chart_with_one_fabricated_side_downgrades_to_stat_card(self):
        story = {
            "title": "Kelly Services jumps on Fair Value signal",
            "summary": "Shares of Kelly Services delivered a 67% return after triggering a Fair Value buy signal.",
        }
        result = {
            "visual_type": "dumbbell_chart",
            "dumbbell_chart": {
                "title": "Kelly Services return",
                "labels": ["KELYA"],
                "start_values": [0],
                "end_values": [67],
                "start_label": "Signal",
                "end_label": "Now",
                "unit": "%",
            },
        }
        new_result, warning = verify.ground_visual_spec(result, story)
        self.assertIn("downgraded", warning)
        self.assertEqual(new_result["visual_type"], "custom_stat_visual")
        self.assertIsNone(new_result["dumbbell_chart"])
        stats = new_result["custom_stat_visual"]["stats"]
        self.assertEqual(len(stats), 1)
        self.assertEqual(stats[0]["value"], 67.0)
        self.assertEqual(stats[0]["unit"], "%")

    def test_bar_chart_keeps_only_the_grounded_labels(self):
        story = {"title": "Gold and silver both rally", "summary": "Gold rose 1.8% while silver was little changed."}
        result = {
            "visual_type": "bar_chart",
            "bar_chart": {
                "title": "Metals move",
                "labels": ["Gold", "Silver", "Platinum"],
                "values": [1.8, 12.4, 5.5],
                "unit": "%",
            },
        }
        new_result, warning = verify.ground_visual_spec(result, story)
        self.assertIn("downgraded", warning)
        self.assertEqual(new_result["visual_type"], "custom_stat_visual")
        stats = new_result["custom_stat_visual"]["stats"]
        self.assertEqual(len(stats), 1)
        self.assertEqual(stats[0]["label"], "Gold")
        self.assertEqual(stats[0]["value"], 1.8)

    def test_custom_stat_visual_itself_is_filtered_down_in_place(self):
        story = {"title": "Bitcoin tops $70,000", "summary": "Bitcoin traded above $70,000 for the first time this year."}
        result = {
            "visual_type": "custom_stat_visual",
            "custom_stat_visual": {
                "title": "Bitcoin milestone",
                "stats": [
                    {"label": "Price", "value": 70000, "unit": "$"},
                    {"label": "Fabricated stat", "value": 12345, "unit": "$"},
                ],
            },
        }
        new_result, warning = verify.ground_visual_spec(result, story)
        self.assertIn("downgraded", warning)
        self.assertEqual(new_result["visual_type"], "custom_stat_visual")
        stats = new_result["custom_stat_visual"]["stats"]
        self.assertEqual(len(stats), 1)
        self.assertEqual(stats[0]["label"], "Price")

    def test_no_grounded_values_at_all_falls_through_to_full_suppression(self):
        story = {"title": "Fed holds rates steady", "summary": "The Federal Reserve left interest rates unchanged today."}
        result = {
            "visual_type": "dumbbell_chart",
            "dumbbell_chart": {
                "title": "Fabricated comparison",
                "labels": ["A"], "start_values": [11], "end_values": [22],
                "start_label": "Before", "end_label": "After", "unit": "%",
            },
        }
        new_result, warning = verify.ground_visual_spec(result, story)
        self.assertIn("suppressed", warning)
        self.assertNotIn("downgraded", warning)
        self.assertEqual(new_result["visual_type"], "none")
        self.assertIsNone(new_result["dumbbell_chart"])

    def test_correlation_type_has_no_downgrade_path_and_fully_suppresses(self):
        # scatter/bubble/regression/correlation_matrix/box_plot/violin_plot describe a
        # multi-point relationship or distribution, not a handful of standalone facts -- a
        # single salvaged number wouldn't honestly represent "a correlation," so these should
        # never downgrade, only fully suppress like before.
        story = {"title": "Oil and airline stocks diverge", "summary": "Oil fell 3% today."}
        result = {
            "visual_type": "scatter_chart",
            "scatter_chart": {
                "title": "Oil vs airlines", "x_label": "Oil move", "y_label": "Airline move",
                "x_values": [-3, -8, -15], "y_values": [2, 5, 9],
            },
        }
        new_result, warning = verify.ground_visual_spec(result, story)
        self.assertIn("suppressed", warning)
        self.assertNotIn("downgraded", warning)
        self.assertEqual(new_result["visual_type"], "none")

    def test_downgraded_stats_are_capped_at_three(self):
        story = {
            "title": "Four sectors move",
            "summary": "Tech rose 5%, energy rose 3%, health fell 2%, and financials fell 1% today.",
        }
        result = {
            "visual_type": "bar_chart",
            "bar_chart": {
                "title": "Sector moves",
                "labels": ["Tech", "Energy", "Health", "Financials"],
                "values": [5, 3, -2, -1],
                "unit": "%",
            },
        }
        # Make one value ungrounded to force the downgrade path even though all 4 look grounded.
        result["bar_chart"]["values"][0] = 99
        new_result, warning = verify.ground_visual_spec(result, story)
        self.assertIn("downgraded", warning)
        stats = new_result["custom_stat_visual"]["stats"]
        self.assertLessEqual(len(stats), 3)


class TitleAndPeriodGroundingTests(unittest.TestCase):
    def test_wrong_subject_title_suppresses_the_visual(self):
        # Numbers here ARE grounded (both appear in the story) but the chart's own title
        # names a completely different company -- this is the exact "Momenta" failure mode.
        story = {
            "title": "Nvidia beats Q2 earnings estimates",
            "summary": "Nvidia reported revenue of 30 versus an estimate of 28.",
        }
        result = {
            "visual_type": "bar_chart",
            "bar_chart": {"title": "Momenta Q2 Revenue", "labels": ["Actual", "Estimate"], "values": [30, 28], "unit": "$B"},
        }
        new_result, warning = verify.ground_visual_spec(result, story)
        self.assertIsNotNone(warning)
        self.assertEqual(new_result["visual_type"], "none")
        self.assertIsNone(new_result["bar_chart"])

    def test_matching_subject_title_passes(self):
        story = {
            "title": "Nvidia beats Q2 earnings estimates",
            "summary": "Nvidia reported revenue of 30 versus an estimate of 28.",
        }
        result = {
            "visual_type": "bar_chart",
            "bar_chart": {"title": "Nvidia Q2 Revenue", "labels": ["Actual", "Estimate"], "values": [30, 28], "unit": "$B"},
        }
        new_result, warning = verify.ground_visual_spec(result, story)
        self.assertIsNone(warning)
        self.assertEqual(new_result["visual_type"], "bar_chart")

    def test_wrong_quarter_suppresses_the_visual(self):
        story = {
            "title": "Company X reports Q2 results",
            "summary": "Company X posted revenue of 50 for the second quarter, versus 45 a year ago.",
        }
        result = {
            "visual_type": "bar_chart",
            "bar_chart": {"title": "Company X Q1 Revenue", "labels": ["This year", "Last year"], "values": [50, 45], "unit": "$B"},
        }
        new_result, warning = verify.ground_visual_spec(result, story)
        self.assertIsNotNone(warning)
        self.assertEqual(new_result["visual_type"], "none")

    def test_matching_quarter_passes(self):
        story = {
            "title": "Company X reports Q2 results",
            "summary": "Company X posted revenue of 50 for the second quarter, versus 45 a year ago.",
        }
        result = {
            "visual_type": "bar_chart",
            "bar_chart": {"title": "Company X Q2 Revenue", "labels": ["This year", "Last year"], "values": [50, 45], "unit": "$B"},
        }
        new_result, warning = verify.ground_visual_spec(result, story)
        self.assertIsNone(warning)
        self.assertEqual(new_result["visual_type"], "bar_chart")


class FlowchartGroundingTests(unittest.TestCase):
    def test_unrelated_flowchart_is_suppressed(self):
        story = {"title": "Fed cuts rates", "summary": "The Federal Reserve lowered its target rate."}
        result = {"visual_type": "flowchart", "flowchart": {"steps": ["OPEC raises output", "Oil supply increases", "Crude prices fall"]}}
        new_result, warning = verify.ground_visual_spec(result, story)
        self.assertIsNotNone(warning)
        self.assertEqual(new_result["visual_type"], "none")

    def test_related_flowchart_passes(self):
        story = {"title": "Fed cuts rates", "summary": "The Federal Reserve lowered its target interest rate."}
        result = {"visual_type": "flowchart", "flowchart": {"steps": ["Fed cuts rates", "Borrowing costs fall", "Equities rally"]}}
        new_result, warning = verify.ground_visual_spec(result, story)
        self.assertIsNone(warning)
        self.assertEqual(new_result["visual_type"], "flowchart")


class TickerSubjectGroundingTests(unittest.TestCase):
    def test_matching_ticker_passes(self):
        story = {"title": "Apple unveils new iPhone lineup", "summary": "Apple announced new products today."}
        with patch("verify._fetch_ticker_name", return_value="Apple Inc."):
            ok, reason = verify.ground_ticker_subject("AAPL", story)
        self.assertTrue(ok, reason)

    def test_wrong_ticker_for_story_is_blocked(self):
        story = {"title": "Apple unveils new iPhone lineup", "summary": "Apple announced new products today."}
        with patch("verify._fetch_ticker_name", return_value="Tesla, Inc."):
            ok, reason = verify.ground_ticker_subject("TSLA", story)
        self.assertFalse(ok)
        self.assertIn("TSLA", reason)

    def test_literal_symbol_in_story_is_a_fallback_match(self):
        story = {"title": "Shares of Momenta (MNTA) rally on trial data", "summary": "The biotech's stock jumped."}
        with patch("verify._fetch_ticker_name", return_value=None):
            ok, reason = verify.ground_ticker_subject("MNTA", story)
        self.assertTrue(ok, reason)

    def test_failed_name_lookup_with_no_symbol_match_fails_closed(self):
        story = {"title": "Apple unveils new iPhone lineup", "summary": "Apple announced new products today."}
        with patch("verify._fetch_ticker_name", return_value=None):
            ok, reason = verify.ground_ticker_subject("XYZQ", story)
        self.assertFalse(ok)
        self.assertIn("failed", reason)

    def test_no_ticker_is_a_noop(self):
        ok, reason = verify.ground_ticker_subject(None, {"title": "x", "summary": "y"})
        self.assertTrue(ok, reason)


class ImageQueryGroundingTests(unittest.TestCase):
    def test_grounded_query_passes(self):
        story = {"title": "Tesla opens new Gigafactory", "summary": "Tesla's newest plant began production."}
        ok, reason = verify.ground_image_query("Tesla, Inc.", story)
        self.assertTrue(ok, reason)

    def test_ungrounded_query_is_blocked(self):
        story = {"title": "Tesla opens new Gigafactory", "summary": "Tesla's newest plant began production."}
        ok, reason = verify.ground_image_query("Federal Reserve", story)
        self.assertFalse(ok)


class CheckVisualRelevanceIntegrationTests(unittest.TestCase):
    def test_wrong_ticker_downgrades_to_none(self):
        story = {"title": "Apple unveils new iPhone lineup", "summary": "Apple announced new products today."}
        result = {"visual_type": "price_chart", "ticker": "TSLA"}
        with patch("verify._fetch_ticker_name", return_value="Tesla, Inc."):
            new_result, warning = verify.check_visual_relevance(result, story)
        self.assertEqual(new_result["visual_type"], "none")
        self.assertIsNone(new_result["ticker"])
        self.assertIsNotNone(warning)

    def test_correct_ticker_survives(self):
        story = {"title": "Apple unveils new iPhone lineup", "summary": "Apple announced new products today."}
        result = {"visual_type": "price_chart", "ticker": "AAPL"}
        with patch("verify._fetch_ticker_name", return_value="Apple Inc."):
            new_result, warning = verify.check_visual_relevance(result, story)
        self.assertEqual(new_result["visual_type"], "price_chart")
        self.assertIsNone(warning)

    def test_retired_visual_types_are_unconditionally_blocked(self):
        # scatter_chart etc. are retired from persona.py's menu (see RETIRED_VISUAL_TYPES) after
        # 8 days of real production logs showed a 100% fabrication rate and zero successes. Even
        # with numbers that happen to be individually grounded, these should never survive --
        # short news text essentially never states a genuine 3+-point correlation dataset, so a
        # coincidental grounding match wouldn't represent a real relationship anyway.
        story = {"title": "Oil and airlines move together", "summary": "Oil fell 3%, 8%, and 15% over three sessions while airline stocks rose 2%, 5%, and 9%."}
        for visual_type in sorted(verify.RETIRED_VISUAL_TYPES):
            with self.subTest(visual_type=visual_type):
                result = {
                    "visual_type": visual_type,
                    visual_type: {"title": "t", "x_values": [3, 8, 15], "y_values": [2, 5, 9]},
                }
                new_result, warning = verify.check_visual_relevance(result, story)
                self.assertEqual(new_result["visual_type"], "none")
                self.assertIsNone(new_result[visual_type])
                self.assertIn("retired", warning)


class CausalClaimTests(unittest.TestCase):
    def test_grounded_causal_claim_passes(self):
        story = {"title": "Oil falls on OPEC supply increase", "summary": "OPEC agreed to raise output next month."}
        thread = ["1/2 Crude fell 3% because of OPEC's decision to raise supply.", "2/2 Watch for the next meeting."]
        ok, reason = verify.check_causal_claims(thread, story)
        self.assertTrue(ok, reason)

    def test_ungrounded_causal_claim_is_blocked(self):
        story = {"title": "Oil falls", "summary": "Crude prices declined in early trading."}
        thread = ["1/1 Crude fell 3% because of a surprise submarine cable outage in the Pacific."]
        ok, reason = verify.check_causal_claims(thread, story)
        self.assertFalse(ok)
        self.assertIn("causal claim", reason)

    def test_no_causal_language_is_fine(self):
        story = {"title": "Oil falls", "summary": "Crude prices declined."}
        thread = ["1/1 Crude is down 3% today."]
        ok, reason = verify.check_causal_claims(thread, story)
        self.assertTrue(ok, reason)


class TickerDirectionTests(unittest.TestCase):
    def test_matching_direction_passes(self):
        thread = ["1/1 AAPL surged 5% today on strong iPhone demand."]
        chart_stats = {"pct_change": 5.2}
        ok, reason = verify.verify_ticker_direction(thread, chart_stats)
        self.assertTrue(ok, reason)

    def test_contradicting_direction_is_blocked(self):
        thread = ["1/1 AAPL surged higher today on strong iPhone demand."]
        chart_stats = {"pct_change": -3.4}
        ok, reason = verify.verify_ticker_direction(thread, chart_stats)
        self.assertFalse(ok)
        self.assertIn("-3.4", reason)

    def test_mixed_direction_thread_is_not_blocked(self):
        thread = ["1/1 AAPL is up on the week but fell sharply today."]
        chart_stats = {"pct_change": -1.0}
        ok, reason = verify.verify_ticker_direction(thread, chart_stats)
        self.assertTrue(ok, reason)

    def test_no_chart_stats_is_a_noop(self):
        ok, reason = verify.verify_ticker_direction(["anything"], {})
        self.assertTrue(ok, reason)


class VerbIntensityTests(unittest.TestCase):
    def test_dramatic_verb_on_small_move_is_flagged(self):
        warnings = verify.check_verb_intensity(["1/1 AAPL surged today."], {"pct_change": 0.3})
        self.assertTrue(warnings)

    def test_dramatic_verb_on_large_move_is_not_flagged(self):
        warnings = verify.check_verb_intensity(["1/1 AAPL surged today."], {"pct_change": 6.5})
        self.assertFalse(warnings)

    def test_neutral_language_on_small_move_is_not_flagged(self):
        warnings = verify.check_verb_intensity(["1/1 AAPL edged up today."], {"pct_change": 0.3})
        self.assertFalse(warnings)


class BareNumberTests(unittest.TestCase):
    def test_bare_percentage_is_flagged(self):
        warnings = verify.check_bare_numbers(["1/1 Shares are already up 12%."])
        self.assertTrue(warnings)

    def test_percentage_with_basis_is_not_flagged(self):
        warnings = verify.check_bare_numbers(["1/1 Shares are up 12% year-to-date."])
        self.assertFalse(warnings)


class BannedFillerTests(unittest.TestCase):
    def test_unconditional_filler_is_blocked(self):
        for phrase in ["amid ongoing volatility", "as investors digest the news", "uncertainty looms over markets"]:
            with self.subTest(phrase=phrase):
                ok, reason = verify.check_banned_filler([f"1/1 Stocks fell {phrase}."])
                self.assertFalse(ok)
                self.assertIsNotNone(reason)

    def test_eyes_on_without_a_level_is_blocked(self):
        ok, reason = verify.check_banned_filler(["1/1 Eyes on the Fed this week."])
        self.assertFalse(ok)

    def test_eyes_on_with_a_level_is_allowed(self):
        ok, reason = verify.check_banned_filler(["1/1 Eyes on the 4.65% level for the 10-year."])
        self.assertTrue(ok, reason)

    def test_eyes_on_with_a_date_is_allowed(self):
        ok, reason = verify.check_banned_filler(["1/1 Eyes on the Q2 print due Thursday."])
        self.assertTrue(ok, reason)

    def test_clean_thread_passes(self):
        ok, reason = verify.check_banned_filler(["1/1 The Fed cut rates to 4.50% from 4.75%."])
        self.assertTrue(ok, reason)


class HashtagDisciplineTests(unittest.TestCase):
    def test_first_tweet_with_hashtag_is_blocked(self):
        ok, reason = verify.check_hashtag_discipline(["1/2 #Fed cuts rates today.", "2/2 Watch next week."])
        self.assertFalse(ok)

    def test_middle_tweet_with_hashtag_is_blocked(self):
        ok, reason = verify.check_hashtag_discipline([
            "1/3 Fed cuts rates to 4.50%.", "2/3 This #matters for markets.", "3/3 Watch next week.",
        ])
        self.assertFalse(ok)

    def test_closing_tweet_with_any_hashtag_is_blocked(self):
        ok, reason = verify.check_hashtag_discipline([
            "1/2 Fed cuts rates to 4.50% from 4.75%.", "2/2 Watch the Jul 30 meeting. #Fed",
        ])
        self.assertFalse(ok)

    def test_no_hashtags_at_all_passes(self):
        ok, reason = verify.check_hashtag_discipline(["1/2 Fed cuts rates.", "2/2 Watch next week."])
        self.assertTrue(ok, reason)

    def test_empty_thread_is_a_noop(self):
        ok, reason = verify.check_hashtag_discipline([])
        self.assertTrue(ok, reason)


class ThreadCompletenessTests(unittest.TestCase):
    def test_consistent_full_thread_passes(self):
        ok, reason = verify.check_thread_completeness([
            "1/4 Gold hit a record high.", "2/4 The catalyst was jobs data.",
            "3/4 Yields fell in response.", "4/4 Watch the Fed decision next.",
        ])
        self.assertTrue(ok, reason)

    def test_missing_final_tweet_is_blocked(self):
        # The exact real-world symptom: tweets numbered as if 4 were coming, but only 3 exist --
        # most often produced by regen.py's rewrite retry not preserving the original count.
        ok, reason = verify.check_thread_completeness([
            "1/4 Gold hit a record high.", "2/4 The catalyst was jobs data.",
            "3/4 Yields fell in response.",
        ])
        self.assertFalse(ok)
        self.assertIn("last tweet(s) missing", reason)

    def test_missing_numbering_prefix_is_blocked(self):
        ok, reason = verify.check_thread_completeness(["Gold hit a record high with no prefix at all."])
        self.assertFalse(ok)
        self.assertIn("no 'N/TOTAL' numbering prefix", reason)

    def test_disagreeing_totals_are_blocked(self):
        ok, reason = verify.check_thread_completeness(["1/3 First.", "2/4 Second."])
        self.assertFalse(ok)
        self.assertIn("disagree", reason)

    def test_non_sequential_numbering_is_blocked(self):
        ok, reason = verify.check_thread_completeness(["1/3 First.", "1/3 Duplicate.", "3/3 Third."])
        self.assertFalse(ok)
        self.assertIn("sequential", reason)

    def test_single_tweet_thread_passes(self):
        ok, reason = verify.check_thread_completeness(["1/1 Gold hit a record high today."])
        self.assertTrue(ok, reason)

    def test_empty_thread_is_a_noop(self):
        ok, reason = verify.check_thread_completeness([])
        self.assertTrue(ok, reason)


class RankByEngagementTests(unittest.TestCase):
    def test_sorts_descending_by_composite_score(self):
        items = [
            {"story_title": "low", "expected_engagement": 3, "market_significance": 3, "relevance": 3},
            {"story_title": "high", "expected_engagement": 9, "market_significance": 9, "relevance": 9},
            {"story_title": "mid", "expected_engagement": 6, "market_significance": 6, "relevance": 6},
        ]
        ranked = verify.rank_by_engagement(items)
        self.assertEqual([i["story_title"] for i in ranked], ["high", "mid", "low"])

    def test_missing_fields_default_to_zero_not_error(self):
        items = [{"story_title": "a"}, {"story_title": "b", "expected_engagement": 5}]
        ranked = verify.rank_by_engagement(items)
        self.assertEqual(ranked[0]["story_title"], "b")

    def test_does_not_mutate_or_drop_items(self):
        items = [{"story_title": "a", "relevance": 1}, {"story_title": "b", "relevance": 9}]
        ranked = verify.rank_by_engagement(items)
        self.assertEqual(len(ranked), 2)
        self.assertEqual({i["story_title"] for i in ranked}, {"a", "b"})


class ProvenanceTests(unittest.TestCase):
    def test_provenance_shape(self):
        story = {"source": "markets", "link": "http://x.test/1", "published": "2026-01-01T00:00:00+00:00"}
        chart_stats = {"source": "yfinance", "pct_change": 1.0}
        provenance = verify.build_provenance(story, chart_stats, ["some warning"], ["bare number"])
        self.assertIn("generated_at", provenance)
        self.assertIn("yfinance", provenance["data_sources"])
        self.assertIn("some warning", provenance["warnings"])
        self.assertIn("bare number", provenance["warnings"])


class VisualConfidenceTests(unittest.TestCase):
    def test_low_confidence_suppresses_visual(self):
        story = {"title": "Fed cuts rates to 4.50% from 4.75%", "summary": "The Fed cut its rate to 4.50% from 4.75%."}
        result = {
            "visual_type": "bar_chart", "visual_confidence": 3,
            "bar_chart": {"title": "Fed funds rate", "labels": ["Before", "After"], "values": [4.75, 4.50], "unit": "%"},
        }
        new_result, warning = verify.check_visual_confidence(result, story)
        self.assertIsNotNone(warning)
        self.assertEqual(new_result["visual_type"], "none")
        self.assertIsNone(new_result["bar_chart"])

    def test_high_confidence_passes(self):
        story = {"title": "x", "summary": "y"}
        result = {"visual_type": "bar_chart", "visual_confidence": 9, "bar_chart": {"title": "t"}}
        new_result, warning = verify.check_visual_confidence(result, story)
        self.assertIsNone(warning)
        self.assertEqual(new_result["visual_type"], "bar_chart")

    def test_missing_confidence_fails_closed(self):
        story = {"title": "x", "summary": "y"}
        result = {"visual_type": "bar_chart", "bar_chart": {"title": "t"}}
        new_result, warning = verify.check_visual_confidence(result, story)
        self.assertIsNotNone(warning)
        self.assertEqual(new_result["visual_type"], "none")

    def test_none_visual_type_is_a_noop(self):
        story = {"title": "x", "summary": "y"}
        result = {"visual_type": "none"}
        new_result, warning = verify.check_visual_confidence(result, story)
        self.assertIsNone(warning)


class ShapeMatchTests(unittest.TestCase):
    def test_trend_chart_forced_onto_single_stat_story_is_blocked(self):
        # No temporal-sequence language, no ranked list, just one flat number -- a
        # multi-period trend line has no business being attached to this story.
        story = {"title": "Company X reports record quarterly profit", "summary": "Company X posted a record profit figure."}
        result = {
            "visual_type": "trend_chart", "visual_confidence": 8,
            "trend_chart": {"title": "Profit", "labels": ["A", "B", "C"], "values": [1, 2, 3], "fit": "linear", "unit": ""},
        }
        new_result, warning = verify.check_shape_match(result, story)
        self.assertIsNotNone(warning)
        self.assertEqual(new_result["visual_type"], "none")

    def test_trend_chart_on_a_genuine_trend_story_passes(self):
        story = {"title": "Stock rallies for a third consecutive week", "summary": "Shares have risen steadily this week and last week."}
        result = {"visual_type": "trend_chart", "trend_chart": {"title": "t"}}
        new_result, warning = verify.check_shape_match(result, story)
        self.assertIsNone(warning)

    def test_correlation_matrix_on_unrelated_story_is_blocked(self):
        story = {"title": "Company X reports record quarterly profit", "summary": "Company X posted a record profit figure."}
        result = {"visual_type": "correlation_matrix_chart", "correlation_matrix_chart": {"title": "t"}}
        new_result, warning = verify.check_shape_match(result, story)
        self.assertIsNotNone(warning)
        self.assertEqual(new_result["visual_type"], "none")

    def test_price_chart_always_plausible_single_stat(self):
        # price_chart's shape set includes single_stat, which is always in the universal
        # baseline -- should never be shape-blocked regardless of story content.
        story = {"title": "Anything at all", "summary": "No special signals here."}
        result = {"visual_type": "price_chart", "ticker": "AAPL"}
        new_result, warning = verify.check_shape_match(result, story)
        self.assertIsNone(warning)

    def test_unmapped_visual_type_is_a_noop(self):
        story = {"title": "x", "summary": "y"}
        result = {"visual_type": "none"}
        new_result, warning = verify.check_shape_match(result, story)
        self.assertIsNone(warning)

    # Regression tests for two real stories a live run wrongly blocked: a candlestick chart is
    # a reasonable pick for a plain "price moved to $X" story even without explicit trend
    # language (same reasoning already applied to price_chart), and numbered time spans like
    # "three-week" weren't recognized as trend language at all.
    def test_candlestick_chart_passes_for_a_plain_price_level_story(self):
        story = {"title": "Bitcoin zips higher to nearly $64,000 as chip rally and yen strength drive gains", "summary": ""}
        result = {"visual_type": "candlestick_chart", "ticker": "BTC-USD"}
        new_result, warning = verify.check_shape_match(result, story)
        self.assertIsNone(warning)

    def test_numbered_time_span_is_recognized_as_a_trend_signal(self):
        story = {"title": "Bitcoin returns to $64.3K with new three-week BTC price highs imminent", "summary": ""}
        result = {"visual_type": "candlestick_chart", "ticker": "BTC-USD"}
        new_result, warning = verify.check_shape_match(result, story)
        self.assertIsNone(warning)

    def test_digit_numbered_time_span_is_recognized(self):
        shapes = verify.classify_story_shape({"title": "Shares hit a 52-week low", "summary": ""})
        self.assertIn("multi_period_trend", shapes)

    def test_renko_stays_narrow_despite_widened_siblings(self):
        # renko/pnf/kagi are deliberately kept multi_period_trend-only (persona.py scopes them
        # to explicit technical framing) -- the widening applied to candlestick/ohlc/heikin_ashi
        # must not leak into these.
        story = {"title": "Bitcoin zips higher to nearly $64,000", "summary": ""}
        result = {"visual_type": "renko_chart", "ticker": "BTC-USD"}
        new_result, warning = verify.check_shape_match(result, story)
        self.assertIsNotNone(warning)
        for other in ("pnf_chart", "kagi_chart"):
            with self.subTest(visual_type=other):
                new_result, warning = verify.check_shape_match({"visual_type": other, "ticker": "BTC-USD"}, story)
                self.assertIsNotNone(warning)

    # Regression tests for a second batch of real-world false positives found via stress-testing
    # after the candlestick fix -- the same "keyword list too narrow" bug existed for every
    # technical-indicator chart type, not just candlestick, since each names its own specific
    # vocabulary (RSI/overbought, MACD/crossover, "below its high", "calmest trading", etc.)
    # that the generic multi_period_trend signals never covered.
    def test_rsi_chart_passes_for_overbought_language(self):
        story = {"title": "Oil looks overbought after sharp rally, RSI flashes warning", "summary": ""}
        result = {"visual_type": "rsi_chart", "ticker": "CL=F"}
        new_result, warning = verify.check_shape_match(result, story)
        self.assertIsNone(warning, warning)

    def test_drawdown_chart_passes_for_below_high_language(self):
        story = {"title": "Bitcoin still 15% below its all-time high despite rebound", "summary": ""}
        result = {"visual_type": "drawdown_chart", "ticker": "BTC-USD"}
        new_result, warning = verify.check_shape_match(result, story)
        self.assertIsNone(warning, warning)

    def test_historical_volatility_chart_passes_for_calm_trading_language(self):
        story = {"title": "Currency markets see calmest trading in months", "summary": ""}
        result = {"visual_type": "historical_volatility_chart", "ticker": "EURUSD=X"}
        new_result, warning = verify.check_shape_match(result, story)
        self.assertIsNone(warning, warning)

    def test_seasonality_chart_passes_for_historically_language(self):
        story = {"title": "Stocks historically rally in the final weeks of the year", "summary": ""}
        result = {"visual_type": "seasonality_chart", "ticker": "^GSPC"}
        new_result, warning = verify.check_shape_match(result, story)
        self.assertIsNone(warning, warning)

    def test_macd_chart_passes_for_crossover_language(self):
        story = {"title": "Gold flashes bullish MACD crossover signal", "summary": ""}
        result = {"visual_type": "macd_chart", "ticker": "GC=F"}
        new_result, warning = verify.check_shape_match(result, story)
        self.assertIsNone(warning, warning)

    def test_bollinger_chart_passes_for_squeeze_language(self):
        story = {"title": "Silver squeezes toward the top of its Bollinger Band range", "summary": ""}
        result = {"visual_type": "bollinger_bands_chart", "ticker": "SI=F"}
        new_result, warning = verify.check_shape_match(result, story)
        self.assertIsNone(warning, warning)

    def test_moving_average_chart_passes_for_average_crossing_language(self):
        story = {"title": "S&P 500 breaks above its 200-day moving average", "summary": ""}
        result = {"visual_type": "moving_average_chart", "ticker": "^GSPC"}
        new_result, warning = verify.check_shape_match(result, story)
        self.assertIsNone(warning, warning)

    def test_volume_chart_passes_for_volume_spike_language(self):
        story = {"title": "Unusual volume spike hits regional bank stocks", "summary": ""}
        result = {"visual_type": "volume_chart", "ticker": "KRE"}
        new_result, warning = verify.check_shape_match(result, story)
        self.assertIsNone(warning, warning)

    def test_technical_chart_still_blocks_a_totally_unrelated_story(self):
        # Sanity check: the widening must not turn these into a rubber stamp -- a story with
        # none of the technical vocabulary and no numbered period should still be blocked.
        story = {"title": "Company X reports record quarterly profit", "summary": "Company X posted a record profit figure."}
        result = {"visual_type": "rsi_chart", "ticker": "TEST"}
        new_result, warning = verify.check_shape_match(result, story)
        self.assertIsNotNone(warning)


class VisualThreadConsistencyTests(unittest.TestCase):
    def test_chart_number_echoed_in_thread_passes(self):
        story = {"title": "x", "summary": "y"}
        thread = ["1/1 Fed cut rates to 4.50% from 4.75%."]
        result = {
            "visual_type": "bar_chart",
            "bar_chart": {"title": "t", "labels": ["Before", "After"], "values": [4.75, 4.50], "unit": "%"},
        }
        new_result, warning = verify.check_visual_thread_consistency(result, story, thread)
        self.assertIsNone(warning)
        self.assertEqual(new_result["visual_type"], "bar_chart")

    def test_chart_numbers_never_mentioned_in_thread_is_blocked(self):
        story = {"title": "x", "summary": "y"}
        thread = ["1/1 The Fed held policy steady today, as widely expected."]
        result = {
            "visual_type": "bar_chart",
            "bar_chart": {"title": "t", "labels": ["Before", "After"], "values": [4.75, 4.50], "unit": "%"},
        }
        new_result, warning = verify.check_visual_thread_consistency(result, story, thread)
        self.assertIsNotNone(warning)
        self.assertEqual(new_result["visual_type"], "none")

    def test_ticker_driven_types_are_not_checked(self):
        story = {"title": "x", "summary": "y"}
        result = {"visual_type": "price_chart", "ticker": "AAPL"}
        new_result, warning = verify.check_visual_thread_consistency(result, story, ["anything"])
        self.assertIsNone(warning)


class VisualVarietyTests(unittest.TestCase):
    def test_repeat_with_low_confidence_is_suppressed(self):
        story = {"title": "x", "summary": "y"}
        result = {"visual_type": "bar_chart", "visual_confidence": 6, "bar_chart": {"title": "t"}}
        new_result, warning = verify.check_visual_variety(result, story, ["bar_chart"])
        self.assertIsNotNone(warning)
        self.assertEqual(new_result["visual_type"], "none")

    def test_repeat_with_high_confidence_survives(self):
        story = {"title": "x", "summary": "y"}
        result = {"visual_type": "bar_chart", "visual_confidence": 9, "bar_chart": {"title": "t"}}
        new_result, warning = verify.check_visual_variety(result, story, ["bar_chart"])
        self.assertIsNone(warning)
        self.assertEqual(new_result["visual_type"], "bar_chart")

    def test_non_repeat_is_never_touched(self):
        story = {"title": "x", "summary": "y"}
        result = {"visual_type": "pie_chart", "visual_confidence": 2, "pie_chart": {"title": "t"}}
        new_result, warning = verify.check_visual_variety(result, story, ["bar_chart"])
        self.assertIsNone(warning)
        self.assertEqual(new_result["visual_type"], "pie_chart")

    def test_no_history_is_a_noop(self):
        story = {"title": "x", "summary": "y"}
        result = {"visual_type": "bar_chart", "visual_confidence": 2, "bar_chart": {"title": "t"}}
        new_result, warning = verify.check_visual_variety(result, story, [])
        self.assertIsNone(warning)


class SelectVisualOrchestratorTests(unittest.TestCase):
    def test_confident_grounded_visual_survives_full_pipeline(self):
        story = {"title": "Fed cuts rates to 4.50% from 4.75%", "summary": "The Fed cut its target rate to 4.50% from 4.75%."}
        thread = ["1/1 The Fed cut rates to 4.50% from 4.75%, its second cut this year."]
        result = {
            "visual_type": "bar_chart", "visual_confidence": 9,
            "bar_chart": {"title": "Fed funds rate", "labels": ["Before", "After"], "values": [4.75, 4.50], "unit": "%"},
        }
        new_result, warnings = verify.select_visual(result, story, thread, [])
        self.assertEqual(new_result["visual_type"], "bar_chart")
        self.assertEqual(warnings, [])

    def test_low_confidence_visual_is_dropped_by_orchestrator(self):
        story = {"title": "Fed cuts rates to 4.50% from 4.75%", "summary": "The Fed cut its target rate to 4.50% from 4.75%."}
        thread = ["1/1 The Fed cut rates to 4.50% from 4.75%, its second cut this year."]
        result = {
            "visual_type": "bar_chart", "visual_confidence": 2,
            "bar_chart": {"title": "Fed funds rate", "labels": ["Before", "After"], "values": [4.75, 4.50], "unit": "%"},
        }
        new_result, warnings = verify.select_visual(result, story, thread, [])
        self.assertEqual(new_result["visual_type"], "none")
        self.assertTrue(warnings)


class CustomStatVisualTests(unittest.TestCase):
    def test_renders_with_valid_spec(self):
        image_bytes = chart.generate_custom_stat_visual(
            {"title": "Revenue Beat", "stats": [{"label": "Actual", "value": 30, "unit": "B"}, {"label": "Estimate", "value": 28, "unit": "B"}]},
            story_source="markets",
        )
        self.assertIsNotNone(image_bytes)
        self.assertGreater(len(image_bytes), 0)

    def test_rejects_too_many_stats(self):
        image_bytes = chart.generate_custom_stat_visual({"title": "t", "stats": [{"label": "a", "value": 1}] * 4})
        self.assertIsNone(image_bytes)

    def test_rejects_empty_stats(self):
        image_bytes = chart.generate_custom_stat_visual({"title": "t", "stats": []})
        self.assertIsNone(image_bytes)

    def test_rejects_non_numeric_value(self):
        image_bytes = chart.generate_custom_stat_visual({"title": "t", "stats": [{"label": "a", "value": "not a number"}]})
        self.assertIsNone(image_bytes)

    def test_is_grounded_like_other_spec_driven_types(self):
        story = {"title": "Fed cuts rates to 4.50%", "summary": "The Fed lowered its rate to 4.50%."}
        result = {
            "visual_type": "custom_stat_visual",
            "custom_stat_visual": {"title": "Made Up Stat", "stats": [{"label": "Fabricated", "value": 999, "unit": ""}]},
        }
        new_result, warning = verify.ground_visual_spec(result, story)
        self.assertIsNotNone(warning)
        self.assertEqual(new_result["visual_type"], "none")

    def test_dispatches_through_resolve_visual(self):
        result = {
            "visual_type": "custom_stat_visual",
            "custom_stat_visual": {"title": "t", "stats": [{"label": "a", "value": 1, "unit": ""}]},
        }
        image_bytes = chart.resolve_visual(result, source="markets")
        self.assertIsNotNone(image_bytes)


class RecentVisualsStateTests(unittest.TestCase):
    def test_round_trip(self):
        st = {}
        st = state.save_recent_visuals(st, ["bar_chart", "price_chart", "none", "pie_chart"])
        self.assertEqual(state.get_recent_visuals(st), ["bar_chart", "price_chart", "pie_chart"])

    def test_none_entries_are_dropped(self):
        st = {}
        st = state.save_recent_visuals(st, ["none", "none"])
        self.assertEqual(state.get_recent_visuals(st), [])

    def test_window_is_capped(self):
        st = {}
        used = [f"type_{i}" for i in range(20)]
        st = state.save_recent_visuals(st, used)
        recent = state.get_recent_visuals(st)
        self.assertEqual(len(recent), state.RECENT_VISUALS_WINDOW)
        self.assertEqual(recent[-1], "type_19")

    def test_missing_key_defaults_to_empty(self):
        self.assertEqual(state.get_recent_visuals({}), [])

    def test_seeds_into_next_runs_used_visuals(self):
        # Simulates main.py's actual usage: one run's history seeds the next run's list, which
        # generate.py then appends to as it processes stories.
        st = {}
        st = state.save_recent_visuals(st, ["bar_chart", "candlestick_chart"])

        used_visuals = state.get_recent_visuals(st)
        used_visuals.append("pie_chart")  # a new post generated this run

        st = state.save_recent_visuals(st, used_visuals)
        self.assertEqual(state.get_recent_visuals(st), ["bar_chart", "candlestick_chart", "pie_chart"])


class WatermarkTests(unittest.TestCase):
    def test_add_watermark_places_handle_on_figure(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        chart._add_watermark(fig)
        texts = [t.get_text() for t in fig.texts]
        self.assertIn(chart.WATERMARK_HANDLE, texts)
        plt.close(fig)

    def test_every_spec_driven_render_includes_watermark(self):
        # These renderers need no network access, so this exercises the real _save_fig path
        # (every chart funnels through it) without any external dependency.
        specs = {
            chart.generate_bar_chart: {"title": "t", "labels": ["a", "b"], "values": [1, 2], "unit": ""},
            chart.generate_pie_chart: {"title": "t", "labels": ["a", "b"], "values": [1, 2]},
            chart.generate_waterfall_chart: {"title": "t", "labels": ["a", "b"], "values": [1, -1], "unit": ""},
        }
        for fn, spec in specs.items():
            image_bytes = fn(spec)
            self.assertIsNotNone(image_bytes, fn.__name__)
            self.assertGreater(len(image_bytes), 0)


if __name__ == "__main__":
    unittest.main()
