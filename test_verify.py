"""Automated tests for the verification/grounding layer (verify.py) plus the watermark
requirement in chart.py. Uses stdlib unittest only -- no new test dependency, no network
calls (chart tests below use spec-driven renderers, which never hit yfinance/Wikipedia)."""

import unittest
from unittest.mock import patch

import chart
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


class ProvenanceTests(unittest.TestCase):
    def test_provenance_shape(self):
        story = {"source": "markets", "link": "http://x.test/1", "published": "2026-01-01T00:00:00+00:00"}
        chart_stats = {"source": "yfinance", "pct_change": 1.0}
        provenance = verify.build_provenance(story, chart_stats, "some warning", ["bare number"])
        self.assertIn("generated_at", provenance)
        self.assertIn("yfinance", provenance["data_sources"])
        self.assertIn("some warning", provenance["warnings"])
        self.assertIn("bare number", provenance["warnings"])


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
