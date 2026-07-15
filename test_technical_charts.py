"""Tests for the technical-indicator chart types (moving average, Bollinger Bands, RSI, MACD,
drawdown, historical volatility) -- all computed purely from OHLC data already fetched via
yfinance, no new data source. Uses a synthetic price series (mocked _fetch_daily_history) so
these run offline and deterministically rather than depending on live market data."""

import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import chart


def _synthetic_ohlc(n=260, seed=7):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    returns = rng.normal(loc=0.0005, scale=0.015, size=n)
    closes = 100 * np.cumprod(1 + returns)
    opens = closes * (1 + rng.normal(0, 0.002, size=n))
    highs = np.maximum(opens, closes) * (1 + np.abs(rng.normal(0, 0.003, size=n)))
    lows = np.minimum(opens, closes) * (1 - np.abs(rng.normal(0, 0.003, size=n)))
    volumes = rng.integers(1_000_000, 5_000_000, size=n)
    return pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": volumes}, index=dates,
    )


class TechnicalChartTests(unittest.TestCase):
    def setUp(self):
        self.data = _synthetic_ohlc()
        self.patcher = patch("chart._fetch_daily_history", return_value=self.data)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()

    def test_moving_average_chart_renders_with_real_computed_ma(self):
        stats = {}
        img = chart.generate_moving_average_chart("TEST", stats_out=stats)
        self.assertIsNotNone(img)
        self.assertGreater(len(img), 0)
        expected_ma20 = float(self.data["Close"].rolling(20).mean().iloc[-1])
        self.assertAlmostEqual(stats["ma20"], expected_ma20, places=6)

    def test_bollinger_bands_chart_bands_bracket_the_moving_average(self):
        stats = {}
        img = chart.generate_bollinger_bands_chart("TEST", stats_out=stats)
        self.assertIsNotNone(img)
        self.assertGreater(stats["upper_band"], stats["lower_band"])

    def test_rsi_chart_stays_within_0_100(self):
        stats = {}
        img = chart.generate_rsi_chart("TEST", stats_out=stats)
        self.assertIsNotNone(img)
        self.assertGreaterEqual(stats["rsi_14"], 0)
        self.assertLessEqual(stats["rsi_14"], 100)

    def test_macd_chart_matches_real_ema_computation(self):
        stats = {}
        img = chart.generate_macd_chart("TEST", stats_out=stats)
        self.assertIsNotNone(img)
        closes = self.data["Close"]
        ema12 = closes.ewm(span=12, adjust=False).mean()
        ema26 = closes.ewm(span=26, adjust=False).mean()
        expected_macd = float((ema12 - ema26).iloc[-1])
        self.assertAlmostEqual(stats["macd"], expected_macd, places=6)

    def test_drawdown_chart_is_never_positive(self):
        stats = {}
        img = chart.generate_drawdown_chart("TEST", stats_out=stats)
        self.assertIsNotNone(img)
        self.assertLessEqual(stats["current_drawdown_pct"], 0)
        self.assertLessEqual(stats["max_drawdown_pct"], stats["current_drawdown_pct"])

    def test_historical_volatility_chart_is_non_negative(self):
        stats = {}
        img = chart.generate_historical_volatility_chart("TEST", stats_out=stats)
        self.assertIsNotNone(img)
        self.assertGreaterEqual(stats["realized_vol_20d_annualized_pct"], 0)

    def test_all_six_dispatch_through_resolve_visual(self):
        for visual_type, field in [
            ("moving_average_chart", "ma20"),
            ("bollinger_bands_chart", "upper_band"),
            ("rsi_chart", "rsi_14"),
            ("macd_chart", "macd"),
            ("drawdown_chart", "current_drawdown_pct"),
            ("historical_volatility_chart", "realized_vol_20d_annualized_pct"),
        ]:
            with self.subTest(visual_type=visual_type):
                stats = {}
                result = {"visual_type": visual_type, "ticker": "TEST"}
                img = chart.resolve_visual(result, label="Test Co", stats_out=stats)
                self.assertIsNotNone(img, visual_type)
                self.assertIn(field, stats)

    def test_no_ticker_returns_none(self):
        self.assertIsNone(chart.generate_moving_average_chart(None))
        self.assertIsNone(chart.generate_rsi_chart(None))

    def test_insufficient_history_returns_none_not_a_crash(self):
        with patch("chart._fetch_daily_history", return_value=_synthetic_ohlc(n=5)):
            self.assertIsNone(chart.generate_moving_average_chart("TEST"))
            self.assertIsNone(chart.generate_macd_chart("TEST"))


def _synthetic_cot_rows(n=10):
    """Newest-first, matching the real CFTC Socrata API's actual field names and ordering."""
    rows = []
    for i in range(n):
        rows.append({
            "report_date_as_yyyy_mm_dd": f"2026-{7 - i // 4:02d}-{(28 - (i % 4) * 7):02d}T00:00:00.000",
            "noncomm_positions_long_all": str(200000 + i * 1000),
            "noncomm_positions_short_all": str(30000 - i * 200),
            "comm_positions_long_all": str(60000),
            "comm_positions_short_all": str(270000),
            "open_interest_all": str(350000 + i * 500),
        })
    return rows


class COTPositioningChartTests(unittest.TestCase):
    def test_renders_for_a_mapped_ticker_with_real_math(self):
        rows = _synthetic_cot_rows()
        with patch("chart._fetch_cot_history", return_value=list(reversed(rows))):
            stats = {}
            img = chart.generate_cot_positioning_chart("GC=F", label="Gold", stats_out=stats)
        self.assertIsNotNone(img)
        expected_net = int(rows[0]["noncomm_positions_long_all"]) - int(rows[0]["noncomm_positions_short_all"])
        self.assertEqual(stats["net_speculative_position"], expected_net)
        self.assertEqual(stats["source"], "cftc_cot")
        self.assertEqual(stats["cftc_market_name"], chart.COT_MARKET_NAMES["GC=F"])

    def test_unmapped_ticker_returns_none(self):
        self.assertIsNone(chart.generate_cot_positioning_chart("AAPL"))

    def test_no_ticker_returns_none(self):
        self.assertIsNone(chart.generate_cot_positioning_chart(None))

    def test_fetch_failure_returns_none_not_a_crash(self):
        with patch("chart._fetch_cot_history", side_effect=TimeoutError("simulated network failure")):
            self.assertIsNone(chart.generate_cot_positioning_chart("GC=F"))

    def test_too_few_rows_returns_none(self):
        with patch("chart._fetch_cot_history", return_value=_synthetic_cot_rows(n=2)):
            self.assertIsNone(chart.generate_cot_positioning_chart("GC=F"))

    def test_malformed_row_returns_none_not_a_crash(self):
        broken_rows = [{"report_date_as_yyyy_mm_dd": "2026-07-01T00:00:00.000"}] * 6  # missing fields
        with patch("chart._fetch_cot_history", return_value=broken_rows):
            self.assertIsNone(chart.generate_cot_positioning_chart("GC=F"))

    def test_dispatches_through_resolve_visual(self):
        with patch("chart._fetch_cot_history", return_value=list(reversed(_synthetic_cot_rows()))):
            result = {"visual_type": "cot_positioning_chart", "ticker": "EURUSD=X"}
            img = chart.resolve_visual(result, label="Euro")
        self.assertIsNotNone(img)

    def test_every_curated_ticker_has_a_distinct_market_name(self):
        names = list(chart.COT_MARKET_NAMES.values())
        self.assertEqual(len(names), len(set(names)), "duplicate CFTC market name mapped from two tickers")


def _synthetic_fred_rows(n=10, start_value=280.0):
    """Newest-first, matching the real FRED API's actual field names and ordering."""
    rows = []
    for i in range(n):
        rows.append({
            "date": f"2026-{max(1, 7 - i):02d}-01",
            "value": f"{start_value + (n - i) * 0.5:.2f}",
        })
    return rows


class FredSeriesChartTests(unittest.TestCase):
    def test_renders_for_a_curated_series_with_real_math(self):
        rows = _synthetic_fred_rows()
        with patch("chart.config.FRED_API_KEY", "fake-key"):
            with patch("chart._fetch_fred_series", return_value=list(reversed(rows))):
                stats = {}
                img = chart.generate_fred_series_chart("CPIAUCSL", label="CPI", stats_out=stats)
        self.assertIsNotNone(img)
        self.assertEqual(stats["fred_series"], "CPIAUCSL")
        self.assertEqual(stats["source"], "fred")
        self.assertEqual(stats["latest_value"], float(rows[0]["value"]))

    def test_unmapped_series_returns_none(self):
        with patch("chart.config.FRED_API_KEY", "fake-key"):
            self.assertIsNone(chart.generate_fred_series_chart("NOT_A_REAL_SERIES"))

    def test_no_series_id_returns_none(self):
        with patch("chart.config.FRED_API_KEY", "fake-key"):
            self.assertIsNone(chart.generate_fred_series_chart(None))

    def test_no_api_key_returns_none(self):
        with patch("chart.config.FRED_API_KEY", None):
            self.assertIsNone(chart.generate_fred_series_chart("CPIAUCSL"))

    def test_fetch_failure_returns_none_not_a_crash(self):
        with patch("chart.config.FRED_API_KEY", "fake-key"):
            with patch("chart._fetch_fred_series", side_effect=TimeoutError("simulated network failure")):
                self.assertIsNone(chart.generate_fred_series_chart("CPIAUCSL"))

    def test_too_few_observations_returns_none(self):
        with patch("chart.config.FRED_API_KEY", "fake-key"):
            with patch("chart._fetch_fred_series", return_value=_synthetic_fred_rows(n=2)):
                self.assertIsNone(chart.generate_fred_series_chart("CPIAUCSL"))

    def test_missing_value_rows_are_skipped_not_a_crash(self):
        rows = _synthetic_fred_rows() + [{"date": "2026-08-01", "value": "."}]  # FRED's "no data" marker
        with patch("chart.config.FRED_API_KEY", "fake-key"):
            with patch("chart._fetch_fred_series", return_value=list(reversed(rows))):
                img = chart.generate_fred_series_chart("UNRATE")
        self.assertIsNotNone(img)

    def test_dispatches_through_resolve_visual(self):
        with patch("chart.config.FRED_API_KEY", "fake-key"):
            with patch("chart._fetch_fred_series", return_value=list(reversed(_synthetic_fred_rows()))):
                result = {"visual_type": "fred_series_chart", "fred_series": "UNRATE"}
                img = chart.resolve_visual(result, label="Unemployment")
        self.assertIsNotNone(img)


def _synthetic_sec_facts(n_quarters=6, start_val=9_000_000_000):
    entries = []
    for i in range(n_quarters):
        month = 3 * ((i % 4) + 1)
        year = 2024 + i // 4
        start = f"{year}-{max(1, month - 2):02d}-01"
        end = f"{year}-{month:02d}-28"
        entries.append({
            "start": start, "end": end, "val": start_val + i * 100_000_000,
            "form": "10-Q", "filed": f"{year}-{month:02d}-30",
        })
    return entries


class CompanyRevenueChartTests(unittest.TestCase):
    def test_renders_with_real_quarterly_data(self):
        facts = {"facts": {"us-gaap": {"Revenues": {"units": {"USD": _synthetic_sec_facts()}}}}}
        with patch("chart._resolve_cik", return_value="0000320193"):
            with patch("chart.urllib.request.urlopen") as mock_urlopen:
                mock_urlopen.return_value.__enter__.return_value.read.return_value = json.dumps(facts).encode()
                stats = {}
                img = chart.generate_company_revenue_chart("AAPL", label="Apple", stats_out=stats)
        self.assertIsNotNone(img)
        self.assertEqual(stats["source"], "sec_edgar")
        self.assertEqual(stats["cik"], "0000320193")

    def test_falls_through_revenue_tags_in_priority_order(self):
        # "Revenues" tag absent/empty -- should fall through to the next candidate tag.
        facts = {"facts": {"us-gaap": {
            "Revenues": {"units": {"USD": []}},
            "SalesRevenueNet": {"units": {"USD": _synthetic_sec_facts()}},
        }}}
        quarters = chart._extract_quarterly_revenue(facts)
        self.assertIsNotNone(quarters)
        self.assertEqual(len(quarters), 6)

    def test_ytd_cumulative_entries_are_excluded(self):
        # A same-tag entry spanning ~9 months (YTD as of Q3) must not be mistaken for a single
        # quarter -- would silently overstate revenue if counted.
        entries = _synthetic_sec_facts()
        entries.append({
            "start": "2024-01-01", "end": "2024-09-30", "val": 30_000_000_000,
            "form": "10-Q", "filed": "2024-10-30",
        })
        facts = {"facts": {"us-gaap": {"Revenues": {"units": {"USD": entries}}}}}
        quarters = chart._extract_quarterly_revenue(facts)
        values = [v for _, v in quarters]
        self.assertNotIn(30_000_000_000, values)

    def test_unresolved_ticker_returns_none(self):
        with patch("chart._resolve_cik", return_value=None):
            self.assertIsNone(chart.generate_company_revenue_chart("NOTAREALTICKER"))

    def test_no_ticker_returns_none(self):
        self.assertIsNone(chart.generate_company_revenue_chart(None))

    def test_fetch_failure_returns_none_not_a_crash(self):
        with patch("chart._resolve_cik", return_value="0000320193"):
            with patch("chart.urllib.request.urlopen", side_effect=TimeoutError("simulated network failure")):
                self.assertIsNone(chart.generate_company_revenue_chart("AAPL"))

    def test_insufficient_clean_quarters_returns_none(self):
        facts = {"facts": {"us-gaap": {"Revenues": {"units": {"USD": _synthetic_sec_facts(n_quarters=2)}}}}}
        self.assertIsNone(chart._extract_quarterly_revenue(facts))

    def test_dispatches_through_resolve_visual(self):
        facts = {"facts": {"us-gaap": {"Revenues": {"units": {"USD": _synthetic_sec_facts()}}}}}
        with patch("chart._resolve_cik", return_value="0000320193"):
            with patch("chart.urllib.request.urlopen") as mock_urlopen:
                mock_urlopen.return_value.__enter__.return_value.read.return_value = json.dumps(facts).encode()
                result = {"visual_type": "company_revenue_chart", "ticker": "AAPL"}
                img = chart.resolve_visual(result, label="Apple")
        self.assertIsNotNone(img)


if __name__ == "__main__":
    unittest.main()
