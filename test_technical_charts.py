"""Tests for the technical-indicator chart types (moving average, Bollinger Bands, RSI, MACD,
drawdown, historical volatility) -- all computed purely from OHLC data already fetched via
yfinance, no new data source. Uses a synthetic price series (mocked _fetch_daily_history) so
these run offline and deterministically rather than depending on live market data."""

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


if __name__ == "__main__":
    unittest.main()
