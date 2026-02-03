"""
Candlestick Pattern Recognition for MTFTR Strategy

Identifies reversal patterns for entry confirmation:
- Bullish Engulfing: Bullish candle engulfs previous bearish candle
- Bearish Engulfing: Bearish candle engulfs previous bullish candle
- Hammer/Pin Bar: Long lower wick (≥2× body), body in upper 60%
- Shooting Star: Long upper wick (≥2× body), body in lower 40%

All patterns designed to match MQL5 reference implementation.
"""

import pandas as pd
from typing import Optional, List
from dataclasses import dataclass

from src.core.logging_config import get_logger

logger = get_logger("patterns")


@dataclass
class PatternResult:
    """Result of pattern detection"""
    pattern_name: str
    confidence: float  # 0.0 to 1.0
    candle_index: int
    description: str


class PatternRecognizer:
    """
    Recognize candlestick reversal patterns for entry confirmation.

    All methods are static for easy testing and reusability.
    Patterns verified against MQL5 reference implementation.

    Usage:
        recognizer = PatternRecognizer()
        if recognizer.is_bullish_engulfing(latest, previous):
            # Entry signal confirmed
    """

    @staticmethod
    def is_bullish_engulfing(
        current: pd.Series,
        previous: pd.Series
    ) -> bool:
        """
        Check if current and previous candles form a bullish engulfing pattern.

        Pattern criteria:
        1. Previous candle is bearish (close < open)
        2. Current candle is bullish (close > open)
        3. Current body engulfs previous body:
           - current.open <= previous.close
           - current.close >= previous.open

        Args:
            current: Current candle (Series with open, high, low, close)
            previous: Previous candle (Series with open, high, low, close)

        Returns:
            True if bullish engulfing pattern detected
        """
        # Previous candle is bearish
        if previous['close'] >= previous['open']:
            return False

        # Current candle is bullish
        if current['close'] <= current['open']:
            return False

        # Current body engulfs previous body
        if current['open'] > previous['close']:
            return False

        if current['close'] < previous['open']:
            return False

        logger.debug(
            "Bullish engulfing pattern detected",
            prev_open=previous['open'],
            prev_close=previous['close'],
            curr_open=current['open'],
            curr_close=current['close']
        )

        return True

    @staticmethod
    def is_bearish_engulfing(
        current: pd.Series,
        previous: pd.Series
    ) -> bool:
        """
        Check if current and previous candles form a bearish engulfing pattern.

        Pattern criteria:
        1. Previous candle is bullish (close > open)
        2. Current candle is bearish (close < open)
        3. Current body engulfs previous body:
           - current.open >= previous.close
           - current.close <= previous.open

        Args:
            current: Current candle (Series with open, high, low, close)
            previous: Previous candle (Series with open, high, low, close)

        Returns:
            True if bearish engulfing pattern detected
        """
        # Previous candle is bullish
        if previous['close'] <= previous['open']:
            return False

        # Current candle is bearish
        if current['close'] >= current['open']:
            return False

        # Current body engulfs previous body
        if current['open'] < previous['close']:
            return False

        if current['close'] > previous['open']:
            return False

        logger.debug(
            "Bearish engulfing pattern detected",
            prev_open=previous['open'],
            prev_close=previous['close'],
            curr_open=current['open'],
            curr_close=current['close']
        )

        return True

    @staticmethod
    def is_hammer(candle: pd.Series, wick_ratio: float = 2.0) -> bool:
        """
        Check if candle is a hammer (bullish reversal).

        Pattern criteria:
        1. Long lower wick (≥ wick_ratio × body)
        2. Small upper wick (< 50% of lower wick)
        3. Body in upper 60% of candle range
        4. Can be bullish or bearish (color doesn't matter much)

        Args:
            candle: Candle data (Series with open, high, low, close)
            wick_ratio: Minimum ratio of lower wick to body (default: 2.0)

        Returns:
            True if hammer pattern detected
        """
        open_price = candle['open']
        high_price = candle['high']
        low_price = candle['low']
        close_price = candle['close']

        # Calculate body and wicks
        body_size = abs(close_price - open_price)
        body_top = max(open_price, close_price)
        body_bottom = min(open_price, close_price)

        lower_wick = body_bottom - low_price
        upper_wick = high_price - body_top
        total_range = high_price - low_price

        # Avoid division by zero
        if body_size == 0 or total_range == 0:
            return False

        # Check lower wick is long enough
        if lower_wick < wick_ratio * body_size:
            return False

        # Check upper wick is small
        if upper_wick > 0.5 * lower_wick:
            return False

        # Check body is in upper portion (body bottom should be in top 60%)
        body_position = (body_bottom - low_price) / total_range

        if body_position < 0.5:  # Body not in upper half
            return False

        logger.debug(
            "Hammer pattern detected",
            body_size=body_size,
            lower_wick=lower_wick,
            upper_wick=upper_wick,
            body_position=body_position
        )

        return True

    @staticmethod
    def is_shooting_star(candle: pd.Series, wick_ratio: float = 2.0) -> bool:
        """
        Check if candle is a shooting star (bearish reversal).

        Pattern criteria:
        1. Long upper wick (≥ wick_ratio × body)
        2. Small lower wick (< 50% of upper wick)
        3. Body in lower 40% of candle range
        4. Can be bullish or bearish (color doesn't matter much)

        Args:
            candle: Candle data (Series with open, high, low, close)
            wick_ratio: Minimum ratio of upper wick to body (default: 2.0)

        Returns:
            True if shooting star pattern detected
        """
        open_price = candle['open']
        high_price = candle['high']
        low_price = candle['low']
        close_price = candle['close']

        # Calculate body and wicks
        body_size = abs(close_price - open_price)
        body_top = max(open_price, close_price)
        body_bottom = min(open_price, close_price)

        lower_wick = body_bottom - low_price
        upper_wick = high_price - body_top
        total_range = high_price - low_price

        # Avoid division by zero
        if body_size == 0 or total_range == 0:
            return False

        # Check upper wick is long enough
        if upper_wick < wick_ratio * body_size:
            return False

        # Check lower wick is small
        if lower_wick > 0.5 * upper_wick:
            return False

        # Check body is in lower portion (body top should be in bottom 40%)
        body_position = (body_top - low_price) / total_range

        if body_position > 0.5:  # Body not in lower half
            return False

        logger.debug(
            "Shooting star pattern detected",
            body_size=body_size,
            lower_wick=lower_wick,
            upper_wick=upper_wick,
            body_position=body_position
        )

        return True

    @staticmethod
    def scan_reversal_patterns(
        df: pd.DataFrame,
        direction: str,
        lookback: int = 3
    ) -> Optional[PatternResult]:
        """
        Scan recent candles for reversal patterns.

        Args:
            df: DataFrame with OHLC data
            direction: "bullish" or "bearish"
            lookback: Number of recent bars to check (default: 3)

        Returns:
            PatternResult if pattern found, None otherwise
        """
        if len(df) < 2:
            return None

        # Check most recent bars
        for i in range(1, min(lookback + 1, len(df))):
            current_idx = len(df) - i
            current = df.iloc[current_idx]

            if current_idx == 0:
                break

            previous = df.iloc[current_idx - 1]

            if direction == "bullish":
                # Check bullish patterns
                if PatternRecognizer.is_bullish_engulfing(current, previous):
                    return PatternResult(
                        pattern_name="bullish_engulfing",
                        confidence=0.8,
                        candle_index=current_idx,
                        description="Bullish engulfing pattern - strong reversal signal"
                    )

                if PatternRecognizer.is_hammer(current):
                    return PatternResult(
                        pattern_name="hammer",
                        confidence=0.75,
                        candle_index=current_idx,
                        description="Hammer pattern - bullish reversal"
                    )

            elif direction == "bearish":
                # Check bearish patterns
                if PatternRecognizer.is_bearish_engulfing(current, previous):
                    return PatternResult(
                        pattern_name="bearish_engulfing",
                        confidence=0.8,
                        candle_index=current_idx,
                        description="Bearish engulfing pattern - strong reversal signal"
                    )

                if PatternRecognizer.is_shooting_star(current):
                    return PatternResult(
                        pattern_name="shooting_star",
                        confidence=0.75,
                        candle_index=current_idx,
                        description="Shooting star pattern - bearish reversal"
                    )

        return None

    @staticmethod
    def get_all_patterns(
        df: pd.DataFrame,
        lookback: int = 10
    ) -> List[PatternResult]:
        """
        Find all reversal patterns in recent bars.

        Args:
            df: DataFrame with OHLC data
            lookback: Number of recent bars to scan

        Returns:
            List of PatternResult objects
        """
        patterns = []

        if len(df) < 2:
            return patterns

        # Scan recent bars
        for i in range(1, min(lookback + 1, len(df))):
            current_idx = len(df) - i
            current = df.iloc[current_idx]

            if current_idx == 0:
                break

            previous = df.iloc[current_idx - 1]

            # Check bullish patterns
            if PatternRecognizer.is_bullish_engulfing(current, previous):
                patterns.append(PatternResult(
                    pattern_name="bullish_engulfing",
                    confidence=0.8,
                    candle_index=current_idx,
                    description="Bullish engulfing"
                ))

            if PatternRecognizer.is_hammer(current):
                patterns.append(PatternResult(
                    pattern_name="hammer",
                    confidence=0.75,
                    candle_index=current_idx,
                    description="Hammer"
                ))

            # Check bearish patterns
            if PatternRecognizer.is_bearish_engulfing(current, previous):
                patterns.append(PatternResult(
                    pattern_name="bearish_engulfing",
                    confidence=0.8,
                    candle_index=current_idx,
                    description="Bearish engulfing"
                ))

            if PatternRecognizer.is_shooting_star(current):
                patterns.append(PatternResult(
                    pattern_name="shooting_star",
                    confidence=0.75,
                    candle_index=current_idx,
                    description="Shooting star"
                ))

        return patterns

    @staticmethod
    def is_pin_bar(
        candle: pd.Series,
        direction: str,
        wick_ratio: float = 2.0
    ) -> bool:
        """
        Check if candle is a pin bar (long wick rejection candle).

        Args:
            candle: Candle data (Series with open, high, low, close)
            direction: "bullish" (long lower wick) or "bearish" (long upper wick)
            wick_ratio: Minimum ratio of rejection wick to body

        Returns:
            True if pin bar pattern detected
        """
        if direction == "bullish":
            return PatternRecognizer.is_hammer(candle, wick_ratio)
        elif direction == "bearish":
            return PatternRecognizer.is_shooting_star(candle, wick_ratio)
        else:
            return False

    @staticmethod
    def get_pattern_confidence(pattern_name: str) -> float:
        """
        Get default confidence level for a pattern.

        Args:
            pattern_name: Name of the pattern

        Returns:
            Confidence level (0.0 to 1.0)
        """
        confidence_map = {
            "bullish_engulfing": 0.8,
            "bearish_engulfing": 0.8,
            "hammer": 0.75,
            "shooting_star": 0.75,
            "pin_bar": 0.75
        }

        return confidence_map.get(pattern_name.lower(), 0.5)
