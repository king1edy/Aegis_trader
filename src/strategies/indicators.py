"""
Indicator Calculator for MTFTR Strategy

Calculates technical indicators required for the Multi-Timeframe Trend Rider strategy:
- EMA (Exponential Moving Average)
- Hull MA (Hull Moving Average) - custom implementation
- RSI (Relative Strength Index)
- ATR (Average True Range)
- Swing high/low detection

All calculations designed to match MQL5 reference implementation.
"""

import numpy as np
import pandas as pd
from typing import Optional
from dataclasses import dataclass

from src.core.exceptions import InsufficientDataError, IndicatorCalculationError
from src.core.logging_config import get_logger

logger = get_logger("indicators")


@dataclass
class IndicatorConfig:
    """Configuration for indicator calculations"""
    ema_200: int = 200
    ema_50: int = 50
    ema_21: int = 21
    hull_55: int = 55
    hull_34: int = 34
    rsi_period: int = 14
    atr_period: int = 14
    swing_lookback: int = 5


class IndicatorCalculator:
    """
    Calculate technical indicators for strategy analysis.

    All methods are static for easy testing and reusability.
    Calculations verified against MQL5 reference implementation.
    """

    @staticmethod
    def calculate_all(df: pd.DataFrame, config: IndicatorConfig) -> pd.DataFrame:
        """
        Calculate all indicators and add to dataframe.

        Args:
            df: DataFrame with OHLCV data (columns: open, high, low, close, volume)
            config: Configuration with indicator periods

        Returns:
            DataFrame with added indicator columns

        Raises:
            InsufficientDataError: If not enough bars for calculations
            IndicatorCalculationError: If calculation fails
        """
        df = df.copy()

        try:
            # Validate minimum data requirements
            min_bars = max(config.ema_200, config.hull_55) + 10
            if len(df) < min_bars:
                raise InsufficientDataError(
                    f"Insufficient data for indicator calculation. "
                    f"Need {min_bars} bars, got {len(df)}"
                )

            # Calculate EMAs
            df['ema_200'] = IndicatorCalculator.calculate_ema(df['close'], config.ema_200)
            df['ema_50'] = IndicatorCalculator.calculate_ema(df['close'], config.ema_50)
            df['ema_21'] = IndicatorCalculator.calculate_ema(df['close'], config.ema_21)

            # Calculate Hull MAs
            df['hull_55'] = IndicatorCalculator.calculate_hull_ma(df['close'], config.hull_55)
            df['hull_34'] = IndicatorCalculator.calculate_hull_ma(df['close'], config.hull_34)

            # Calculate RSI
            df['rsi'] = IndicatorCalculator.calculate_rsi(df['close'], config.rsi_period)

            # Calculate ATR
            df['atr'] = IndicatorCalculator.calculate_atr(df, config.atr_period)

            # Detect swing points
            df['swing_high'] = False
            df['swing_low'] = False
            df['swing_high_price'] = np.nan
            df['swing_low_price'] = np.nan

            swing_highs, swing_lows = IndicatorCalculator.find_swings(
                df,
                config.swing_lookback
            )

            for idx in swing_highs:
                if idx < len(df):
                    df.loc[df.index[idx], 'swing_high'] = True
                    df.loc[df.index[idx], 'swing_high_price'] = df.iloc[idx]['high']

            for idx in swing_lows:
                if idx < len(df):
                    df.loc[df.index[idx], 'swing_low'] = True
                    df.loc[df.index[idx], 'swing_low_price'] = df.iloc[idx]['low']

            logger.debug(
                "Indicators calculated successfully",
                bars=len(df),
                nan_ema200=df['ema_200'].isna().sum(),
                nan_hull55=df['hull_55'].isna().sum()
            )

            return df

        except Exception as e:
            logger.exception("Indicator calculation failed", error=str(e))
            raise IndicatorCalculationError(f"Failed to calculate indicators: {str(e)}")

    @staticmethod
    def calculate_ema(series: pd.Series, period: int) -> pd.Series:
        """
        Calculate Exponential Moving Average.

        Args:
            series: Price series (typically close prices)
            period: EMA period

        Returns:
            Series with EMA values
        """
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def calculate_wma(series: pd.Series, period: int) -> pd.Series:
        """
        Calculate Weighted Moving Average.

        WMA gives more weight to recent prices using linear weighting.
        Weight[i] = (period - i) for i in range(period)

        Args:
            series: Price series
            period: WMA period

        Returns:
            Series with WMA values
        """
        weights = np.arange(1, period + 1)

        def wma_calc(x):
            if len(x) < period:
                return np.nan
            return np.sum(weights * x) / weights.sum()

        return series.rolling(window=period).apply(wma_calc, raw=True)

    @staticmethod
    def calculate_hull_ma(series: pd.Series, period: int) -> pd.Series:
        """
        Calculate Hull Moving Average (HMA).

        HMA formula:
        1. Calculate WMA of half period: wma_half = WMA(price, period/2)
        2. Calculate WMA of full period: wma_full = WMA(price, period)
        3. Calculate difference: raw_hma = 2 * wma_half - wma_full
        4. Apply WMA to difference: HMA = WMA(raw_hma, sqrt(period))

        This provides a smoother, more responsive moving average that reduces lag.

        Args:
            series: Price series (typically close prices)
            period: Hull MA period

        Returns:
            Series with Hull MA values
        """
        # Calculate periods
        half_period = int(period / 2)
        sqrt_period = int(np.sqrt(period))

        # Step 1 & 2: Calculate WMAs
        wma_half = IndicatorCalculator.calculate_wma(series, half_period)
        wma_full = IndicatorCalculator.calculate_wma(series, period)

        # Step 3: Calculate raw HMA
        raw_hma = 2 * wma_half - wma_full

        # Step 4: Apply WMA to raw HMA
        hull_ma = IndicatorCalculator.calculate_wma(raw_hma, sqrt_period)

        return hull_ma

    @staticmethod
    def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).

        RSI measures the magnitude of recent price changes to evaluate
        overbought or oversold conditions.

        Formula:
        1. Calculate price changes
        2. Separate gains and losses
        3. Calculate average gain and average loss
        4. RS = average gain / average loss
        5. RSI = 100 - (100 / (1 + RS))

        Args:
            series: Price series (typically close prices)
            period: RSI period (default: 14)

        Returns:
            Series with RSI values (0-100)
        """
        # Calculate price changes
        delta = series.diff()

        # Separate gains and losses
        gains = delta.where(delta > 0, 0.0)
        losses = -delta.where(delta < 0, 0.0)

        # Calculate average gain and loss using EMA (Wilder's smoothing)
        avg_gain = gains.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        avg_loss = losses.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range (ATR).

        ATR measures market volatility by decomposing the entire range of an asset.

        True Range is the greatest of:
        1. Current high - Current low
        2. Abs(Current high - Previous close)
        3. Abs(Current low - Previous close)

        ATR is the EMA of True Range.

        Args:
            df: DataFrame with high, low, close columns
            period: ATR period (default: 14)

        Returns:
            Series with ATR values
        """
        high = df['high']
        low = df['low']
        close = df['close']

        # Calculate True Range components
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())

        # True Range is the maximum of the three
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # ATR is EMA of True Range (Wilder's smoothing)
        atr = true_range.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

        return atr

    @staticmethod
    def find_swings(
        df: pd.DataFrame,
        lookback: int = 5
    ) -> tuple[list[int], list[int]]:
        """
        Identify swing high and swing low points.

        A swing high at bar i is confirmed when:
        - high[i] >= high[i-j] for all j in [1, lookback]
        - high[i] >= high[i+j] for all j in [1, lookback]

        Similarly for swing lows.

        Args:
            df: DataFrame with high and low columns
            lookback: Number of bars on each side to confirm swing

        Returns:
            Tuple of (swing_high_indices, swing_low_indices)
        """
        swing_highs = []
        swing_lows = []

        highs = df['high'].values
        lows = df['low'].values

        # Start from lookback and end at len-lookback to ensure we have bars on both sides
        for i in range(lookback, len(df) - lookback):
            # Check for swing high
            is_swing_high = True
            current_high = highs[i]

            for j in range(1, lookback + 1):
                if current_high < highs[i - j] or current_high < highs[i + j]:
                    is_swing_high = False
                    break

            if is_swing_high:
                swing_highs.append(i)

            # Check for swing low
            is_swing_low = True
            current_low = lows[i]

            for j in range(1, lookback + 1):
                if current_low > lows[i - j] or current_low > lows[i + j]:
                    is_swing_low = False
                    break

            if is_swing_low:
                swing_lows.append(i)

        return swing_highs, swing_lows

    @staticmethod
    def validate_dataframe(df: pd.DataFrame, symbol: str) -> None:
        """
        Validate that dataframe has required columns and data quality.

        Args:
            df: DataFrame to validate
            symbol: Symbol name for error messages

        Raises:
            InsufficientDataError: If validation fails
        """
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required_columns if col not in df.columns]

        if missing:
            raise InsufficientDataError(
                f"Missing required columns for {symbol}: {missing}"
            )

        if len(df) == 0:
            raise InsufficientDataError(f"Empty dataframe for {symbol}")

        # Check for all-NaN columns
        for col in required_columns:
            if df[col].isna().all():
                raise InsufficientDataError(
                    f"Column '{col}' is all NaN for {symbol}"
                )

        # Check for negative prices
        if (df[['open', 'high', 'low', 'close']] < 0).any().any():
            logger.warning(f"Negative prices detected in data for {symbol}")
