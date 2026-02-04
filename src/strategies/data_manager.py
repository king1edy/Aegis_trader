"""
Multi-Timeframe Data Manager

Manages price data across multiple timeframes with:
- Caching to avoid repeated broker calls
- New bar detection to prevent duplicate signals
- Data freshness validation
- Automatic updates

Critical for MTFTR strategy to avoid duplicate signal generation.
"""

import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, List
from dataclasses import dataclass

from src.execution.mt5_connector import BrokerInterface
from src.core.exceptions import StaleDataError, MarketDataError
from src.core.logging_config import get_logger

logger = get_logger("data_manager")


@dataclass
class CachedData:
    """Cached price data with metadata"""
    dataframe: pd.DataFrame
    timestamp: datetime
    last_bar_timestamp: datetime
    bar_count: int


class MultiTimeframeDataManager:
    """
    Manage price data across multiple timeframes.

    Features:
    - Caching to avoid repeated broker calls (TTL-based)
    - New bar detection per timeframe
    - Data freshness validation
    - Automatic cache invalidation

    Usage:
        manager = MultiTimeframeDataManager(broker, cache_ttl_seconds=60)
        df = await manager.get_data("XAUUSD", "H4", count=250)
        is_new = await manager.is_new_bar("XAUUSD", "M15")
    """

    def __init__(
        self,
        broker: BrokerInterface,
        cache_ttl_seconds: int = 60,
        stale_threshold_minutes: int = 5
    ):
        """
        Initialize data manager.

        Args:
            broker: Broker interface for fetching data
            cache_ttl_seconds: Cache time-to-live in seconds (default: 60)
            stale_threshold_minutes: Max age before data is considered stale (default: 5)
        """
        self.broker = broker
        self.cache_ttl = timedelta(seconds=cache_ttl_seconds)
        self.stale_threshold = timedelta(minutes=stale_threshold_minutes)

        # Cache storage: key = "SYMBOL_TIMEFRAME"
        self._cache: Dict[str, CachedData] = {}

        # Track last bar timestamps for new bar detection
        self._last_bar_timestamps: Dict[str, datetime] = {}

        logger.info(
            "Data manager initialized",
            cache_ttl=cache_ttl_seconds,
            stale_threshold=stale_threshold_minutes
        )

    async def get_data(
        self,
        symbol: str,
        timeframe: str,
        count: int = 500
    ) -> pd.DataFrame:
        """
        Get price data with caching.

        Args:
            symbol: Trading symbol (e.g., "XAUUSD")
            timeframe: Timeframe (e.g., "H4", "H1", "M15")
            count: Number of bars to fetch

        Returns:
            DataFrame with OHLCV data and datetime index

        Raises:
            MarketDataError: If data fetch fails
            StaleDataError: If data is too old
        """
        cache_key = self._get_cache_key(symbol, timeframe)

        # Check cache
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            age = datetime.now(timezone.utc) - cached.timestamp

            # Return cached data if still fresh
            if age < self.cache_ttl:
                logger.debug(
                    "Returning cached data",
                    symbol=symbol,
                    timeframe=timeframe,
                    age_seconds=age.total_seconds(),
                    bars=len(cached.dataframe)
                )
                return cached.dataframe.copy()

            logger.debug(
                "Cache expired",
                symbol=symbol,
                timeframe=timeframe,
                age_seconds=age.total_seconds()
            )

        # Fetch fresh data
        df = await self._fetch_fresh_data(symbol, timeframe, count)

        # Validate data
        self._validate_data(df, symbol, timeframe)

        # Cache the data
        if len(df) > 0:
            last_bar_time = df.index[-1]
            self._cache[cache_key] = CachedData(
                dataframe=df.copy(),
                timestamp=datetime.now(timezone.utc),
                last_bar_timestamp=last_bar_time,
                bar_count=len(df)
            )

            logger.debug(
                "Data cached",
                symbol=symbol,
                timeframe=timeframe,
                bars=len(df),
                last_bar=last_bar_time
            )

        return df

    async def is_new_bar(
        self,
        symbol: str,
        timeframe: str
    ) -> bool:
        """
        Check if a new bar has formed since last check.

        CRITICAL: This must be called BEFORE analysis to prevent
        duplicate signal generation on the same bar.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe to check

        Returns:
            True if new bar detected, False otherwise
        """
        cache_key = self._get_cache_key(symbol, timeframe)

        # Get current data (will use cache if available)
        df = await self.get_data(symbol, timeframe, count=10)

        if len(df) == 0:
            return False

        current_bar_time = df.index[-1]

        # Check against last known bar time
        tracking_key = cache_key
        if tracking_key not in self._last_bar_timestamps:
            # First time checking - initialize and return True
            self._last_bar_timestamps[tracking_key] = current_bar_time
            logger.info(
                "First bar check - initialized",
                symbol=symbol,
                timeframe=timeframe,
                bar_time=current_bar_time
            )
            return True

        last_known_time = self._last_bar_timestamps[tracking_key]

        if current_bar_time > last_known_time:
            # New bar detected
            self._last_bar_timestamps[tracking_key] = current_bar_time
            logger.info(
                "New bar detected",
                symbol=symbol,
                timeframe=timeframe,
                previous_bar=last_known_time,
                current_bar=current_bar_time
            )
            return True

        # Same bar
        return False

    async def update_all_timeframes(
        self,
        symbol: str,
        timeframes: List[str],
        count: int = 500
    ) -> None:
        """
        Update data for all required timeframes.

        Useful to pre-fetch all data before analysis to minimize latency.

        Args:
            symbol: Trading symbol
            timeframes: List of timeframes to update (e.g., ["H4", "H1", "M15"])
            count: Number of bars to fetch per timeframe
        """
        logger.debug(
            "Updating multiple timeframes",
            symbol=symbol,
            timeframes=timeframes
        )

        for tf in timeframes:
            try:
                await self.get_data(symbol, tf, count)
            except Exception as e:
                logger.error(
                    "Failed to update timeframe",
                    symbol=symbol,
                    timeframe=tf,
                    error=str(e)
                )
                raise

    async def _fetch_fresh_data(
        self,
        symbol: str,
        timeframe: str,
        count: int
    ) -> pd.DataFrame:
        """
        Fetch fresh data from broker.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            count: Number of bars

        Returns:
            DataFrame with OHLCV data

        Raises:
            MarketDataError: If fetch fails
        """
        try:
            logger.debug(
                "Fetching fresh data from broker",
                symbol=symbol,
                timeframe=timeframe,
                count=count
            )

            price_data_list = await self.broker.get_price_data(
                symbol=symbol,
                timeframe=timeframe,
                count=count
            )

            if not price_data_list:
                raise MarketDataError(
                    f"No data returned for {symbol} {timeframe}"
                )

            # Convert to DataFrame
            data = []
            for bar in price_data_list:
                data.append({
                    'timestamp': bar.timestamp,
                    'open': bar.open,
                    'high': bar.high,
                    'low': bar.low,
                    'close': bar.close,
                    'volume': bar.tick_volume,  # PriceData uses tick_volume
                    'tick_volume': bar.tick_volume,
                    'spread': bar.spread
                })

            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)

            logger.debug(
                "Data fetched successfully",
                symbol=symbol,
                timeframe=timeframe,
                bars=len(df),
                first_bar=df.index[0] if len(df) > 0 else None,
                last_bar=df.index[-1] if len(df) > 0 else None
            )

            return df

        except Exception as e:
            logger.exception(
                "Failed to fetch data",
                symbol=symbol,
                timeframe=timeframe,
                error=str(e)
            )
            raise MarketDataError(
                f"Failed to fetch data for {symbol} {timeframe}: {str(e)}"
            )

    def _validate_data(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str
    ) -> None:
        """
        Validate data quality and freshness.

        Args:
            df: DataFrame to validate
            symbol: Symbol name
            timeframe: Timeframe

        Raises:
            StaleDataError: If data is too old
            MarketDataError: If data is invalid
        """
        if len(df) == 0:
            raise MarketDataError(f"Empty dataframe for {symbol} {timeframe}")

        # Check required columns
        required = ['open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise MarketDataError(
                f"Missing columns for {symbol} {timeframe}: {missing}"
            )

        # Check for NaN values
        if df[required].isna().any().any():
            logger.warning(
                "NaN values detected in data",
                symbol=symbol,
                timeframe=timeframe,
                nan_counts=df[required].isna().sum().to_dict()
            )

        # Check data freshness
        last_bar_time = df.index[-1]
        if not isinstance(last_bar_time, pd.Timestamp):
            last_bar_time = pd.Timestamp(last_bar_time)

        # Ensure timezone-aware
        if last_bar_time.tz is None:
            last_bar_time = last_bar_time.tz_localize('UTC')
        else:
            last_bar_time = last_bar_time.tz_convert('UTC')

        now = datetime.now(timezone.utc)
        age = now - last_bar_time.to_pydatetime()

        if age > self.stale_threshold:
            logger.warning(
                "Stale data detected",
                symbol=symbol,
                timeframe=timeframe,
                last_bar=last_bar_time,
                age_minutes=age.total_seconds() / 60
            )
            # Note: We log warning but don't raise - market may be closed

        # Check for negative prices
        if (df[['open', 'high', 'low', 'close']] < 0).any().any():
            raise MarketDataError(
                f"Negative prices detected for {symbol} {timeframe}"
            )

        # Check high/low consistency
        if (df['low'] > df['high']).any():
            raise MarketDataError(
                f"Invalid OHLC data: low > high for {symbol} {timeframe}"
            )

    def _get_cache_key(self, symbol: str, timeframe: str) -> str:
        """
        Generate cache key.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe

        Returns:
            Cache key string
        """
        return f"{symbol}_{timeframe}"

    async def clear_cache(self, symbol: Optional[str] = None) -> None:
        """
        Clear cache for symbol or all symbols.

        Args:
            symbol: Symbol to clear (None = clear all)
        """
        if symbol is None:
            self._cache.clear()
            self._last_bar_timestamps.clear()
            logger.info("Cache cleared for all symbols")
        else:
            keys_to_remove = [k for k in self._cache.keys() if k.startswith(symbol)]
            for key in keys_to_remove:
                del self._cache[key]
                if key in self._last_bar_timestamps:
                    del self._last_bar_timestamps[key]
            logger.info("Cache cleared", symbol=symbol, keys_removed=len(keys_to_remove))

    async def get_cache_stats(self) -> Dict[str, any]:
        """
        Get cache statistics for monitoring.

        Returns:
            Dictionary with cache statistics
        """
        total_cached = len(self._cache)
        total_bars = sum(cached.bar_count for cached in self._cache.values())

        cache_ages = {
            key: (datetime.now(timezone.utc) - cached.timestamp).total_seconds()
            for key, cached in self._cache.items()
        }

        return {
            'total_cached_timeframes': total_cached,
            'total_bars_cached': total_bars,
            'cache_keys': list(self._cache.keys()),
            'cache_ages_seconds': cache_ages,
            'tracked_bars': len(self._last_bar_timestamps)
        }

    async def force_refresh(self, symbol: str, timeframe: str, count: int = 500) -> pd.DataFrame:
        """
        Force refresh data by bypassing cache.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            count: Number of bars

        Returns:
            Fresh DataFrame from broker
        """
        cache_key = self._get_cache_key(symbol, timeframe)

        # Remove from cache
        if cache_key in self._cache:
            del self._cache[cache_key]

        logger.info("Force refresh requested", symbol=symbol, timeframe=timeframe)

        return await self.get_data(symbol, timeframe, count)
