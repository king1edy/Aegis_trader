"""
Historical Data Loader

Loads price data from MetaTrader 5 .hcs (history cache) files.
Supports multiple timeframes and symbols.
"""

import struct
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Optional
import pandas as pd

from src.core.logging_config import get_logger
from src.core.exceptions import MarketDataError

logger = get_logger("historical_loader")


class MT5BarData:
    """Represents a single OHLC bar from MT5."""

    def __init__(self, timestamp: int, open_price: float, high: float,
                 low: float, close: float, tick_volume: int, spread: int, real_volume: int):
        self.timestamp = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        self.open = open_price
        self.high = high
        self.low = low
        self.close = close
        self.tick_volume = tick_volume
        self.spread = spread
        self.real_volume = real_volume


class HistoricalDataLoader:
    """
    Loads historical price data from MT5 .hcs files.

    MT5 .hcs file structure:
    - Header: 505 bytes (version, copyright, symbol info)
    - Bar data: 60 bytes per bar
      - timestamp (8 bytes, int64)
      - open (8 bytes, double)
      - high (8 bytes, double)
      - low (8 bytes, double)
      - close (8 bytes, double)
      - tick_volume (8 bytes, uint64)
      - spread (4 bytes, int32)
      - real_volume (8 bytes, uint64)
    """

    HEADER_SIZE = 505
    BAR_SIZE = 60

    def __init__(self, data_path: str = "data/history"):
        """
        Initialize historical data loader.

        Args:
            data_path: Base path to historical data directory
        """
        self.data_path = Path(data_path)
        self._cache: Dict[str, pd.DataFrame] = {}

        logger.info("Historical data loader initialized", path=str(self.data_path))

    def load_symbol_data(self, symbol: str, start_year: int = 2019, end_year: int = 2026) -> pd.DataFrame:
        """
        Load all available data for a symbol across multiple years.

        Args:
            symbol: Symbol name (e.g., "XAUUSD")
            start_year: Starting year
            end_year: Ending year

        Returns:
            DataFrame with OHLCV data
        """
        cache_key = f"{symbol}_{start_year}_{end_year}"

        if cache_key in self._cache:
            logger.debug("Returning cached data", symbol=symbol)
            return self._cache[cache_key].copy()

        symbol_path = self.data_path / symbol

        if not symbol_path.exists():
            raise MarketDataError(f"Symbol directory not found: {symbol_path}")

        all_bars = []

        for year in range(start_year, end_year + 1):
            hcs_file = symbol_path / f"{year}.hcs"

            if not hcs_file.exists():
                logger.debug(f"File not found, skipping", file=str(hcs_file))
                continue

            try:
                bars = self._parse_hcs_file(hcs_file)
                all_bars.extend(bars)
                logger.info(f"Loaded data", file=hcs_file.name, bars=len(bars))
            except Exception as e:
                logger.error(f"Failed to parse file", file=str(hcs_file), error=str(e))

        if not all_bars:
            raise MarketDataError(f"No data found for {symbol}")

        # Convert to DataFrame
        df = pd.DataFrame([
            {
                'timestamp': bar.timestamp,
                'open': bar.open,
                'high': bar.high,
                'low': bar.low,
                'close': bar.close,
                'volume': bar.real_volume if bar.real_volume > 0 else bar.tick_volume,
                'tick_volume': bar.tick_volume,
                'spread': bar.spread
            }
            for bar in all_bars
        ])

        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)

        # Remove duplicates
        df = df[~df.index.duplicated(keep='last')]

        logger.info(f"Loaded total bars", symbol=symbol, bars=len(df),
                    start=df.index[0] if len(df) > 0 else None,
                    end=df.index[-1] if len(df) > 0 else None)

        # Cache
        self._cache[cache_key] = df

        return df.copy()

    def _parse_hcs_file(self, file_path: Path) -> List[MT5BarData]:
        """
        Parse a single .hcs file.

        Args:
            file_path: Path to .hcs file

        Returns:
            List of MT5BarData objects
        """
        bars = []

        with open(file_path, 'rb') as f:
            # Skip header
            f.seek(self.HEADER_SIZE)

            # Read bar data
            while True:
                bar_data = f.read(self.BAR_SIZE)

                if len(bar_data) < self.BAR_SIZE:
                    break

                try:
                    # Unpack bar structure
                    # Format: Q=uint64, d=double, I=uint32
                    timestamp, open_price, high, low, close, tick_volume, spread, real_volume = struct.unpack(
                        '<QddddQIQ',  # Little-endian format
                        bar_data
                    )

                    # Skip invalid bars (timestamp = 0 or future dates)
                    if timestamp == 0 or timestamp > datetime.now(timezone.utc).timestamp() + 86400:
                        continue

                    bar = MT5BarData(
                        timestamp=timestamp,
                        open_price=open_price,
                        high=high,
                        low=low,
                        close=close,
                        tick_volume=tick_volume,
                        spread=spread,
                        real_volume=real_volume
                    )

                    bars.append(bar)

                except struct.error as e:
                    logger.warning(f"Failed to unpack bar", error=str(e))
                    continue

        return bars

    def get_timeframe_data(self, symbol: str, timeframe: str, count: int = 500,
                          end_time: Optional[datetime] = None) -> pd.DataFrame:
        """
        Get data for a specific timeframe by resampling M1 data.

        Args:
            symbol: Symbol name
            timeframe: Timeframe (M1, M5, M15, M30, H1, H4, D1)
            count: Number of bars to return
            end_time: End timestamp (default: latest)

        Returns:
            DataFrame with OHLCV data
        """
        # Load full M1 data
        df_m1 = self.load_symbol_data(symbol)

        if df_m1.empty:
            raise MarketDataError(f"No data available for {symbol}")

        # Filter by end_time if specified
        if end_time:
            df_m1 = df_m1[df_m1.index <= end_time]

        # Resample to target timeframe
        df_resampled = self._resample_to_timeframe(df_m1, timeframe)

        # Return last N bars
        return df_resampled.tail(count)

    def _resample_to_timeframe(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Resample DataFrame to target timeframe.

        Args:
            df: Source DataFrame (M1 data)
            timeframe: Target timeframe

        Returns:
            Resampled DataFrame
        """
        # Map MT5 timeframes to pandas resample rules
        timeframe_map = {
            'M1': '1min',
            'M5': '5min',
            'M15': '15min',
            'M30': '30min',
            'H1': '1H',
            'H4': '4H',
            'D1': '1D',
            'W1': '1W',
            'MN1': '1MS'
        }

        if timeframe not in timeframe_map:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        rule = timeframe_map[timeframe]

        # Resample OHLC data
        df_resampled = df.resample(rule).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'tick_volume': 'sum',
            'spread': 'mean'
        })

        # Drop rows with NaN (incomplete bars)
        df_resampled = df_resampled.dropna()

        return df_resampled
