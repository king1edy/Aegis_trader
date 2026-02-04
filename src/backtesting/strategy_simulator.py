"""
MTFTR Strategy Simulator for Backtesting

Simulates the MTFTR strategy logic without requiring live broker connection.
Generates signals based on historical multi-timeframe data.
"""

import pandas as pd
from datetime import datetime, time, timezone
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

from src.strategies.base_strategy import TradingSignal
from src.strategies.indicators import IndicatorConfig
from src.execution.mt5_connector import OrderDirection
from src.backtesting.data_provider import BacktestDataProvider, MultiTimeframeBar
from src.core.logging_config import get_logger

logger = get_logger("strategy_simulator")


class TrendState(Enum):
    """Market trend state"""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


@dataclass
class MTFTRBacktestConfig:
    """Configuration for MTFTR strategy - EXACT MT5 EA MATCH"""
    # Indicator periods (same as MT5 EA)
    ema_200: int = 200
    ema_50: int = 50
    ema_21: int = 21
    hull_55: int = 55
    hull_34: int = 34
    rsi_period: int = 14
    atr_period: int = 14
    swing_lookback: int = 3  # MT5: InpSwingLookback = 3
    
    # Take Profit R:R - MT5 EA: InpTP1_RR=1.0, InpTP2_RR=2.0
    tp1_rr: float = 1.0   # TP1 at 1:1 R:R
    tp2_rr: float = 2.0   # TP2 at 2:1 R:R
    
    # RSI Entry Filters - MT5 EA exact values
    min_rsi_long: float = 40.0   # InpRSI_Long_Min = 40
    max_rsi_long: float = 55.0   # InpRSI_Long_Max = 55
    min_rsi_short: float = 45.0  # InpRSI_Short_Min = 45
    max_rsi_short: float = 60.0  # InpRSI_Short_Max = 60
    
    # Stop loss limits - MT5 EA exact values
    min_sl_atr: float = 1.0   # InpSL_Min_ATR = 1.0
    max_sl_atr: float = 3.0   # InpSL_Max_ATR = 3.0
    sl_buffer_atr: float = 0.5  # InpSL_ATR_Buffer = 0.5
    
    # Trend strength filter - MT5 EA: InpH4_NoTradeATR = 1.5
    min_atr_distance_from_ema: float = 1.5  # No-trade zone
    
    # EMA slope filter - MT5 EA: InpH4_SlopeMin = 0.5
    ema_slope_bars: int = 5
    ema_slope_min: float = 0.5  # Min EMA slope (points)
    
    # Session times (GMT) - MT5 EA exact values
    london_start: time = time(7, 0)   # InpLondonStart = 7
    london_end: time = time(12, 0)    # InpLondonEnd = 12
    ny_start: time = time(13, 0)      # InpNYStart = 13 (1 hour gap)
    ny_end: time = time(16, 0)        # InpNYEnd = 16
    
    # Time-based exit - MT5 EA: InpMaxTradeHours = 8
    max_trade_hours: int = 8
    
    # Friday filter - MT5 EA: stop at 15:00 GMT
    friday_cutoff: time = time(15, 0)


class MTFTRStrategySimulator:
    """
    Simulates MTFTR strategy for backtesting.
    
    Implements the same logic as the live strategy but works with
    historical data provided bar-by-bar.
    """
    
    def __init__(
        self,
        config: MTFTRBacktestConfig,
        data_provider: BacktestDataProvider
    ):
        """
        Initialize strategy simulator.
        
        Args:
            config: Strategy configuration
            data_provider: Historical data provider
        """
        self.config = config
        self.data_provider = data_provider
        
        # State tracking
        self._h4_trend: TrendState = TrendState.NEUTRAL
        self._h1_confirmed: bool = False
        self._last_signal_time: Optional[datetime] = None
        
        # Lookback data cache
        self._h4_lookback: Optional[pd.DataFrame] = None
        self._h1_lookback: Optional[pd.DataFrame] = None
        self._m15_lookback: Optional[pd.DataFrame] = None
        
        logger.info("MTFTR strategy simulator initialized")
    
    def analyze_bar(self, bar: MultiTimeframeBar) -> Optional[TradingSignal]:
        """
        Analyze a single bar and generate signal if conditions met.
        
        Args:
            bar: Multi-timeframe bar data
            
        Returns:
            TradingSignal if all conditions met, None otherwise
        """
        # 1. Check session filter
        if not self._is_trading_session(bar.timestamp):
            return None
        
        # 2. Update lookback data periodically
        if bar.h4_new_bar or self._h4_lookback is None:
            self._h4_lookback = self.data_provider.get_lookback_data(
                bar.timestamp, 'H4', 250
            )
        
        if bar.h1_new_bar or self._h1_lookback is None:
            self._h1_lookback = self.data_provider.get_lookback_data(
                bar.timestamp, 'H1', 200
            )
        
        self._m15_lookback = self.data_provider.get_lookback_data(
            bar.timestamp, 'M15', 100
        )
        
        if self._h4_lookback is None or self._h1_lookback is None or self._m15_lookback is None:
            return None
        
        if len(self._h4_lookback) < 10 or len(self._h1_lookback) < 10 or len(self._m15_lookback) < 10:
            return None
        
        # 3. Analyze 4H trend (on H4 bar close)
        if bar.h4_new_bar:
            self._h4_trend = self._analyze_4h_trend()
        
        if self._h4_trend == TrendState.NEUTRAL:
            return None
        
        # 4. Check 1H confirmation (on H1 bar close)
        if bar.h1_new_bar:
            self._h1_confirmed = self._analyze_1h_confirmation()
        
        if not self._h1_confirmed:
            return None
        
        # 5. Look for 15M entry trigger
        entry_result = self._find_15m_entry(bar)
        
        if entry_result is None:
            return None
        
        entry_price, stop_loss, entry_type, candle_pattern = entry_result
        
        # 6. Calculate TP levels
        sl_distance = abs(entry_price - stop_loss)
        
        if self._h4_trend == TrendState.BULLISH:
            tp1 = entry_price + (sl_distance * self.config.tp1_rr)
            tp2 = entry_price + (sl_distance * self.config.tp2_rr)
        else:
            tp1 = entry_price - (sl_distance * self.config.tp1_rr)
            tp2 = entry_price - (sl_distance * self.config.tp2_rr)
        
        # 7. Build signal
        direction = OrderDirection.BUY if self._h4_trend == TrendState.BULLISH else OrderDirection.SELL
        
        # Calculate confidence
        confidence = self._calculate_confidence(bar, entry_type)
        
        signal = TradingSignal(
            timestamp=bar.timestamp,
            symbol=self.data_provider.symbol,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit_1=tp1,
            take_profit_2=tp2,
            take_profit_final=None,
            confidence=confidence,
            reason=f"{entry_type}: {candle_pattern or 'price action'}",
            market_context={
                "h4_trend": self._h4_trend.value,
                "h4_ema_200": float(bar.h4_ema_200) if bar.h4_ema_200 else None,
                "h4_hull_55": float(bar.h4_hull_55) if bar.h4_hull_55 else None,
                "h1_ema_50": float(bar.h1_ema_50) if bar.h1_ema_50 else None,
                "h1_hull_34": float(bar.h1_hull_34) if bar.h1_hull_34 else None,
                "m15_ema_21": float(bar.m15_ema_21),
                "m15_rsi": float(bar.m15_rsi),
                "m15_atr": float(bar.m15_atr),
                "entry_type": entry_type,
                "candle_pattern": candle_pattern
            },
            strategy_data={
                "trigger_type": entry_type,
                "sl_distance_atr": sl_distance / bar.m15_atr if bar.m15_atr > 0 else 0,
                "rr_tp1": self.config.tp1_rr,
                "rr_tp2": self.config.tp2_rr
            }
        )
        
        self._last_signal_time = bar.timestamp
        
        logger.info(
            "Signal generated",
            timestamp=bar.timestamp.strftime("%Y-%m-%d %H:%M"),
            direction=direction.value,
            entry=entry_price,
            sl=stop_loss,
            tp1=tp1,
            entry_type=entry_type
        )
        
        return signal
    
    def _is_trading_session(self, timestamp: datetime) -> bool:
        """Check if current time is in a trading session - MATCHES MT5 EA"""
        current_time = timestamp.time()
        
        # Skip weekends (0=Monday, 5=Saturday, 6=Sunday)
        if timestamp.weekday() >= 5:
            return False
        
        # Friday filter: stop trading at 15:00 GMT (MT5 EA: InpFridayFilter=true)
        if timestamp.weekday() == 4:  # Friday
            if current_time >= self.config.friday_cutoff:
                return False
        
        # London session: 07:00 – 12:00 GMT (MT5 EA)
        in_london = self.config.london_start <= current_time < self.config.london_end
        
        # NY overlap: 13:00 – 16:00 GMT (MT5 EA)
        in_ny = self.config.ny_start <= current_time < self.config.ny_end
        
        return in_london or in_ny
    
    def _analyze_4h_trend(self) -> TrendState:
        """
        Analyze 4H trend direction - MATCHES MT5 EA Get4HTrendBias()
        
        Conditions for LONG BIAS:
        - Price above EMA 200
        - EMA 200 sloping up (> InpH4_SlopeMin)
        - Hull MA bullish (current > previous)
        - Price not in no-trade zone (within 1.5x ATR of EMA200)
        """
        if self._h4_lookback is None or len(self._h4_lookback) < self.config.ema_slope_bars + 2:
            return TrendState.NEUTRAL
        
        latest = self._h4_lookback.iloc[-1]
        slope_bar = self._h4_lookback.iloc[-1 - self.config.ema_slope_bars]  # 5 bars ago
        prev_bar = self._h4_lookback.iloc[-2]
        
        price = latest['close']
        ema_200 = latest['ema_200']
        ema_200_old = slope_bar['ema_200']
        hull_55 = latest['hull_55']
        hull_55_old = prev_bar['hull_55']
        atr = latest['atr']
        
        # Skip if any values are NaN
        if pd.isna(ema_200) or pd.isna(hull_55) or pd.isna(atr) or pd.isna(hull_55_old):
            return TrendState.NEUTRAL
        
        # No-trade zone: price within 1.5x ATR of EMA 200 (MT5 EA: InpH4_NoTradeATR=1.5)
        if abs(price - ema_200) < self.config.min_atr_distance_from_ema * atr:
            return TrendState.NEUTRAL
        
        # EMA 200 slope check (MT5 EA: InpH4_SlopeMin=0.5)
        ema_slope = ema_200 - ema_200_old
        if abs(ema_slope) < self.config.ema_slope_min:
            return TrendState.NEUTRAL  # EMA too flat
        
        # Hull MA direction (green = bullish, red = bearish)
        hull_bullish = hull_55 > hull_55_old
        hull_bearish = hull_55 < hull_55_old
        
        # LONG BIAS: price > EMA200 + EMA sloping up + Hull bullish
        if price > ema_200 and ema_slope > self.config.ema_slope_min and hull_bullish:
            return TrendState.BULLISH
        
        # SHORT BIAS: price < EMA200 + EMA sloping down + Hull bearish
        if price < ema_200 and ema_slope < -self.config.ema_slope_min and hull_bearish:
            return TrendState.BEARISH
        
        return TrendState.NEUTRAL
    
    def _analyze_1h_confirmation(self) -> bool:
        """
        Check 1H confirmation - MATCHES MT5 EA Check1HConfirmation()
        
        For LONG: price above 1H EMA 50 AND Hull bullish
        For SHORT: price below 1H EMA 50 AND Hull bearish
        """
        if self._h1_lookback is None or len(self._h1_lookback) < 5:
            return False
        
        latest = self._h1_lookback.iloc[-1]
        previous = self._h1_lookback.iloc[-2]
        
        price = latest['close']
        ema_50 = latest['ema_50']
        hull_34 = latest['hull_34']
        hull_34_old = previous['hull_34']
        
        # Skip if NaN
        if pd.isna(ema_50) or pd.isna(hull_34) or pd.isna(hull_34_old):
            return False
        
        # Hull34 must be in same direction as H4 Hull55 (strict alignment)
        hull_bullish = hull_34 > hull_34_old  # H1 Hull "green"
        hull_bearish = hull_34 < hull_34_old  # H1 Hull "red"
        
        # For LONGS: H1 Hull must be green (same as H4)
        if self._h4_trend == TrendState.BULLISH:
            # Price above EMA50 + Hull34 bullish = confirmed long setup
            return price > ema_50 and hull_bullish
        
        # For SHORTS: H1 Hull must be red (same as H4)  
        if self._h4_trend == TrendState.BEARISH:
            # Price below EMA50 + Hull34 bearish = confirmed short setup
            return price < ema_50 and hull_bearish
        
        return False
    
    def _find_15m_entry(
        self,
        bar: MultiTimeframeBar
    ) -> Optional[tuple]:
        """
        Find 15M entry trigger.
        
        Returns:
            Tuple of (entry_price, stop_loss, entry_type, candle_pattern) or None
        """
        if self._m15_lookback is None or len(self._m15_lookback) < 5:
            return None
        
        rsi = bar.m15_rsi
        atr = bar.m15_atr
        
        # Skip if NaN
        if pd.isna(rsi) or pd.isna(atr):
            return None
        
        # RSI filter
        if self._h4_trend == TrendState.BULLISH:
            if not (self.config.min_rsi_long <= rsi <= self.config.max_rsi_long):
                return None
        else:
            if not (self.config.min_rsi_short <= rsi <= self.config.max_rsi_short):
                return None
        
        # Method A: EMA bounce entry
        entry_a = self._check_ema_bounce(bar)
        if entry_a:
            return entry_a
        
        # Method B: Structure break entry
        entry_b = self._check_structure_break(bar)
        if entry_b:
            return entry_b
        
        return None
    
    def _check_momentum_confirmation(self, bar: MultiTimeframeBar) -> bool:
        """Check for momentum confirmation - ensures we're trading with momentum"""
        if self._m15_lookback is None or len(self._m15_lookback) < 5:
            return False
        
        # Get last 5 candles
        recent = self._m15_lookback.tail(5)
        
        if self._h4_trend == TrendState.BULLISH:
            # For longs: 3 of last 5 candles should be bullish
            bullish_count = sum(recent['close'] > recent['open'])
            # Price should be making higher lows
            lows = recent['low'].values
            higher_lows = all(lows[i] <= lows[i+1] for i in range(len(lows)-2, len(lows)-1))
            return bullish_count >= 3 or higher_lows
        else:
            # For shorts: 3 of last 5 candles should be bearish
            bearish_count = sum(recent['close'] < recent['open'])
            # Price should be making lower highs
            highs = recent['high'].values
            lower_highs = all(highs[i] >= highs[i+1] for i in range(len(highs)-2, len(highs)-1))
            return bearish_count >= 3 or lower_highs
        
        return False
    
    def _check_ema_bounce(self, bar: MultiTimeframeBar) -> Optional[tuple]:
        """Check for EMA bounce entry - SIMPLIFIED for more signals"""
        if self._m15_lookback is None or len(self._m15_lookback) < 3:
            return None
        
        latest = self._m15_lookback.iloc[-1]
        ema_21 = latest['ema_21']
        atr = latest['atr']
        
        if pd.isna(ema_21) or pd.isna(atr) or atr <= 0:
            return None
        
        # EMA touch tolerance: 0.5% or 0.3 ATR, whichever is smaller
        touch_tolerance = min(ema_21 * 0.005, atr * 0.3)
        
        # For longs: price near or bounced off EMA21
        if self._h4_trend == TrendState.BULLISH:
            # Price touched or is near EMA21 and closed bullish
            near_ema = abs(bar.m15_low - ema_21) <= touch_tolerance
            above_ema = bar.m15_close > ema_21
            bullish_close = bar.m15_close > bar.m15_open
            
            # Also accept price pullback to EMA that's still above
            pullback_to_ema = bar.m15_low <= ema_21 * 1.003 and bar.m15_close > ema_21
            
            if (near_ema or pullback_to_ema) and above_ema and bullish_close:
                entry_price = bar.m15_close
                # Tighter SL using recent swing low or candle low
                recent_lows = self._m15_lookback.tail(5)['low'].min()
                stop_loss = min(bar.m15_low, recent_lows) - (atr * self.config.sl_buffer_atr)
                
                # Validate SL distance
                sl_atr = abs(entry_price - stop_loss) / atr
                if self.config.min_sl_atr <= sl_atr <= self.config.max_sl_atr:
                    pattern = self._detect_reversal_candle(bar) or "bullish_bounce"
                    return (entry_price, stop_loss, "ema_bounce", pattern)
        
        # For shorts: price near or bounced off EMA21
        if self._h4_trend == TrendState.BEARISH:
            # Price touched or is near EMA21 and closed bearish
            near_ema = abs(bar.m15_high - ema_21) <= touch_tolerance
            below_ema = bar.m15_close < ema_21
            bearish_close = bar.m15_close < bar.m15_open
            
            # Also accept price pullback to EMA that's still below
            pullback_to_ema = bar.m15_high >= ema_21 * 0.997 and bar.m15_close < ema_21
            
            if (near_ema or pullback_to_ema) and below_ema and bearish_close:
                entry_price = bar.m15_close
                # Tighter SL using recent swing high or candle high
                recent_highs = self._m15_lookback.tail(5)['high'].max()
                stop_loss = max(bar.m15_high, recent_highs) + (atr * self.config.sl_buffer_atr)
                
                # Validate SL distance
                sl_atr = abs(entry_price - stop_loss) / atr
                if self.config.min_sl_atr <= sl_atr <= self.config.max_sl_atr:
                    pattern = self._detect_reversal_candle(bar) or "bearish_bounce"
                    return (entry_price, stop_loss, "ema_bounce", pattern)
        
        return None
    
    def _check_structure_break(self, bar: MultiTimeframeBar) -> Optional[tuple]:
        """Check for structure break entry"""
        if self._m15_lookback is None or len(self._m15_lookback) < 20:
            return None
        
        # Find recent swing points
        swing_highs = self._m15_lookback[self._m15_lookback['swing_high'] == True]
        swing_lows = self._m15_lookback[self._m15_lookback['swing_low'] == True]
        
        atr = bar.m15_atr
        if pd.isna(atr):
            return None
        
        # For longs: break above recent swing high
        if self._h4_trend == TrendState.BULLISH:
            if len(swing_highs) > 0:
                recent_high = swing_highs.iloc[-1]['high']
                
                # Close breaks above swing high
                if bar.m15_close > recent_high and bar.m15_open < recent_high:
                    entry_price = bar.m15_close
                    
                    # SL below recent swing low
                    if len(swing_lows) > 0:
                        recent_low = swing_lows.iloc[-1]['low']
                        stop_loss = recent_low - (atr * self.config.sl_buffer_atr)
                    else:
                        stop_loss = bar.m15_low - (atr * self.config.sl_buffer_atr)
                    
                    # Validate SL distance
                    sl_atr = abs(entry_price - stop_loss) / atr
                    if self.config.min_sl_atr <= sl_atr <= self.config.max_sl_atr:
                        return (entry_price, stop_loss, "structure_break", "break_high")
        
        # For shorts: break below recent swing low
        if self._h4_trend == TrendState.BEARISH:
            if len(swing_lows) > 0:
                recent_low = swing_lows.iloc[-1]['low']
                
                # Close breaks below swing low
                if bar.m15_close < recent_low and bar.m15_open > recent_low:
                    entry_price = bar.m15_close
                    
                    # SL above recent swing high
                    if len(swing_highs) > 0:
                        recent_high = swing_highs.iloc[-1]['high']
                        stop_loss = recent_high + (atr * self.config.sl_buffer_atr)
                    else:
                        stop_loss = bar.m15_high + (atr * self.config.sl_buffer_atr)
                    
                    # Validate SL distance
                    sl_atr = abs(entry_price - stop_loss) / atr
                    if self.config.min_sl_atr <= sl_atr <= self.config.max_sl_atr:
                        return (entry_price, stop_loss, "structure_break", "break_low")
        
        return None
    
    def _detect_reversal_candle(self, bar: MultiTimeframeBar) -> Optional[str]:
        """Detect reversal candle patterns"""
        body = abs(bar.m15_close - bar.m15_open)
        upper_wick = bar.m15_high - max(bar.m15_open, bar.m15_close)
        lower_wick = min(bar.m15_open, bar.m15_close) - bar.m15_low
        total_range = bar.m15_high - bar.m15_low
        
        if total_range == 0:
            return None
        
        body_ratio = body / total_range
        
        # Bullish patterns (for longs)
        if self._h4_trend == TrendState.BULLISH:
            # Hammer / Pin bar
            if lower_wick > body * 2 and upper_wick < body * 0.5:
                return "hammer"
            
            # Bullish engulfing
            if len(self._m15_lookback) >= 2:
                prev = self._m15_lookback.iloc[-2]
                if (prev['close'] < prev['open'] and  # Previous bearish
                    bar.m15_close > bar.m15_open and  # Current bullish
                    bar.m15_open < prev['close'] and  # Opens below prev close
                    bar.m15_close > prev['open']):    # Closes above prev open
                    return "bullish_engulfing"
            
            # Simple bullish candle with good body
            if bar.m15_close > bar.m15_open and body_ratio > 0.5:
                return "bullish_candle"
        
        # Bearish patterns (for shorts)
        if self._h4_trend == TrendState.BEARISH:
            # Shooting star / Inverted hammer
            if upper_wick > body * 2 and lower_wick < body * 0.5:
                return "shooting_star"
            
            # Bearish engulfing
            if len(self._m15_lookback) >= 2:
                prev = self._m15_lookback.iloc[-2]
                if (prev['close'] > prev['open'] and  # Previous bullish
                    bar.m15_close < bar.m15_open and  # Current bearish
                    bar.m15_open > prev['close'] and  # Opens above prev close
                    bar.m15_close < prev['open']):    # Closes below prev open
                    return "bearish_engulfing"
            
            # Simple bearish candle with good body
            if bar.m15_close < bar.m15_open and body_ratio > 0.5:
                return "bearish_candle"
        
        return None
    
    def _calculate_confidence(self, bar: MultiTimeframeBar, entry_type: str) -> float:
        """Calculate signal confidence score (0.0 - 1.0)"""
        confidence = 0.5  # Base confidence
        
        # Entry type bonus
        if entry_type == "ema_bounce":
            confidence += 0.1
        elif entry_type == "structure_break":
            confidence += 0.15
        
        # RSI positioning bonus
        rsi = bar.m15_rsi
        if self._h4_trend == TrendState.BULLISH:
            if 40 <= rsi <= 50:  # Ideal oversold bounce zone
                confidence += 0.1
        else:
            if 50 <= rsi <= 60:  # Ideal overbought bounce zone
                confidence += 0.1
        
        # Multi-timeframe alignment bonus
        if bar.h1_hull_34 and bar.h1_hull_34_prev:
            if self._h4_trend == TrendState.BULLISH and bar.h1_hull_34 > bar.h1_hull_34_prev:
                confidence += 0.1
            elif self._h4_trend == TrendState.BEARISH and bar.h1_hull_34 < bar.h1_hull_34_prev:
                confidence += 0.1
        
        return min(1.0, confidence)
