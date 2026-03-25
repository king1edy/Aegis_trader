"""
Multi-Timeframe Trend Rider (MTFTR) Strategy

A comprehensive trend-following strategy that analyzes:
- 4H timeframe for trend direction
- 1H timeframe for confirmation
- 15M timeframe for precise entry timing

Entry Methods:
- Method A: EMA bounce with reversal candle
- Method B: Structure break above/below swing points

Exit Management:
- TP1 at 1:1 R:R → Close 50%, move SL to BE
- TP2 at 2:1 R:R → Close 30%
- Trail remaining 20% on 1H EMA50 with Hull MA flip exit

Risk Management:
- 1% risk per trade
- Max 2 open positions
- Max 3 trades per day
- Session filter: London/NY only
"""

import pandas as pd
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum

from src.strategies.base_strategy import BaseStrategy, StrategyConfig, TradingSignal
from src.strategies.indicators import IndicatorCalculator, IndicatorConfig
from src.strategies.data_manager import MultiTimeframeDataManager
from src.strategies.filters.session_filter import SessionFilter
from src.strategies.patterns import PatternRecognizer
from src.execution.mt5_connector import BrokerInterface, OrderDirection
from src.core.exceptions import InsufficientDataError, MarketDataError
from src.core.logging_config import get_logger

logger = get_logger("mtftr")


class TrendState(Enum):
    """Market trend state"""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


@dataclass
class MTFTRConfig(StrategyConfig):
    """Configuration for MTFTR strategy"""

    # Indicator periods
    ema_200: int = 200  # 4H trend
    ema_50: int = 50    # 1H confirmation
    ema_21: int = 21    # 15M entry
    hull_55: int = 55   # 4H Hull MA
    hull_34: int = 34   # 1H Hull MA
    rsi_period: int = 14
    atr_period: int = 14
    swing_lookback: int = 5

    # Risk parameters
    tp1_rr: float = 1.0  # TP1 at 1:1
    tp2_rr: float = 2.0  # TP2 at 2:1
    tp1_close_percent: float = 0.50  # Close 50% at TP1
    tp2_close_percent: float = 0.30  # Close 30% at TP2
    trail_percent: float = 0.20      # Trail 20%

    # Entry filters
    min_rsi_long: float = 40.0
    max_rsi_long: float = 55.0
    min_rsi_short: float = 45.0
    max_rsi_short: float = 60.0

    # Stop loss limits (in ATR multiples)
    min_sl_atr: float = 1.0
    max_sl_atr: float = 3.0
    sl_buffer_atr: float = 0.5

    # Limits
    max_trade_duration_hours: int = 8


@dataclass
class EntrySignal:
    """Entry trigger details"""
    trigger_type: str  # "ema_bounce" or "structure_break"
    entry_price: float
    stop_loss: float
    confidence: float
    candle_pattern: Optional[str] = None
    reason: str = ""


class MTFTRStrategy(BaseStrategy):
    """
    Multi-Timeframe Trend Rider strategy implementation.

    Analysis flow:
    1. Check session filter (London/NY only)
    2. Check for new 15M bar (prevent duplicate signals)
    3. Fetch and cache multi-timeframe data
    4. Calculate all indicators
    5. Analyze 4H trend direction
    6. Confirm with 1H alignment
    7. Look for 15M entry trigger
    8. Calculate SL and TP levels
    9. Return TradingSignal if all conditions met
    """

    def __init__(
        self,
        config: MTFTRConfig,
        broker: BrokerInterface,
        data_manager: MultiTimeframeDataManager,
        indicator_calc: IndicatorCalculator,
        session_filter: SessionFilter
    ):
        """
        Initialize MTFTR strategy.

        Args:
            config: Strategy configuration
            broker: Broker interface
            data_manager: Multi-timeframe data manager
            indicator_calc: Indicator calculator
            session_filter: Session filter
        """
        super().__init__(config, broker)

        self.config: MTFTRConfig = config  # Type hint for autocomplete
        self.data_manager = data_manager
        self.indicator_calc = indicator_calc
        self.session_filter = session_filter
        self.pattern_recognizer = PatternRecognizer()

        # Create indicator config
        self.indicator_config = IndicatorConfig(
            ema_200=config.ema_200,
            ema_50=config.ema_50,
            ema_21=config.ema_21,
            hull_55=config.hull_55,
            hull_34=config.hull_34,
            rsi_period=config.rsi_period,
            atr_period=config.atr_period,
            swing_lookback=config.swing_lookback
        )

        logger.info(
            "MTFTR strategy initialized",
            symbol=config.symbol,
            ema_200=config.ema_200,
            hull_55=config.hull_55,
            tp1_rr=config.tp1_rr,
            tp2_rr=config.tp2_rr
        )

    async def get_required_timeframes(self) -> List[str]:
        """Get required timeframes"""
        return ["H4", "H1", "M1"]

    async def analyze(self, symbol: str) -> Optional[TradingSignal]:
        """
        Main analysis method - orchestrates the multi-timeframe analysis.

        Args:
            symbol: Trading symbol

        Returns:
            TradingSignal if all conditions met, None otherwise
        """
        try:
            # 1. Check session filter
            is_tradeable, session = await self.session_filter.is_tradeable_time()
            if not is_tradeable:
                logger.debug(
                    "Outside trading sessions",
                    reason=session
                )
                return None

            # 2. Check for new M1 bar (CRITICAL - prevents duplicate signals)
            if not await self.data_manager.is_new_bar(symbol, "M1"):
                return None

            logger.info(
                "New M1 bar detected - starting analysis",
                symbol=symbol,
                session=session
            )

            # 3. Fetch multi-timeframe data
            await self.data_manager.update_all_timeframes(
                symbol,
                ["H4", "H1", "M1"],
                count=500
            )

            df_h4 = await self.data_manager.get_data(symbol, "H4", count=250)
            df_h1 = await self.data_manager.get_data(symbol, "H1", count=200)
            df_m1 = await self.data_manager.get_data(symbol, "M1", count=500)

            # 4. Calculate indicators
            df_h4 = self.indicator_calc.calculate_all(df_h4, self.indicator_config)
            df_h1 = self.indicator_calc.calculate_all(df_h1, self.indicator_config)
            df_m1 = self.indicator_calc.calculate_all(df_m1, self.indicator_config)

            # 5. Analyze 4H trend
            h4_trend = await self.analyze_4h_trend(df_h4)
            if h4_trend == TrendState.NEUTRAL:
                logger.debug("4H trend is neutral - no trade")
                return None

            logger.info(
                "4H trend identified",
                trend=h4_trend.value,
                ema_200=df_h4.iloc[-1]['ema_200'],
                close=df_h4.iloc[-1]['close']
            )

            # 6. Check 1H confirmation
            h1_confirmed = await self.analyze_1h_confirmation(df_h1, h4_trend)
            if not h1_confirmed:
                logger.debug("1H confirmation failed")
                return None

            logger.info(
                "1H confirmation passed",
                ema_50=df_h1.iloc[-1]['ema_50'],
                close=df_h1.iloc[-1]['close']
            )

            # 7. Look for 15M entry
            entry_signal = await self.find_15m_entry(df_m1, h4_trend)
            if not entry_signal:
                logger.debug("No 15M entry trigger found")
                return None

            logger.info(
                "15M entry signal found",
                trigger_type=entry_signal.trigger_type,
                entry=entry_signal.entry_price,
                sl=entry_signal.stop_loss,
                pattern=entry_signal.candle_pattern
            )

            # 8. Calculate TP levels
            sl_distance = abs(entry_signal.entry_price - entry_signal.stop_loss)

            if h4_trend == TrendState.BULLISH:
                tp1 = entry_signal.entry_price + (sl_distance * self.config.tp1_rr)
                tp2 = entry_signal.entry_price + (sl_distance * self.config.tp2_rr)
            else:
                tp1 = entry_signal.entry_price - (sl_distance * self.config.tp1_rr)
                tp2 = entry_signal.entry_price - (sl_distance * self.config.tp2_rr)

            # 9. Build TradingSignal
            direction = OrderDirection.BUY if h4_trend == TrendState.BULLISH else OrderDirection.SELL

            signal = TradingSignal(
                timestamp=datetime.now(timezone.utc),
                symbol=symbol,
                direction=direction,
                entry_price=entry_signal.entry_price,
                stop_loss=entry_signal.stop_loss,
                take_profit_1=tp1,
                take_profit_2=tp2,
                take_profit_final=None,
                confidence=entry_signal.confidence,
                reason=f"{entry_signal.trigger_type}: {entry_signal.reason}",
                market_context={
                    "h4_trend": h4_trend.value,
                    "h4_ema_200": float(df_h4.iloc[-1]['ema_200']),
                    "h4_hull_55": float(df_h4.iloc[-1]['hull_55']),
                    "h1_ema_50": float(df_h1.iloc[-1]['ema_50']),
                    "h1_hull_34": float(df_h1.iloc[-1]['hull_34']),
                    "m15_ema_21": float(df_m1.iloc[-1]['ema_21']),
                    "m15_rsi": float(df_m1.iloc[-1]['rsi']),
                    "m15_atr": float(df_m1.iloc[-1]['atr']),
                    "session": session,
                    "candle_pattern": entry_signal.candle_pattern
                },
                strategy_data={
                    "trigger_type": entry_signal.trigger_type,
                    "sl_distance_atr": sl_distance / float(df_m1.iloc[-1]['atr']),
                    "rr_tp1": self.config.tp1_rr,
                    "rr_tp2": self.config.tp2_rr
                }
            )

            logger.info(
                "Trading signal generated",
                symbol=symbol,
                direction=direction.value,
                entry=entry_signal.entry_price,
                sl=entry_signal.stop_loss,
                tp1=tp1,
                tp2=tp2,
                confidence=entry_signal.confidence,
                rr=signal.get_risk_reward_ratio()
            )

            return signal

        except InsufficientDataError as e:
            logger.warning("Insufficient data for analysis", error=str(e))
            return None
        except MarketDataError as e:
            logger.error("Market data error", error=str(e))
            return None
        except Exception as e:
            logger.exception("Unexpected error in analysis", error=str(e))
            return None

    async def analyze_4h_trend(self, df: pd.DataFrame) -> TrendState:
        """
        Determine 4H trend direction.

        Criteria for BULLISH:
        - Price > EMA200
        - EMA200 sloping up
        - Hull MA bullish (rising)
        - Not in no-trade zone (within 1.5× ATR of EMA200)

        Criteria for BEARISH:
        - Price < EMA200
        - EMA200 sloping down
        - Hull MA bearish (falling)
        - Not in no-trade zone

        Args:
            df: 4H DataFrame with indicators

        Returns:
            TrendState (BULLISH, BEARISH, or NEUTRAL)
        """
        latest = df.iloc[-1]
        old = df.iloc[-6]  # 5 bars ago (~ 1 day)

        price = latest['close']
        ema_200 = latest['ema_200']
        ema_200_old = old['ema_200']
        hull_55 = latest['hull_55']
        hull_55_old = df.iloc[-2]['hull_55']
        atr = latest['atr']

        # Check no-trade zone: within 1.5× ATR of EMA 200
        if abs(price - ema_200) < 1.5 * atr:
            logger.debug(
                "Price in no-trade zone around EMA200",
                price=price,
                ema_200=ema_200,
                distance_atr=(abs(price - ema_200) / atr)
            )
            return TrendState.NEUTRAL

        # EMA slope
        ema_slope = ema_200 - ema_200_old
        if abs(ema_slope) < 0.5:
            logger.debug("EMA200 too flat", slope=ema_slope)
            return TrendState.NEUTRAL

        # Hull direction
        hull_bullish = hull_55 > hull_55_old
        hull_bearish = hull_55 < hull_55_old

        # BULLISH bias
        if price > ema_200 and ema_slope > 0.5 and hull_bullish:
            logger.debug(
                "Bullish 4H trend detected",
                price=price,
                ema_200=ema_200,
                ema_slope=ema_slope,
                hull_55=hull_55,
                hull_55_old=hull_55_old
            )
            return TrendState.BULLISH

        # BEARISH bias
        if price < ema_200 and ema_slope < -0.5 and hull_bearish:
            logger.debug(
                "Bearish 4H trend detected",
                price=price,
                ema_200=ema_200,
                ema_slope=ema_slope,
                hull_55=hull_55,
                hull_55_old=hull_55_old
            )
            return TrendState.BEARISH

        return TrendState.NEUTRAL

    async def analyze_1h_confirmation(
        self,
        df: pd.DataFrame,
        h4_trend: TrendState
    ) -> bool:
        """
        Check if 1H timeframe confirms 4H trend.

        For BULLISH 4H:
        - Price > EMA50
        - Hull MA rising

        For BEARISH 4H:
        - Price < EMA50
        - Hull MA falling

        Args:
            df: 1H DataFrame with indicators
            h4_trend: 4H trend state

        Returns:
            True if confirmed, False otherwise
        """
        latest = df.iloc[-1]
        previous = df.iloc[-2]

        price = latest['close']
        ema_50 = latest['ema_50']
        hull_34 = latest['hull_34']
        hull_34_old = previous['hull_34']

        hull_bullish = hull_34 > hull_34_old
        hull_bearish = hull_34 < hull_34_old

        if h4_trend == TrendState.BULLISH:
            confirmed = price > ema_50 and hull_bullish
            logger.debug(
                "1H bullish confirmation check",
                price_above_ema=price > ema_50,
                hull_rising=hull_bullish,
                confirmed=confirmed
            )
            return confirmed

        if h4_trend == TrendState.BEARISH:
            confirmed = price < ema_50 and hull_bearish
            logger.debug(
                "1H bearish confirmation check",
                price_below_ema=price < ema_50,
                hull_falling=hull_bearish,
                confirmed=confirmed
            )
            return confirmed

        return False

    async def find_15m_entry(
        self,
        df: pd.DataFrame,
        h4_trend: TrendState
    ) -> Optional[EntrySignal]:
        """
        Look for entry trigger on 15M timeframe.

        Checks:
        1. RSI filter
        2. Method A: EMA bounce + reversal candle
        3. Method B: Structure break

        Args:
            df: 15M DataFrame with indicators
            h4_trend: 4H trend direction

        Returns:
            EntrySignal if found, None otherwise
        """
        latest = df.iloc[-1]
        rsi = latest['rsi']

        # RSI filter
        if h4_trend == TrendState.BULLISH:
            if not (self.config.min_rsi_long <= rsi <= self.config.max_rsi_long):
                logger.debug(
                    "RSI filter failed for long",
                    rsi=rsi,
                    min_rsi=self.config.min_rsi_long,
                    max_rsi=self.config.max_rsi_long
                )
                return None
        else:
            if not (self.config.min_rsi_short <= rsi <= self.config.max_rsi_short):
                logger.debug(
                    "RSI filter failed for short",
                    rsi=rsi,
                    min_rsi=self.config.min_rsi_short,
                    max_rsi=self.config.max_rsi_short
                )
                return None

        # Method A: EMA bounce
        entry_a = await self.check_ema_bounce_entry(df, h4_trend)
        if entry_a:
            return entry_a

        # Method B: Structure break
        entry_b = await self.check_structure_break_entry(df, h4_trend)
        if entry_b:
            return entry_b

        return None

    async def check_ema_bounce_entry(
        self,
        df: pd.DataFrame,
        trend: TrendState
    ) -> Optional[EntrySignal]:
        """
        Check for EMA bounce entry.

        Criteria:
        - Price pulled back to EMA21 in last 1-2 bars
        - Reversal candle formed (engulfing or hammer/shooting star)

        Args:
            df: 15M DataFrame
            trend: Trend direction

        Returns:
            EntrySignal if found, None otherwise
        """
        latest = df.iloc[-1]
        previous = df.iloc[-2]
        ema_21 = latest['ema_21']

        if trend == TrendState.BULLISH:
            # Check pullback to EMA
            pullback = (latest['low'] <= ema_21) or (previous['low'] <= ema_21)
            if not pullback:
                return None

            # Check reversal pattern
            if self.pattern_recognizer.is_bullish_engulfing(latest, previous):
                sl = await self.calculate_stop_loss(OrderDirection.BUY, latest['close'], df)
                return EntrySignal(
                    trigger_type="ema_bounce",
                    entry_price=latest['close'],
                    stop_loss=sl,
                    confidence=0.8,
                    candle_pattern="bullish_engulfing",
                    reason="Bullish engulfing at EMA21"
                )

            if self.pattern_recognizer.is_hammer(latest):
                sl = await self.calculate_stop_loss(OrderDirection.BUY, latest['close'], df)
                return EntrySignal(
                    trigger_type="ema_bounce",
                    entry_price=latest['close'],
                    stop_loss=sl,
                    confidence=0.75,
                    candle_pattern="hammer",
                    reason="Hammer at EMA21"
                )

        else:  # BEARISH
            # Check pullback to EMA
            pullback = (latest['high'] >= ema_21) or (previous['high'] >= ema_21)
            if not pullback:
                return None

            # Check reversal pattern
            if self.pattern_recognizer.is_bearish_engulfing(latest, previous):
                sl = await self.calculate_stop_loss(OrderDirection.SELL, latest['close'], df)
                return EntrySignal(
                    trigger_type="ema_bounce",
                    entry_price=latest['close'],
                    stop_loss=sl,
                    confidence=0.8,
                    candle_pattern="bearish_engulfing",
                    reason="Bearish engulfing at EMA21"
                )

            if self.pattern_recognizer.is_shooting_star(latest):
                sl = await self.calculate_stop_loss(OrderDirection.SELL, latest['close'], df)
                return EntrySignal(
                    trigger_type="ema_bounce",
                    entry_price=latest['close'],
                    stop_loss=sl,
                    confidence=0.75,
                    candle_pattern="shooting_star",
                    reason="Shooting star at EMA21"
                )

        return None

    async def check_structure_break_entry(
        self,
        df: pd.DataFrame,
        trend: TrendState
    ) -> Optional[EntrySignal]:
        """
        Check for structure break entry.

        Criteria:
        - For LONG: Price breaks above recent swing high after pullback
        - For SHORT: Price breaks below recent swing low after pullback

        Args:
            df: 15M DataFrame
            trend: Trend direction

        Returns:
            EntrySignal if found, None otherwise
        """
        if trend == TrendState.BULLISH:
            # Find recent swing low (pullback)
            swing_lows = df[df['swing_low'] == True].tail(3)
            if swing_lows.empty:
                return None

            swing_low_price = swing_lows.iloc[-1]['low']
            swing_low_idx = swing_lows.index[-1]

            # Find swing high before the swing low
            swing_highs_before = df[df['swing_high'] == True]
            swing_highs_before = swing_highs_before[swing_highs_before.index < swing_low_idx].tail(3)

            if swing_highs_before.empty:
                return None

            swing_high_price = swing_highs_before.iloc[-1]['high']

            # Check if latest bar broke above swing high
            latest = df.iloc[-1]
            if latest['close'] > swing_high_price:
                sl = await self.calculate_stop_loss(OrderDirection.BUY, latest['close'], df)
                return EntrySignal(
                    trigger_type="structure_break",
                    entry_price=latest['close'],
                    stop_loss=sl,
                    confidence=0.85,
                    candle_pattern=None,
                    reason=f"Break above {swing_high_price:.2f}"
                )

        else:  # BEARISH
            # Find recent swing high (pullback)
            swing_highs = df[df['swing_high'] == True].tail(3)
            if swing_highs.empty:
                return None

            swing_high_price = swing_highs.iloc[-1]['high']
            swing_high_idx = swing_highs.index[-1]

            # Find swing low before the swing high
            swing_lows_before = df[df['swing_low'] == True]
            swing_lows_before = swing_lows_before[swing_lows_before.index < swing_high_idx].tail(3)

            if swing_lows_before.empty:
                return None

            swing_low_price = swing_lows_before.iloc[-1]['low']

            # Check if latest bar broke below swing low
            latest = df.iloc[-1]
            if latest['close'] < swing_low_price:
                sl = await self.calculate_stop_loss(OrderDirection.SELL, latest['close'], df)
                return EntrySignal(
                    trigger_type="structure_break",
                    entry_price=latest['close'],
                    stop_loss=sl,
                    confidence=0.85,
                    candle_pattern=None,
                    reason=f"Break below {swing_low_price:.2f}"
                )

        return None

    async def calculate_stop_loss(
        self,
        direction: OrderDirection,
        entry_price: float,
        df: pd.DataFrame
    ) -> float:
        """
        Calculate stop loss based on recent swing + ATR buffer.

        For LONG:
        SL = recent_swing_low - (0.5 × ATR)

        For SHORT:
        SL = recent_swing_high + (0.5 × ATR)

        Validates:
        - Min: 1.0 × ATR
        - Max: 3.0 × ATR

        Args:
            direction: Order direction
            entry_price: Entry price
            df: DataFrame with indicators

        Returns:
            Stop loss price
        """
        atr = df.iloc[-1]['atr']

        if direction == OrderDirection.BUY:
            # Find recent swing low
            swing_lows = df[df['swing_low'] == True].tail(5)
            if not swing_lows.empty:
                swing_low = swing_lows.iloc[-1]['low']
                sl = swing_low - (self.config.sl_buffer_atr * atr)
            else:
                # Fallback
                sl = entry_price - (self.config.min_sl_atr * atr)

            # Validate distance
            sl_distance = entry_price - sl
            if sl_distance > self.config.max_sl_atr * atr:
                sl = entry_price - (self.config.max_sl_atr * atr)
            elif sl_distance < self.config.min_sl_atr * atr:
                sl = entry_price - (self.config.min_sl_atr * atr)

        else:  # SELL
            # Find recent swing high
            swing_highs = df[df['swing_high'] == True].tail(5)
            if not swing_highs.empty:
                swing_high = swing_highs.iloc[-1]['high']
                sl = swing_high + (self.config.sl_buffer_atr * atr)
            else:
                # Fallback
                sl = entry_price + (self.config.min_sl_atr * atr)

            # Validate distance
            sl_distance = sl - entry_price
            if sl_distance > self.config.max_sl_atr * atr:
                sl = entry_price + (self.config.max_sl_atr * atr)
            elif sl_distance < self.config.min_sl_atr * atr:
                sl = entry_price + (self.config.min_sl_atr * atr)

        logger.debug(
            "Stop loss calculated",
            direction=direction.value,
            entry=entry_price,
            sl=sl,
            distance_atr=(abs(entry_price - sl) / atr)
        )

        return sl

    def get_info(self) -> Dict[str, Any]:
        """Get strategy information"""
        return {
            **super().get_info(),
            "required_timeframes": ["H4", "H1", "M1"],
            "indicators": {
                "ema_200": self.config.ema_200,
                "ema_50": self.config.ema_50,
                "ema_21": self.config.ema_21,
                "hull_55": self.config.hull_55,
                "hull_34": self.config.hull_34,
                "rsi_period": self.config.rsi_period,
                "atr_period": self.config.atr_period
            },
            "risk_params": {
                "tp1_rr": self.config.tp1_rr,
                "tp2_rr": self.config.tp2_rr,
                "tp1_close_pct": self.config.tp1_close_percent,
                "tp2_close_pct": self.config.tp2_close_percent,
                "trail_pct": self.config.trail_percent
            }
        }
