"""
Position Manager

Manages open positions with scaled exits and trailing stops.

Position State Machine:
INITIAL → TP1_HIT → TP2_HIT → TRAILING

Exit triggers:
- TP1 hit: Close 50%, move SL to BE → TP1_HIT
- TP2 hit: Close 30%, leave 20% → TP2_HIT
- Trailing: Update SL on 1H bar, exit on Hull flip
- Time exit: Close all if no TP hit after 8 hours
"""

from datetime import datetime, timedelta, timezone
from typing import Optional
from decimal import Decimal

from src.execution.mt5_connector import BrokerInterface, OrderDirection
from src.database.repository import TradeRepository, get_session
from src.database.models import Trade, TradeStatus, PartialClose, TradeModification, OrderType as DBOrderType
from src.strategies.data_manager import MultiTimeframeDataManager
from src.strategies.indicators import IndicatorCalculator, IndicatorConfig
from src.strategies.mtftr import MTFTRConfig
from src.core.logging_config import get_logger

logger = get_logger("position_manager")


class PositionManager:
    """
    Manage open positions with scaled exits and trailing stops.

    Responsibilities:
    - Check TP1/TP2 hits and execute partial closes
    - Move SL to breakeven at TP1
    - Trail SL on 1H EMA50
    - Exit on Hull MA flip
    - Time-based exit if no TP hit
    """

    def __init__(
        self,
        broker: BrokerInterface,
        data_manager: MultiTimeframeDataManager,
        indicator_calc: IndicatorCalculator,
        config: MTFTRConfig
    ):
        """
        Initialize position manager.

        Args:
            broker: Broker interface
            data_manager: Data manager for price data
            indicator_calc: Indicator calculator
            config: Strategy configuration
        """
        self.broker = broker
        self.data_manager = data_manager
        self.indicator_calc = indicator_calc
        self.config = config

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

        logger.info("Position manager initialized")

    async def manage_positions(self) -> None:
        """
        Main position management loop.

        Called every tick from main trading loop.
        Checks all open positions and takes appropriate actions.
        """
        async with get_session() as session:
            trade_repo = TradeRepository(session)
            open_trades = await trade_repo.get_open_trades()

            for trade in open_trades:
                try:
                    await self._manage_single_position(trade, trade_repo)
                except Exception as e:
                    logger.exception(
                        "Error managing position",
                        ticket=trade.ticket,
                        symbol=trade.symbol,
                        error=str(e)
                    )

    async def _manage_single_position(
        self,
        trade: Trade,
        trade_repo: TradeRepository
    ) -> None:
        """
        Manage a single position.

        Args:
            trade: Trade record
            trade_repo: Trade repository
        """
        # Get current price
        current_tick = await self.broker.get_current_tick(trade.symbol)

        # Get position state (stored in strategy_data for now)
        position_state = self._get_position_state(trade)

        # Check time exit (if still in INITIAL state)
        if position_state == "initial":
            if await self._check_time_exit(trade):
                await self._execute_full_close(trade, trade_repo, "time_limit")
                return

        # Check TP levels based on state
        if position_state == "initial":
            if await self._check_tp1_hit(trade, current_tick):
                await self._handle_tp1(trade, trade_repo, current_tick)

        elif position_state == "tp1_hit":
            if await self._check_tp2_hit(trade, current_tick):
                await self._handle_tp2(trade, trade_repo, current_tick)

        elif position_state == "tp2_hit":
            # Update trailing stop on 1H bar
            if await self.data_manager.is_new_bar(trade.symbol, "H1"):
                await self._update_trailing_stop(trade, trade_repo)

            # Check for trail exit conditions
            await self._check_trail_exit(trade, trade_repo, current_tick)

    def _get_position_state(self, trade: Trade) -> str:
        """
        Get current position state.

        Args:
            trade: Trade record

        Returns:
            Position state: "initial", "tp1_hit", "tp2_hit", or "trailing"
        """
        if trade.strategy_data and "position_state" in trade.strategy_data:
            return trade.strategy_data["position_state"]
        return "initial"

    def _set_position_state(self, trade: Trade, state: str) -> None:
        """
        Set position state.

        Args:
            trade: Trade record
            state: New state
        """
        if trade.strategy_data is None:
            trade.strategy_data = {}
        trade.strategy_data["position_state"] = state

    async def _check_time_exit(self, trade: Trade) -> bool:
        """
        Check if position should be closed due to time limit.

        Args:
            trade: Trade record

        Returns:
            True if time limit exceeded
        """
        if not trade.entry_time:
            return False

        max_duration = timedelta(hours=self.config.max_trade_duration_hours)
        current_time = datetime.now(timezone.utc)
        trade_duration = current_time - trade.entry_time

        if trade_duration > max_duration:
            logger.info(
                "Time limit exceeded",
                ticket=trade.ticket,
                duration_hours=trade_duration.total_seconds() / 3600,
                max_hours=self.config.max_trade_duration_hours
            )
            return True

        return False

    async def _check_tp1_hit(self, trade: Trade, current_tick) -> bool:
        """
        Check if TP1 has been hit.

        Args:
            trade: Trade record
            current_tick: Current price tick

        Returns:
            True if TP1 hit
        """
        if not trade.take_profit_1:
            return False

        tp1 = float(trade.take_profit_1)

        if trade.order_type in [DBOrderType.BUY, DBOrderType.BUY_LIMIT, DBOrderType.BUY_STOP]:
            # Long position - check if bid >= TP1
            return current_tick.bid >= tp1
        else:
            # Short position - check if ask <= TP1
            return current_tick.ask <= tp1

    async def _check_tp2_hit(self, trade: Trade, current_tick) -> bool:
        """
        Check if TP2 has been hit.

        Args:
            trade: Trade record
            current_tick: Current price tick

        Returns:
            True if TP2 hit
        """
        if not trade.take_profit_2:
            return False

        tp2 = float(trade.take_profit_2)

        if trade.order_type in [DBOrderType.BUY, DBOrderType.BUY_LIMIT, DBOrderType.BUY_STOP]:
            return current_tick.bid >= tp2
        else:
            return current_tick.ask <= tp2

    async def _handle_tp1(
        self,
        trade: Trade,
        trade_repo: TradeRepository,
        current_tick
    ) -> None:
        """
        Handle TP1 hit:
        1. Close 50% of position
        2. Move SL to breakeven
        3. Update state to TP1_HIT

        Args:
            trade: Trade record
            trade_repo: Trade repository
            current_tick: Current price tick
        """
        logger.info(
            "TP1 hit - executing partial close",
            ticket=trade.ticket,
            tp1=float(trade.take_profit_1)
        )

        # Calculate close volume (50% of initial)
        close_volume = float(trade.initial_lot_size) * self.config.tp1_close_percent

        # Close partial position
        result = await self.broker.close_position(
            ticket=trade.ticket,
            volume=close_volume
        )

        if result.success:
            # Record partial close
            partial = PartialClose(
                trade_id=trade.id,
                close_time=datetime.now(timezone.utc),
                close_price=Decimal(str(result.price)),
                lots_closed=Decimal(str(close_volume)),
                profit_loss=Decimal("0"),  # Will be calculated later
                reason="TP1"
            )
            await trade_repo.add_partial_close(partial)

            # Move SL to breakeven
            breakeven_sl = float(trade.entry_price)
            modify_result = await self.broker.modify_position(
                ticket=trade.ticket,
                sl=breakeven_sl,
                tp=None
            )

            if modify_result.success:
                # Record modification
                modification = TradeModification(
                    trade_id=trade.id,
                    modification_time=datetime.now(timezone.utc),
                    old_sl=trade.stop_loss,
                    new_sl=Decimal(str(breakeven_sl)),
                    old_tp=trade.take_profit_1,
                    new_tp=None,
                    reason="TP1_hit_move_to_BE"
                )
                await trade_repo.add_modification(modification)

                # Update trade record
                trade.stop_loss = Decimal(str(breakeven_sl))
                trade.lot_size = Decimal(str(float(trade.lot_size) - close_volume))
                self._set_position_state(trade, "tp1_hit")
                await trade_repo.update(trade)

                logger.info(
                    "TP1 handled successfully",
                    ticket=trade.ticket,
                    closed_lots=close_volume,
                    new_sl=breakeven_sl,
                    remaining_lots=float(trade.lot_size)
                )
            else:
                logger.error(
                    "Failed to move SL to breakeven",
                    ticket=trade.ticket,
                    error=modify_result.error_message
                )
        else:
            logger.error(
                "Failed to close partial position at TP1",
                ticket=trade.ticket,
                error=result.error_message
            )

    async def _handle_tp2(
        self,
        trade: Trade,
        trade_repo: TradeRepository,
        current_tick
    ) -> None:
        """
        Handle TP2 hit:
        1. Close 30% of original position
        2. Leave 20% for trailing
        3. Update state to TP2_HIT

        Args:
            trade: Trade record
            trade_repo: Trade repository
            current_tick: Current price tick
        """
        logger.info(
            "TP2 hit - executing second partial close",
            ticket=trade.ticket,
            tp2=float(trade.take_profit_2)
        )

        # Calculate close volume (30% of initial)
        close_volume = float(trade.initial_lot_size) * self.config.tp2_close_percent

        # Close partial position
        result = await self.broker.close_position(
            ticket=trade.ticket,
            volume=close_volume
        )

        if result.success:
            # Record partial close
            partial = PartialClose(
                trade_id=trade.id,
                close_time=datetime.now(timezone.utc),
                close_price=Decimal(str(result.price)),
                lots_closed=Decimal(str(close_volume)),
                profit_loss=Decimal("0"),  # Will be calculated later
                reason="TP2"
            )
            await trade_repo.add_partial_close(partial)

            # Update trade record
            trade.lot_size = Decimal(str(float(trade.lot_size) - close_volume))
            self._set_position_state(trade, "tp2_hit")
            await trade_repo.update(trade)

            logger.info(
                "TP2 handled successfully",
                ticket=trade.ticket,
                closed_lots=close_volume,
                remaining_lots=float(trade.lot_size)
            )
        else:
            logger.error(
                "Failed to close partial position at TP2",
                ticket=trade.ticket,
                error=result.error_message
            )

    async def _update_trailing_stop(
        self,
        trade: Trade,
        trade_repo: TradeRepository
    ) -> None:
        """
        Update trailing stop on 1H EMA50.

        For LONG: new_sl = 1H_EMA50 - (0.5 × ATR)
        For SHORT: new_sl = 1H_EMA50 + (0.5 × ATR)

        Only move if:
        - new_sl > current_sl (for longs)
        - new_sl < current_price (don't trigger immediately)

        Args:
            trade: Trade record
            trade_repo: Trade repository
        """
        # Get 1H data with indicators
        df_h1 = await self.data_manager.get_data(trade.symbol, "H1", count=50)
        df_h1 = self.indicator_calc.calculate_all(df_h1, self.indicator_config)

        latest = df_h1.iloc[-1]
        ema_50 = latest['ema_50']
        atr = latest['atr']

        # Get current price
        current_tick = await self.broker.get_current_tick(trade.symbol)

        current_sl = float(trade.stop_loss)
        is_long = trade.order_type in [DBOrderType.BUY, DBOrderType.BUY_LIMIT, DBOrderType.BUY_STOP]

        if is_long:
            new_sl = ema_50 - (0.5 * atr)
            should_update = new_sl > current_sl and new_sl < current_tick.bid

            if should_update:
                result = await self.broker.modify_position(
                    ticket=trade.ticket,
                    sl=new_sl,
                    tp=None
                )

                if result.success:
                    # Record modification
                    modification = TradeModification(
                        trade_id=trade.id,
                        modification_time=datetime.now(timezone.utc),
                        old_sl=trade.stop_loss,
                        new_sl=Decimal(str(new_sl)),
                        old_tp=None,
                        new_tp=None,
                        reason="trailing_stop_update"
                    )
                    await trade_repo.add_modification(modification)

                    trade.stop_loss = Decimal(str(new_sl))
                    trade.trailing_stop = Decimal(str(new_sl))
                    await trade_repo.update(trade)

                    logger.info(
                        "Trailing stop updated",
                        ticket=trade.ticket,
                        old_sl=current_sl,
                        new_sl=new_sl,
                        ema_50=ema_50
                    )

        else:  # SHORT
            new_sl = ema_50 + (0.5 * atr)
            should_update = new_sl < current_sl and new_sl > current_tick.ask

            if should_update:
                result = await self.broker.modify_position(
                    ticket=trade.ticket,
                    sl=new_sl,
                    tp=None
                )

                if result.success:
                    # Record modification
                    modification = TradeModification(
                        trade_id=trade.id,
                        modification_time=datetime.now(timezone.utc),
                        old_sl=trade.stop_loss,
                        new_sl=Decimal(str(new_sl)),
                        old_tp=None,
                        new_tp=None,
                        reason="trailing_stop_update"
                    )
                    await trade_repo.add_modification(modification)

                    trade.stop_loss = Decimal(str(new_sl))
                    trade.trailing_stop = Decimal(str(new_sl))
                    await trade_repo.update(trade)

                    logger.info(
                        "Trailing stop updated",
                        ticket=trade.ticket,
                        old_sl=current_sl,
                        new_sl=new_sl,
                        ema_50=ema_50
                    )

    async def _check_trail_exit(
        self,
        trade: Trade,
        trade_repo: TradeRepository,
        current_tick
    ) -> None:
        """
        Check if trailing position should be exited:
        1. Hull MA flips color
        2. Price closes below/above 1H EMA50

        Args:
            trade: Trade record
            trade_repo: Trade repository
            current_tick: Current price tick
        """
        # Get 1H data with indicators
        df_h1 = await self.data_manager.get_data(trade.symbol, "H1", count=50)
        df_h1 = self.indicator_calc.calculate_all(df_h1, self.indicator_config)

        latest = df_h1.iloc[-1]
        previous = df_h1.iloc[-2]

        hull_34 = latest['hull_34']
        hull_34_old = previous['hull_34']
        ema_50 = latest['ema_50']
        close_price = latest['close']

        should_exit = False
        reason = ""

        is_long = trade.order_type in [DBOrderType.BUY, DBOrderType.BUY_LIMIT, DBOrderType.BUY_STOP]

        if is_long:
            # Hull flipped bearish
            if hull_34 < hull_34_old:
                should_exit = True
                reason = "hull_flip_bearish"
            # Closed below EMA50
            elif close_price < ema_50:
                should_exit = True
                reason = "close_below_ema50"
        else:  # SHORT
            # Hull flipped bullish
            if hull_34 > hull_34_old:
                should_exit = True
                reason = "hull_flip_bullish"
            # Closed above EMA50
            elif close_price > ema_50:
                should_exit = True
                reason = "close_above_ema50"

        if should_exit:
            logger.info(
                "Trail exit condition met",
                ticket=trade.ticket,
                reason=reason,
                hull_34=hull_34,
                hull_34_old=hull_34_old,
                close=close_price,
                ema_50=ema_50
            )
            await self._execute_full_close(trade, trade_repo, reason)

    async def _execute_full_close(
        self,
        trade: Trade,
        trade_repo: TradeRepository,
        reason: str
    ) -> None:
        """
        Close entire remaining position.

        Args:
            trade: Trade record
            trade_repo: Trade repository
            reason: Exit reason
        """
        logger.info(
            "Closing full position",
            ticket=trade.ticket,
            reason=reason,
            remaining_lots=float(trade.lot_size)
        )

        result = await self.broker.close_position(
            ticket=trade.ticket,
            volume=None  # Close all
        )

        if result.success:
            trade.status = TradeStatus.CLOSED
            trade.exit_price = Decimal(str(result.price))
            trade.exit_time = datetime.now(timezone.utc)
            trade.exit_reason = reason
            await trade_repo.update(trade)

            logger.info(
                "Position closed successfully",
                ticket=trade.ticket,
                exit_price=result.price,
                reason=reason
            )
        else:
            logger.error(
                "Failed to close position",
                ticket=trade.ticket,
                error=result.error_message
            )
