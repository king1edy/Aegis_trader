"""
Trading Automation System - Main Entry Point
=============================================

This is the main entry point for the trading automation system.
It initializes all components and starts the trading loop.
"""

import asyncio
import signal
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

from src.core.config import settings
from src.core.logging_config import setup_logging, get_logger, LogMessages, TradingContextLogger
from src.core.exceptions import TradingSystemError
from src.database import init_database, close_database, get_session
from src.database.repository import TradeRepository, SignalRepository, PerformanceRepository, SystemRepository
from src.database.models import Trade, Signal, SignalSource, TradeStatus, OrderType
from src.execution import get_broker_connector, BrokerInterface
from src.strategies.data_manager import MultiTimeframeDataManager
from src.strategies.indicators import IndicatorCalculator
from src.strategies.filters.session_filter import SessionFilter
from src.strategies.mtftr import MTFTRStrategy, MTFTRConfig
from src.strategies.position_manager import PositionManager
from src.risk.position_sizer import PositionSizer
from src.risk.risk_checker import RiskChecker

# Initialize logging
logger = setup_logging()


class TradingSystem:
    """
    Main trading system orchestrator.
    
    Coordinates all components: broker connection, strategies,
    risk management, and execution.
    """
    
    def __init__(self):
        self.broker: Optional[BrokerInterface] = None
        self.running = False
        self._shutdown_event = asyncio.Event()

        # MTFTR Strategy components
        self.data_manager: Optional[MultiTimeframeDataManager] = None
        self.indicator_calc: Optional[IndicatorCalculator] = None
        self.session_filter: Optional[SessionFilter] = None
        self.strategy: Optional[MTFTRStrategy] = None
        self.position_manager: Optional[PositionManager] = None
        self.position_sizer: Optional[PositionSizer] = None
        self.risk_checker: Optional[RiskChecker] = None
    
    async def initialize(self) -> None:
        """Initialize all system components."""
        logger.info(LogMessages.SYSTEM_STARTED, version="1.0.0", env=settings.app_env.value)
        
        # Validate configuration
        warnings = settings.validate_trading_config()
        for warning in warnings:
            logger.warning("Configuration warning", message=warning)
        
        # Initialize database
        logger.info("Initializing database...")
        await init_database()
        
        # Connect to broker
        logger.info("Connecting to broker...")
        self.broker = get_broker_connector(demo=settings.app_env.value != "production")
        await self.broker.connect()
        
        # Log account info
        account = await self.broker.get_account_info()
        logger.info(
            "Account loaded",
            balance=float(account.balance),
            equity=float(account.equity),
            leverage=account.leverage,
            currency=account.currency
        )

        # Initialize MTFTR strategy components
        if settings.mtftr_enabled:
            logger.info("Initializing MTFTR strategy components...")

            async with get_session() as session:
                trade_repo = TradeRepository(session)
                perf_repo = PerformanceRepository(session)
                system_repo = SystemRepository(session)

                # Initialize components
                self.data_manager = MultiTimeframeDataManager(
                    broker=self.broker,
                    cache_ttl_seconds=60
                )

                self.indicator_calc = IndicatorCalculator()

                self.session_filter = SessionFilter(
                    london_start=settings.london_session_start,
                    london_end=settings.london_session_end,
                    ny_start=settings.ny_session_start,
                    ny_end=settings.ny_session_end
                )

                self.position_sizer = PositionSizer(settings)
                self.risk_checker = RiskChecker(settings, trade_repo, perf_repo, system_repo)

                # MTFTR strategy
                strategy_config = MTFTRConfig(
                    name="MTFTR",
                    symbol=settings.default_symbol,
                    enabled=True,
                    max_positions=settings.max_open_positions,
                    max_daily_trades=settings.max_daily_trades,
                    risk_per_trade=settings.max_risk_per_trade,
                    # Indicator periods
                    ema_200=settings.mtftr_ema_200,
                    ema_50=settings.mtftr_ema_50,
                    ema_21=settings.mtftr_ema_21,
                    hull_55=settings.mtftr_hull_55,
                    hull_34=settings.mtftr_hull_34,
                    rsi_period=settings.mtftr_rsi_period,
                    atr_period=settings.mtftr_atr_period,
                    swing_lookback=settings.mtftr_swing_lookback,
                    # Risk parameters
                    tp1_rr=settings.mtftr_tp1_rr,
                    tp2_rr=settings.mtftr_tp2_rr,
                    tp1_close_percent=settings.mtftr_tp1_close_percent,
                    tp2_close_percent=settings.mtftr_tp2_close_percent,
                    trail_percent=settings.mtftr_trail_percent,
                    # Entry filters
                    min_rsi_long=settings.mtftr_min_rsi_long,
                    max_rsi_long=settings.mtftr_max_rsi_long,
                    min_rsi_short=settings.mtftr_min_rsi_short,
                    max_rsi_short=settings.mtftr_max_rsi_short,
                    # Stop loss limits
                    min_sl_atr=settings.mtftr_min_sl_atr,
                    max_sl_atr=settings.mtftr_max_sl_atr,
                    sl_buffer_atr=settings.mtftr_sl_buffer_atr,
                    # Limits
                    max_trade_duration_hours=settings.mtftr_max_trade_hours
                )

                self.strategy = MTFTRStrategy(
                    config=strategy_config,
                    broker=self.broker,
                    data_manager=self.data_manager,
                    indicator_calc=self.indicator_calc,
                    session_filter=self.session_filter
                )

                self.position_manager = PositionManager(
                    broker=self.broker,
                    data_manager=self.data_manager,
                    indicator_calc=self.indicator_calc,
                    config=strategy_config
                )

            logger.info("MTFTR strategy initialized successfully")

        logger.info("System initialization complete")
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the system."""
        logger.info("Initiating shutdown...")
        self.running = False
        self._shutdown_event.set()
        
        # Disconnect from broker
        if self.broker:
            await self.broker.disconnect()
        
        # Close database connections
        await close_database()
        
        logger.info(LogMessages.SYSTEM_STOPPED)
    
    async def run(self) -> None:
        """Main trading loop."""
        self.running = True
        
        logger.info("Starting main trading loop...")
        
        try:
            while self.running:
                try:
                    # Check broker connection
                    if not await self.broker.is_connected():
                        logger.warning(LogMessages.CONNECTION_LOST)
                        await self.broker.connect()
                        logger.info(LogMessages.CONNECTION_RESTORED)
                    
                    # Main loop iteration
                    await self._trading_iteration()
                    
                    # Wait before next iteration (configurable)
                    await asyncio.sleep(1)
                    
                except asyncio.CancelledError:
                    logger.info("Trading loop cancelled")
                    break
                except TradingSystemError as e:
                    logger.error("Trading error", error=str(e), details=e.details)
                    await asyncio.sleep(5)  # Brief pause before retry
                except Exception as e:
                    logger.exception("Unexpected error in trading loop", error=str(e))
                    await asyncio.sleep(10)  # Longer pause for unexpected errors
                    
        finally:
            await self.shutdown()
    
    async def _trading_iteration(self) -> None:
        """
        Single iteration of the trading loop.

        Main trading logic:
        1. Manage existing positions (every tick)
        2. Look for new signals (only on new 15M bar)
        3. Validate signals against risk limits
        4. Execute approved signals
        """
        # Get current account state
        account = await self.broker.get_account_info()

        # PHASE 1: Manage existing positions (every tick)
        if self.position_manager:
            await self.position_manager.manage_positions()

        # PHASE 2: Look for new signals (only on new 15M bar)
        if not self.strategy or not self.data_manager:
            await asyncio.sleep(1)
            return

        # Check for new 15M bar
        if not await self.data_manager.is_new_bar(settings.default_symbol, "M15"):
            await asyncio.sleep(1)
            return

        async with get_session() as session:
            trade_repo = TradeRepository(session)
            signal_repo = SignalRepository(session)

            # Check if we have room for more positions
            open_trades = await trade_repo.get_open_trades(symbol=settings.default_symbol)

            if len(open_trades) >= self.strategy.config.max_positions:
                logger.debug(
                    "Max positions reached",
                    open_positions=len(open_trades),
                    max_positions=self.strategy.config.max_positions
                )
                return

            try:
                # Generate signal
                signal = await self.strategy.analyze(settings.default_symbol)

                if not signal:
                    return

                # Save signal to database
                db_signal = Signal(
                    timestamp=signal.timestamp,
                    symbol=signal.symbol,
                    strategy_name="MTFTR",
                    signal_source=SignalSource.HULL_SUITE,
                    direction=signal.direction.value,
                    proposed_entry=signal.entry_price,
                    proposed_sl=signal.stop_loss,
                    proposed_tp=signal.take_profit_1,
                    confidence=signal.confidence,
                    market_context=signal.market_context,
                    was_executed=False
                )
                await signal_repo.save(db_signal)

                logger.info(
                    "Signal generated",
                    symbol=signal.symbol,
                    direction=signal.direction.value,
                    entry=signal.entry_price,
                    sl=signal.stop_loss,
                    tp1=signal.take_profit_1,
                    tp2=signal.take_profit_2,
                    reason=signal.reason,
                    confidence=signal.confidence
                )

                # Check risk limits
                can_trade, rejection_reason = await self.risk_checker.check_all_limits(signal, account)

                if not can_trade:
                    logger.info("Signal rejected by risk check", reason=rejection_reason)
                    return

                # Calculate position size
                symbol_info = await self.broker.get_symbol_info(signal.symbol)
                lot_size = await self.position_sizer.calculate_lot_size(
                    symbol=signal.symbol,
                    entry_price=signal.entry_price,
                    stop_loss=signal.stop_loss,
                    account_balance=float(account.balance),
                    risk_percent=settings.max_risk_per_trade,
                    symbol_info=symbol_info
                )

                # Execute trade
                await self._execute_signal(signal, lot_size, trade_repo, signal_repo, db_signal)

            except Exception as e:
                logger.exception("Error in strategy analysis", error=str(e))

        await asyncio.sleep(1)

    async def _execute_signal(
        self,
        signal,
        lot_size: float,
        trade_repo: TradeRepository,
        signal_repo: SignalRepository,
        db_signal: Signal
    ) -> None:
        """Execute a trading signal."""

        with TradingContextLogger(symbol=signal.symbol, strategy="MTFTR"):
            try:
                result = await self.broker.open_position(
                    symbol=signal.symbol,
                    direction=signal.direction,
                    volume=lot_size,
                    sl=signal.stop_loss,
                    tp=None,  # We manage TPs manually
                    comment="MTFTR",
                    magic=123456
                )

                if result.success:
                    # Create trade record
                    trade = Trade(
                        ticket=result.ticket,
                        symbol=signal.symbol,
                        order_type=OrderType.BUY if signal.direction.value == "BUY" else OrderType.SELL,
                        status=TradeStatus.OPEN,
                        signal_source=SignalSource.HULL_SUITE,
                        strategy_name="MTFTR",
                        signal_time=signal.timestamp,
                        entry_price=result.price,
                        entry_time=datetime.now(timezone.utc),
                        lot_size=lot_size,
                        initial_lot_size=lot_size,
                        stop_loss=signal.stop_loss,
                        take_profit_1=signal.take_profit_1,
                        take_profit_2=signal.take_profit_2,
                        take_profit_final=signal.take_profit_final,
                        position_state="initial",
                        account_balance_before=float((await self.broker.get_account_info()).balance),
                        market_context=signal.market_context,
                        strategy_data=signal.strategy_data
                    )

                    await trade_repo.create(trade)

                    # Mark signal as executed
                    db_signal.was_executed = True
                    db_signal.execution_price = result.price
                    db_signal.execution_time = datetime.now(timezone.utc)
                    await signal_repo.save(db_signal)

                    logger.info(
                        "Trade opened successfully",
                        ticket=result.ticket,
                        direction=signal.direction.value,
                        entry=result.price,
                        sl=signal.stop_loss,
                        lot_size=lot_size
                    )
                else:
                    logger.error("Trade rejected by broker", reason=result.error_message)

            except Exception as e:
                logger.exception("Failed to execute signal", error=str(e))


async def main() -> None:
    """Main entry point."""
    system = TradingSystem()
    
    # Setup signal handlers for graceful shutdown (Unix only)
    if sys.platform != "win32":
        loop = asyncio.get_event_loop()
        
        def signal_handler(sig):
            logger.info(f"Received signal {sig.name}")
            asyncio.create_task(system.shutdown())
        
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, lambda s=sig: signal_handler(s))
    
    try:
        await system.initialize()
        await system.run()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.exception("Fatal error", error=str(e))
        sys.exit(1)
    finally:
        await system.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
