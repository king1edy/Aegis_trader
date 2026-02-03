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
from src.core.logging_config import setup_logging, get_logger, LogMessages
from src.core.exceptions import TradingSystemError
from src.database import init_database, close_database, get_session
from src.execution import get_broker_connector, BrokerInterface

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
        
        This is where strategy signals are processed and trades executed.
        For Phase 1, this is a placeholder that logs account status.
        """
        # Get current account state
        account = await self.broker.get_account_info()
        positions = await self.broker.get_positions()
        
        # Log periodic status (every minute in production)
        # For now, log more frequently for debugging
        logger.debug(
            "Trading iteration",
            balance=float(account.balance),
            equity=float(account.equity),
            open_positions=len(positions),
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
        # TODO: Phase 2 - Add strategy signal processing
        # TODO: Phase 3 - Add risk management checks
        # TODO: Phase 4 - Add execution logic


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
