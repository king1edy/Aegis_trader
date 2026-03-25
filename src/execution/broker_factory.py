"""
Broker Factory
==============
Factory for creating the appropriate broker connector based on environment and configuration.
"""

import sys
import structlog

from src.core.config import Settings
from src.core.exceptions import MT5ConnectionError

logger = structlog.get_logger()


async def create_broker(settings: Settings):
    """
    Create the appropriate broker connector based on settings and environment.
    
    Decision logic:
    1. If broker_mode == "direct": Use MT5Connector (requires Windows + MT5)
    2. If broker_mode == "bridge": Use MT5APIClient (connects to bridge server)
    3. If broker_mode == "paper": Use PaperTradingBroker (simulation)
    4. If broker_mode == "auto":
       - On Windows with MT5 available: Use MT5Connector
       - Otherwise: Try bridge, then fall back to paper trading
    
    Args:
        settings: Application settings
        
    Returns:
        Broker connector instance (MT5Connector, MT5APIClient, or PaperTradingBroker)
        
    Raises:
        MT5ConnectionError: If no suitable broker can be created
    """
    mode = settings.broker_mode.lower()
    
    logger.info("Creating broker", mode=mode)
    
    if mode == "direct":
        return await _create_direct_broker(settings)
    
    elif mode == "bridge":
        return await _create_bridge_broker(settings)
    
    elif mode == "paper":
        return await _create_paper_broker(settings)
    
    elif mode == "auto":
        return await _auto_select_broker(settings)
    
    else:
        raise ValueError(f"Unknown broker mode: {mode}")


async def _create_direct_broker(settings: Settings):
    """Create direct MT5 connector (Windows only)."""
    if sys.platform != "win32":
        raise MT5ConnectionError(
            "Direct MT5 connection requires Windows. "
            "Use broker_mode='bridge' to connect via API bridge, "
            "or broker_mode='paper' for paper trading."
        )
    
    try:
        from src.execution.mt5_connector import MT5Connector
        broker = MT5Connector(
            login=settings.mt5_login,
            password=settings.mt5_password,
            server=settings.mt5_server,
            path=settings.mt5_path
        )
        logger.info("Created direct MT5 connector")
        return broker
    except ImportError as e:
        raise MT5ConnectionError(f"Failed to import MT5Connector: {e}")


async def _create_bridge_broker(settings: Settings):
    """Create MT5 API bridge client, with fallback to paper trading if MT5 not connected."""
    import httpx
    
    # Check if the bridge is running AND MT5 is connected
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{settings.mt5_bridge_url}/health")
            if response.status_code == 200:
                health = response.json()
                if health.get("mt5_connected"):
                    from src.execution.mt5_api_client import MT5APIClient
                    broker = MT5APIClient(settings)
                    logger.info("Created MT5 API bridge client", url=settings.mt5_bridge_url)
                    return broker
                else:
                    logger.warning(
                        "MT5 bridge is running but MT5 is not connected. "
                        "Falling back to paper trading. "
                        "To use live trading, ensure MT5 is running on the bridge host."
                    )
    except httpx.RequestError as e:
        logger.warning(
            "Could not reach MT5 bridge, falling back to paper trading",
            url=settings.mt5_bridge_url,
            error=str(e)
        )
    except Exception as e:
        logger.warning("Error checking MT5 bridge health", error=str(e))
    
    # Fall back to paper trading
    return await _create_paper_broker(settings)


async def _create_paper_broker(settings: Settings):
    """Create paper trading broker for simulation."""
    from src.execution.paper_broker import PaperTradingBroker
    
    broker = PaperTradingBroker(settings)
    logger.info("Created paper trading broker")
    return broker


async def _auto_select_broker(settings: Settings):
    """
    Automatically select the best broker based on environment.
    
    Priority:
    1. Direct MT5 (if on Windows and MT5 is available)
    2. Bridge (if bridge URL is reachable)
    3. Paper trading (fallback)
    """
    logger.info("Auto-selecting broker...")
    
    # Try direct MT5 on Windows
    if sys.platform == "win32":
        try:
            # Check if MT5 module is available
            import MetaTrader5
            from src.execution.mt5_connector import MT5Connector
            
            broker = MT5Connector(
                login=settings.mt5_login,
                password=settings.mt5_password,
                server=settings.mt5_server,
                path=settings.mt5_path
            )
            logger.info("Auto-selected: Direct MT5 connector (Windows)")
            return broker
            
        except ImportError:
            logger.debug("MT5 module not available, trying bridge...")
        except Exception as e:
            logger.debug("Direct MT5 not available", error=str(e))
    
    # Try bridge connection
    try:
        import httpx
        
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{settings.mt5_bridge_url}/health")
            if response.status_code == 200:
                health = response.json()
                if health.get("mt5_connected"):
                    from src.execution.mt5_api_client import MT5APIClient
                    broker = MT5APIClient(settings)
                    logger.info("Auto-selected: MT5 API bridge client", url=settings.mt5_bridge_url)
                    return broker
                else:
                    logger.warning("Bridge is running but MT5 not connected")
    except Exception as e:
        logger.debug("Bridge not available", error=str(e))
    
    # Fall back to paper trading
    logger.warning("No live broker available, falling back to paper trading")
    from src.execution.paper_broker import PaperTradingBroker
    broker = PaperTradingBroker(settings)
    logger.info("Auto-selected: Paper trading broker")
    return broker


def get_broker_status_message(broker) -> str:
    """Get a human-readable status message for the broker."""
    broker_type = type(broker).__name__
    
    if broker_type == "MT5Connector":
        return "🟢 Live trading via direct MT5 connection"
    elif broker_type == "MT5APIClient":
        return "🟢 Live trading via MT5 API bridge"
    elif broker_type == "PaperTradingBroker":
        return "📝 Paper trading mode (no real orders)"
    else:
        return f"❓ Unknown broker type: {broker_type}"
