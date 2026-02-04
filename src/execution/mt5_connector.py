"""
MetaTrader 5 Connector
======================
Handles connection and communication with MetaTrader 5.

IMPORTANT: The MetaTrader5 Python package only works on Windows.
For Linux/Docker deployment, you have two options:

1. Run this module on a Windows machine and expose via API
2. Use the MT5 WebSocket bridge approach

This module supports both approaches through the abstract interface.
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Any

from src.core.config import settings
from src.core.logging_config import get_logger, TradingContextLogger
from src.core.exceptions import (
    MT5ConnectionError,
    OrderExecutionError,
    PositionNotFoundError,
    SymbolNotFoundError,
    InsufficientMarginError,
)

# Import HistoricalDataLoader for demo connector
try:
    from src.data.historical_loader import HistoricalDataLoader
    HISTORICAL_DATA_AVAILABLE = True
except ImportError:
    HistoricalDataLoader = None
    HISTORICAL_DATA_AVAILABLE = False

logger = get_logger("mt5_connector")


# =============================================================================
# Data Classes
# =============================================================================

class OrderDirection(Enum):
    """Order direction."""
    BUY = "BUY"
    SELL = "SELL"


class OrderFillType(Enum):
    """Order fill type."""
    FOK = "FOK"  # Fill or Kill
    IOC = "IOC"  # Immediate or Cancel
    RETURN = "RETURN"  # Return remaining


@dataclass
class SymbolInfo:
    """Trading symbol information."""
    name: str
    description: str
    digits: int
    point: float
    spread: int
    tick_size: float
    tick_value: float
    min_lot: float
    max_lot: float
    lot_step: float
    contract_size: float
    margin_required: float
    trade_allowed: bool
    
    def normalize_price(self, price: float) -> float:
        """Round price to symbol's precision."""
        return round(price, self.digits)
    
    def normalize_lot(self, lot: float) -> float:
        """Round lot size to valid step."""
        steps = round(lot / self.lot_step)
        return round(steps * self.lot_step, 2)


@dataclass
class AccountInfo:
    """Trading account information."""
    login: int
    name: str
    server: str
    currency: str
    balance: Decimal
    equity: Decimal
    margin: Decimal
    free_margin: Decimal
    margin_level: Optional[float]
    profit: Decimal
    leverage: int


@dataclass
class Position:
    """Open position information."""
    ticket: int
    symbol: str
    direction: OrderDirection
    volume: float
    price_open: float
    price_current: float
    sl: Optional[float]
    tp: Optional[float]
    profit: float
    swap: float
    commission: float
    time_open: datetime
    magic: int
    comment: str


@dataclass
class PriceData:
    """OHLCV price data."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    tick_volume: int
    spread: int
    real_volume: int


@dataclass
class Tick:
    """Current price tick."""
    symbol: str
    timestamp: datetime
    bid: float
    ask: float
    last: float
    volume: int


@dataclass
class TradeResult:
    """Result of a trade operation."""
    success: bool
    ticket: Optional[int]
    order_id: Optional[int]
    volume: float
    price: float
    comment: str
    error_code: Optional[int]
    error_message: Optional[str]


# =============================================================================
# Abstract Broker Interface
# =============================================================================

class BrokerInterface(ABC):
    """
    Abstract interface for broker connections.
    Implement this for different brokers or connection methods.
    """
    
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to broker."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from broker."""
        pass
    
    @abstractmethod
    async def is_connected(self) -> bool:
        """Check if connected to broker."""
        pass
    
    @abstractmethod
    async def get_account_info(self) -> AccountInfo:
        """Get account information."""
        pass
    
    @abstractmethod
    async def get_symbol_info(self, symbol: str) -> SymbolInfo:
        """Get symbol information."""
        pass
    
    @abstractmethod
    async def get_current_tick(self, symbol: str) -> Tick:
        """Get current price tick."""
        pass
    
    @abstractmethod
    async def get_price_data(
        self,
        symbol: str,
        timeframe: str,
        count: int,
        start_time: Optional[datetime] = None
    ) -> List[PriceData]:
        """Get historical price data."""
        pass
    
    @abstractmethod
    async def get_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """Get open positions."""
        pass
    
    @abstractmethod
    async def open_position(
        self,
        symbol: str,
        direction: OrderDirection,
        volume: float,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        comment: str = "",
        magic: int = 0
    ) -> TradeResult:
        """Open a new position."""
        pass
    
    @abstractmethod
    async def close_position(
        self,
        ticket: int,
        volume: Optional[float] = None
    ) -> TradeResult:
        """Close a position (fully or partially)."""
        pass
    
    @abstractmethod
    async def modify_position(
        self,
        ticket: int,
        sl: Optional[float] = None,
        tp: Optional[float] = None
    ) -> TradeResult:
        """Modify position SL/TP."""
        pass


# =============================================================================
# MT5 Connector Implementation (Windows)
# =============================================================================

class MT5Connector(BrokerInterface):
    """
    MetaTrader 5 connector using the official Python package.
    NOTE: This only works on Windows where MT5 terminal is installed.
    """
    
    TIMEFRAME_MAP = {
        "M1": 1,
        "M5": 5,
        "M15": 15,
        "M30": 30,
        "H1": 60,
        "H4": 240,
        "D1": 1440,
        "W1": 10080,
        "MN1": 43200,
    }
    
    def __init__(
        self,
        login: Optional[int] = None,
        password: Optional[str] = None,
        server: Optional[str] = None,
        path: Optional[str] = None,
        magic_number: int = 123456
    ):
        self.login = login or settings.mt5_login
        self.password = password or settings.mt5_password
        self.server = server or settings.mt5_server
        self.path = path or settings.mt5_path
        self.magic_number = magic_number
        self._connected = False
        self._mt5 = None
    
    def _load_mt5(self):
        """Lazy load MT5 module (only available on Windows)."""
        if self._mt5 is None:
            try:
                import MetaTrader5 as mt5
                self._mt5 = mt5
            except ImportError:
                raise MT5ConnectionError(
                    "MetaTrader5 package not available. "
                    "This only works on Windows with MT5 installed. "
                    "For Linux, use the MT5 API bridge approach."
                )
        return self._mt5
    
    async def connect(self) -> bool:
        """Connect to MT5 terminal."""
        mt5 = self._load_mt5()
        
        logger.info("Connecting to MT5", server=self.server, login=self.login)
        
        # Initialize MT5
        if not mt5.initialize(path=self.path):
            error = mt5.last_error()
            raise MT5ConnectionError(
                f"Failed to initialize MT5: {error}",
                details={"error_code": error[0], "error_message": error[1]}
            )
        
        # Login to account
        if not mt5.login(login=self.login, password=self.password, server=self.server):
            error = mt5.last_error()
            mt5.shutdown()
            raise MT5ConnectionError(
                f"Failed to login to MT5: {error}",
                details={"error_code": error[0], "error_message": error[1]}
            )
        
        self._connected = True
        
        # Enable default trading symbol in Market Watch
        default_symbol = settings.default_symbol
        if default_symbol:
            if not mt5.symbol_select(default_symbol, True):
                logger.warning(
                    "Failed to enable symbol in Market Watch",
                    symbol=default_symbol,
                    error=mt5.last_error()
                )
            else:
                logger.info("Symbol enabled in Market Watch", symbol=default_symbol)
        
        # Log account info
        account = await self.get_account_info()
        logger.info(
            "Connected to MT5",
            account=account.name,
            balance=float(account.balance),
            leverage=account.leverage
        )
        
        return True
    
    async def disconnect(self) -> None:
        """Disconnect from MT5."""
        if self._mt5 and self._connected:
            self._mt5.shutdown()
            self._connected = False
            logger.info("Disconnected from MT5")
    
    async def is_connected(self) -> bool:
        """Check connection status."""
        if not self._connected or not self._mt5:
            return False
        info = self._mt5.terminal_info()
        return info is not None and info.connected
    
    async def get_account_info(self) -> AccountInfo:
        """Get account information."""
        mt5 = self._load_mt5()
        info = mt5.account_info()
        
        if info is None:
            raise MT5ConnectionError("Failed to get account info")
        
        return AccountInfo(
            login=info.login,
            name=info.name,
            server=info.server,
            currency=info.currency,
            balance=Decimal(str(info.balance)),
            equity=Decimal(str(info.equity)),
            margin=Decimal(str(info.margin)),
            free_margin=Decimal(str(info.margin_free)),
            margin_level=info.margin_level if info.margin_level else None,
            profit=Decimal(str(info.profit)),
            leverage=info.leverage
        )
    
    async def get_symbol_info(self, symbol: str) -> SymbolInfo:
        """Get symbol information."""
        mt5 = self._load_mt5()
        info = mt5.symbol_info(symbol)
        
        if info is None:
            raise SymbolNotFoundError(symbol)
        
        # Ensure symbol is visible in Market Watch
        if not info.visible:
            if not mt5.symbol_select(symbol, True):
                raise SymbolNotFoundError(symbol)
            info = mt5.symbol_info(symbol)
        
        return SymbolInfo(
            name=info.name,
            description=info.description,
            digits=info.digits,
            point=info.point,
            spread=info.spread,
            tick_size=info.trade_tick_size,
            tick_value=info.trade_tick_value,
            min_lot=info.volume_min,
            max_lot=info.volume_max,
            lot_step=info.volume_step,
            contract_size=info.trade_contract_size,
            margin_required=info.margin_initial,
            trade_allowed=info.trade_mode != 0
        )
    
    async def get_current_tick(self, symbol: str) -> Tick:
        """Get current price tick."""
        mt5 = self._load_mt5()
        tick = mt5.symbol_info_tick(symbol)
        
        if tick is None:
            raise SymbolNotFoundError(symbol)
        
        return Tick(
            symbol=symbol,
            timestamp=datetime.fromtimestamp(tick.time, tz=timezone.utc),
            bid=tick.bid,
            ask=tick.ask,
            last=tick.last,
            volume=int(tick.volume)
        )
    
    async def get_price_data(
        self,
        symbol: str,
        timeframe: str,
        count: int,
        start_time: Optional[datetime] = None
    ) -> List[PriceData]:
        """Get historical price data."""
        mt5 = self._load_mt5()
        
        tf = self.TIMEFRAME_MAP.get(timeframe.upper())
        if tf is None:
            raise ValueError(f"Invalid timeframe: {timeframe}")
        
        # Get timeframe constant
        tf_const = getattr(mt5, f"TIMEFRAME_M{tf}" if tf < 60 else f"TIMEFRAME_H{tf//60}" if tf < 1440 else f"TIMEFRAME_D1")
        
        if start_time:
            rates = mt5.copy_rates_from(symbol, tf_const, start_time, count)
        else:
            rates = mt5.copy_rates_from_pos(symbol, tf_const, 0, count)
        
        if rates is None or len(rates) == 0:
            return []
        
        return [
            PriceData(
                timestamp=datetime.fromtimestamp(r['time'], tz=timezone.utc),
                open=r['open'],
                high=r['high'],
                low=r['low'],
                close=r['close'],
                tick_volume=r['tick_volume'],
                spread=r['spread'],
                real_volume=r['real_volume']
            )
            for r in rates
        ]
    
    async def get_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """Get open positions."""
        mt5 = self._load_mt5()
        
        if symbol:
            positions = mt5.positions_get(symbol=symbol)
        else:
            positions = mt5.positions_get()
        
        if positions is None:
            return []
        
        return [
            Position(
                ticket=p.ticket,
                symbol=p.symbol,
                direction=OrderDirection.BUY if p.type == 0 else OrderDirection.SELL,
                volume=p.volume,
                price_open=p.price_open,
                price_current=p.price_current,
                sl=p.sl if p.sl > 0 else None,
                tp=p.tp if p.tp > 0 else None,
                profit=p.profit,
                swap=p.swap,
                commission=p.commission if hasattr(p, 'commission') else 0,
                time_open=datetime.fromtimestamp(p.time, tz=timezone.utc),
                magic=p.magic,
                comment=p.comment
            )
            for p in positions
        ]
    
    async def open_position(
        self,
        symbol: str,
        direction: OrderDirection,
        volume: float,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        comment: str = "",
        magic: int = 0
    ) -> TradeResult:
        """Open a new position."""
        mt5 = self._load_mt5()
        
        with TradingContextLogger(symbol=symbol, strategy=comment):
            # Get symbol info for validation
            symbol_info = await self.get_symbol_info(symbol)
            
            # Normalize volume
            volume = symbol_info.normalize_lot(volume)
            
            # Get current tick for price
            tick = await self.get_current_tick(symbol)
            
            # Determine order type and price
            if direction == OrderDirection.BUY:
                order_type = mt5.ORDER_TYPE_BUY
                price = tick.ask
            else:
                order_type = mt5.ORDER_TYPE_SELL
                price = tick.bid
            
            # Normalize SL/TP
            if sl:
                sl = symbol_info.normalize_price(sl)
            if tp:
                tp = symbol_info.normalize_price(tp)
            
            # Build request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": order_type,
                "price": price,
                "deviation": 20,  # Max slippage in points
                "magic": magic or self.magic_number,
                "comment": comment[:31] if comment else "",  # MT5 limit
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            if sl:
                request["sl"] = sl
            if tp:
                request["tp"] = tp
            
            logger.info(
                "Opening position",
                direction=direction.value,
                volume=volume,
                price=price,
                sl=sl,
                tp=tp
            )
            
            # Send order
            result = mt5.order_send(request)
            
            if result is None:
                error = mt5.last_error()
                raise OrderExecutionError(
                    direction.value, symbol,
                    error_code=error[0],
                    broker_message=error[1]
                )
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(
                    "Order rejected",
                    retcode=result.retcode,
                    comment=result.comment
                )
                return TradeResult(
                    success=False,
                    ticket=None,
                    order_id=result.order,
                    volume=0,
                    price=0,
                    comment=result.comment,
                    error_code=result.retcode,
                    error_message=result.comment
                )
            
            logger.info(
                "Position opened",
                ticket=result.order,
                price=result.price,
                volume=result.volume
            )
            
            return TradeResult(
                success=True,
                ticket=result.order,
                order_id=result.order,
                volume=result.volume,
                price=result.price,
                comment=result.comment,
                error_code=None,
                error_message=None
            )
    
    async def close_position(
        self,
        ticket: int,
        volume: Optional[float] = None
    ) -> TradeResult:
        """Close a position (fully or partially)."""
        mt5 = self._load_mt5()
        
        # Get position info
        positions = await self.get_positions()
        position = next((p for p in positions if p.ticket == ticket), None)
        
        if position is None:
            raise PositionNotFoundError(ticket)
        
        with TradingContextLogger(symbol=position.symbol, trade_id=str(ticket)):
            # Get current tick
            tick = await self.get_current_tick(position.symbol)
            
            # Determine close price and order type
            if position.direction == OrderDirection.BUY:
                order_type = mt5.ORDER_TYPE_SELL
                price = tick.bid
            else:
                order_type = mt5.ORDER_TYPE_BUY
                price = tick.ask
            
            # Use full volume if not specified
            close_volume = volume if volume else position.volume
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": position.symbol,
                "volume": close_volume,
                "type": order_type,
                "position": ticket,
                "price": price,
                "deviation": 20,
                "magic": position.magic,
                "comment": "Close",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            logger.info(
                "Closing position",
                ticket=ticket,
                volume=close_volume,
                price=price
            )
            
            result = mt5.order_send(request)
            
            if result is None:
                error = mt5.last_error()
                raise OrderExecutionError(
                    "CLOSE", position.symbol,
                    error_code=error[0],
                    broker_message=error[1]
                )
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                return TradeResult(
                    success=False,
                    ticket=ticket,
                    order_id=result.order,
                    volume=0,
                    price=0,
                    comment=result.comment,
                    error_code=result.retcode,
                    error_message=result.comment
                )
            
            logger.info(
                "Position closed",
                ticket=ticket,
                price=result.price,
                volume=result.volume
            )
            
            return TradeResult(
                success=True,
                ticket=ticket,
                order_id=result.order,
                volume=result.volume,
                price=result.price,
                comment=result.comment,
                error_code=None,
                error_message=None
            )
    
    async def modify_position(
        self,
        ticket: int,
        sl: Optional[float] = None,
        tp: Optional[float] = None
    ) -> TradeResult:
        """Modify position SL/TP."""
        mt5 = self._load_mt5()
        
        # Get position info
        positions = await self.get_positions()
        position = next((p for p in positions if p.ticket == ticket), None)
        
        if position is None:
            raise PositionNotFoundError(ticket)
        
        with TradingContextLogger(symbol=position.symbol, trade_id=str(ticket)):
            # Get symbol info for price normalization
            symbol_info = await self.get_symbol_info(position.symbol)
            
            # Use existing values if not provided
            new_sl = symbol_info.normalize_price(sl) if sl else position.sl
            new_tp = symbol_info.normalize_price(tp) if tp else position.tp
            
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": position.symbol,
                "position": ticket,
                "sl": new_sl or 0,
                "tp": new_tp or 0,
            }
            
            logger.info(
                "Modifying position",
                ticket=ticket,
                sl=new_sl,
                tp=new_tp
            )
            
            result = mt5.order_send(request)
            
            if result is None:
                error = mt5.last_error()
                raise OrderExecutionError(
                    "MODIFY", position.symbol,
                    error_code=error[0],
                    broker_message=error[1]
                )
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                return TradeResult(
                    success=False,
                    ticket=ticket,
                    order_id=None,
                    volume=position.volume,
                    price=position.price_open,
                    comment=result.comment,
                    error_code=result.retcode,
                    error_message=result.comment
                )
            
            logger.info("Position modified", ticket=ticket)
            
            return TradeResult(
                success=True,
                ticket=ticket,
                order_id=None,
                volume=position.volume,
                price=position.price_open,
                comment="Modified",
                error_code=None,
                error_message=None
            )


# =============================================================================
# Demo/Paper Trading Connector
# =============================================================================

class DemoConnector(BrokerInterface):
    """
    Demo connector for testing without real broker connection.
    Simulates trading with realistic behavior.
    Uses historical data from .hcs files if available.
    """

    def __init__(self, initial_balance: float = 10000.0):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.equity = initial_balance
        self.positions: Dict[int, Position] = {}
        self.next_ticket = 1000
        self._connected = False
        self._prices: Dict[str, Tick] = {}
        self._historical_loader = None
    
    async def connect(self) -> bool:
        self._connected = True

        # Initialize historical data loader if available
        if HISTORICAL_DATA_AVAILABLE and HistoricalDataLoader:
            try:
                self._historical_loader = HistoricalDataLoader(data_path="data/history")
                logger.info("Demo connector connected with historical data", balance=self.balance)
            except Exception as e:
                logger.warning("Failed to initialize historical data loader", error=str(e))
                logger.info("Demo connector connected (without historical data)", balance=self.balance)
        else:
            logger.info("Demo connector connected (without historical data)", balance=self.balance)

        return True
    
    async def disconnect(self) -> None:
        self._connected = False
        logger.info("Demo connector disconnected")
    
    async def is_connected(self) -> bool:
        return self._connected
    
    async def get_account_info(self) -> AccountInfo:
        return AccountInfo(
            login=12345,
            name="Demo Account",
            server="Demo",
            currency="USD",
            balance=Decimal(str(self.balance)),
            equity=Decimal(str(self.equity)),
            margin=Decimal("0"),
            free_margin=Decimal(str(self.equity)),
            margin_level=None,
            profit=Decimal(str(self.equity - self.balance)),
            leverage=100
        )
    
    async def get_symbol_info(self, symbol: str) -> SymbolInfo:
        # Default XAUUSD info
        return SymbolInfo(
            name=symbol,
            description=f"{symbol} Demo",
            digits=2 if "XAU" in symbol else 5,
            point=0.01 if "XAU" in symbol else 0.00001,
            spread=20,
            tick_size=0.01,
            tick_value=1.0,
            min_lot=0.01,
            max_lot=100.0,
            lot_step=0.01,
            contract_size=100.0,
            margin_required=1000.0,
            trade_allowed=True
        )
    
    async def get_current_tick(self, symbol: str) -> Tick:
        # Return cached or generate demo tick
        if symbol not in self._prices:
            base_price = 2000.0 if "XAU" in symbol else 1.1000
            self._prices[symbol] = Tick(
                symbol=symbol,
                timestamp=datetime.now(timezone.utc),
                bid=base_price,
                ask=base_price + 0.20,
                last=base_price,
                volume=1000
            )
        return self._prices[symbol]
    
    async def get_price_data(
        self,
        symbol: str,
        timeframe: str,
        count: int,
        start_time: Optional[datetime] = None
    ) -> List[PriceData]:
        # Try to load from historical data
        if self._historical_loader:
            try:
                df = self._historical_loader.get_timeframe_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    count=count,
                    end_time=start_time
                )

                # Convert DataFrame to List[PriceData]
                price_data_list = []
                for timestamp, row in df.iterrows():
                    vol = int(row.get('volume', row.get('tick_volume', 0)))
                    price_data_list.append(PriceData(
                        timestamp=timestamp,
                        open=row['open'],
                        high=row['high'],
                        low=row['low'],
                        close=row['close'],
                        tick_volume=vol,
                        spread=int(row.get('spread', 20)),
                        real_volume=int(row.get('real_volume', 0))
                    ))

                logger.debug(
                    "Loaded historical data",
                    symbol=symbol,
                    timeframe=timeframe,
                    bars=len(price_data_list)
                )

                return price_data_list

            except Exception as e:
                logger.error(
                    "Failed to load historical data",
                    symbol=symbol,
                    timeframe=timeframe,
                    error=str(e)
                )
                return []

        # No historical data available
        return []
    
    async def get_positions(self, symbol: Optional[str] = None) -> List[Position]:
        positions = list(self.positions.values())
        if symbol:
            positions = [p for p in positions if p.symbol == symbol]
        return positions
    
    async def open_position(
        self,
        symbol: str,
        direction: OrderDirection,
        volume: float,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        comment: str = "",
        magic: int = 0
    ) -> TradeResult:
        tick = await self.get_current_tick(symbol)
        price = tick.ask if direction == OrderDirection.BUY else tick.bid
        
        ticket = self.next_ticket
        self.next_ticket += 1
        
        self.positions[ticket] = Position(
            ticket=ticket,
            symbol=symbol,
            direction=direction,
            volume=volume,
            price_open=price,
            price_current=price,
            sl=sl,
            tp=tp,
            profit=0,
            swap=0,
            commission=0,
            time_open=datetime.now(timezone.utc),
            magic=magic,
            comment=comment
        )
        
        logger.info(
            "Demo position opened",
            ticket=ticket,
            direction=direction.value,
            volume=volume,
            price=price
        )
        
        return TradeResult(
            success=True,
            ticket=ticket,
            order_id=ticket,
            volume=volume,
            price=price,
            comment="Demo order filled",
            error_code=None,
            error_message=None
        )
    
    async def close_position(
        self,
        ticket: int,
        volume: Optional[float] = None
    ) -> TradeResult:
        if ticket not in self.positions:
            raise PositionNotFoundError(ticket)
        
        position = self.positions[ticket]
        tick = await self.get_current_tick(position.symbol)
        
        if position.direction == OrderDirection.BUY:
            close_price = tick.bid
        else:
            close_price = tick.ask
        
        # Calculate P/L
        if position.direction == OrderDirection.BUY:
            pips = close_price - position.price_open
        else:
            pips = position.price_open - close_price
        
        profit = pips * position.volume * 100  # Simplified
        self.balance += profit
        self.equity = self.balance
        
        del self.positions[ticket]
        
        logger.info(
            "Demo position closed",
            ticket=ticket,
            price=close_price,
            profit=profit
        )
        
        return TradeResult(
            success=True,
            ticket=ticket,
            order_id=ticket,
            volume=position.volume,
            price=close_price,
            comment=f"Closed with P/L: {profit:.2f}",
            error_code=None,
            error_message=None
        )
    
    async def modify_position(
        self,
        ticket: int,
        sl: Optional[float] = None,
        tp: Optional[float] = None
    ) -> TradeResult:
        if ticket not in self.positions:
            raise PositionNotFoundError(ticket)
        
        position = self.positions[ticket]
        if sl:
            position.sl = sl
        if tp:
            position.tp = tp
        
        logger.info("Demo position modified", ticket=ticket, sl=sl, tp=tp)
        
        return TradeResult(
            success=True,
            ticket=ticket,
            order_id=None,
            volume=position.volume,
            price=position.price_open,
            comment="Modified",
            error_code=None,
            error_message=None
        )


# =============================================================================
# Factory Function
# =============================================================================

def get_broker_connector(demo: bool = False) -> BrokerInterface:
    """
    Factory function to get appropriate broker connector.
    
    Args:
        demo: If True, return demo connector for testing
    
    Returns:
        BrokerInterface implementation
    """
    # Use MT5 if explicitly enabled or if MT5 credentials are configured
    use_real_mt5 = settings.use_mt5 or (settings.mt5_login > 0 and settings.mt5_password)
    
    if use_real_mt5 and not demo:
        logger.info("Using MT5 connector", login=settings.mt5_login, server=settings.mt5_server)
        return MT5Connector()
    
    if demo or settings.app_env.value == "development":
        logger.info("Using demo connector")
        return DemoConnector(initial_balance=10000.0)
    
    logger.info("Using MT5 connector")
    return MT5Connector()
