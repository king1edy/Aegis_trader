"""
Paper Trading Broker
====================
Simulated broker for testing and development without real money.
Implements the same interface as MT5Connector for seamless integration.
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Optional, Dict, Any, List
import structlog
import random
import math

from src.core.config import Settings

logger = structlog.get_logger()


@dataclass
class PriceData:
    """Historical price data (OHLCV bar)."""
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    tick_volume: int
    spread: int = 0
    real_volume: int = 0


class PaperTradingBroker:
    """
    Paper trading broker for simulation.
    
    Simulates trading operations without connecting to a real broker.
    Useful for testing strategies and development.
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize paper trading broker.
        
        Args:
            settings: Application settings
        """
        self.settings = settings
        self._connected = False
        self._positions: Dict[int, Dict[str, Any]] = {}
        self._next_ticket = 1000000
        self._balance = Decimal("10000.00")  # Starting balance
        self._equity = self._balance
        
        # Simulated market prices (will be updated)
        self._prices: Dict[str, Dict[str, Decimal]] = {
            "XAUUSDm": {"bid": Decimal("2050.50"), "ask": Decimal("2050.70")},
            "XAUUSD": {"bid": Decimal("2050.50"), "ask": Decimal("2050.70")},
        }
        
        self.logger = logger.bind(component="paper_broker")
    
    @property
    def is_connected(self) -> bool:
        """Check if broker is connected (always True for paper trading)."""
        return self._connected
    
    async def connect(self) -> bool:
        """
        Simulate connection to broker.
        
        Returns:
            True (always succeeds for paper trading)
        """
        self._connected = True
        self.logger.info(
            "Paper trading broker connected",
            balance=float(self._balance),
            mode="SIMULATION"
        )
        return True
    
    async def disconnect(self):
        """Disconnect paper broker."""
        self._connected = False
        self.logger.info("Paper trading broker disconnected")
    
    async def get_account_info(self) -> Dict[str, Any]:
        """
        Get simulated account information.
        
        Returns:
            Account info dictionary
        """
        # Update equity based on open positions
        self._update_equity()
        
        return {
            "login": 0,
            "server": "PaperTrading",
            "balance": self._balance,
            "equity": self._equity,
            "margin": Decimal("0"),
            "free_margin": self._equity,
            "leverage": 100,
            "currency": "USD",
            "profit": self._equity - self._balance,
        }
    
    async def get_tick(self, symbol: str) -> Dict[str, Any]:
        """
        Get simulated tick for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Tick data dictionary
        """
        # Get or create prices for symbol
        if symbol not in self._prices:
            # Default price for unknown symbols
            self._prices[symbol] = {
                "bid": Decimal("100.00"),
                "ask": Decimal("100.02")
            }
        
        prices = self._prices[symbol]
        
        # Add small random variation
        import random
        variation = Decimal(str(random.uniform(-0.1, 0.1)))
        bid = prices["bid"] + variation
        ask = bid + Decimal("0.20")  # Fixed spread
        
        # Update stored prices
        self._prices[symbol]["bid"] = bid
        self._prices[symbol]["ask"] = ask
        
        return {
            "symbol": symbol,
            "bid": bid,
            "ask": ask,
            "time": datetime.utcnow(),
            "spread": ask - bid,
        }
    
    async def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get simulated symbol information.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Symbol info dictionary
        """
        # Default symbol info for gold
        tick = await self.get_tick(symbol)
        
        return {
            "name": symbol,
            "digits": 2,
            "point": Decimal("0.01"),
            "trade_contract_size": Decimal("100"),
            "volume_min": Decimal("0.01"),
            "volume_max": Decimal("100.0"),
            "volume_step": Decimal("0.01"),
            "spread": 20,
            "bid": tick["bid"],
            "ask": tick["ask"],
        }
    
    async def get_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get simulated open positions.
        
        Args:
            symbol: Filter by symbol (optional)
            
        Returns:
            List of position dictionaries
        """
        positions = []
        
        for ticket, pos in self._positions.items():
            if symbol and pos["symbol"] != symbol:
                continue
            
            # Update current price and P&L
            tick = await self.get_tick(pos["symbol"])
            if pos["type"] == "buy":
                current_price = tick["bid"]
                profit = (current_price - pos["price_open"]) * pos["volume"] * Decimal("100")
            else:
                current_price = tick["ask"]
                profit = (pos["price_open"] - current_price) * pos["volume"] * Decimal("100")
            
            positions.append({
                "ticket": ticket,
                "symbol": pos["symbol"],
                "type": pos["type"],
                "volume": pos["volume"],
                "price_open": pos["price_open"],
                "price_current": current_price,
                "sl": pos["sl"],
                "tp": pos["tp"],
                "profit": profit,
                "swap": Decimal("0"),
                "time": pos["time"],
                "magic": pos["magic"],
                "comment": pos["comment"],
            })
        
        return positions
    
    async def get_position(self, ticket: int) -> Optional[Dict[str, Any]]:
        """
        Get a specific position by ticket.
        
        Args:
            ticket: Position ticket number
            
        Returns:
            Position dictionary or None
        """
        if ticket not in self._positions:
            return None
        
        positions = await self.get_positions()
        for pos in positions:
            if pos["ticket"] == ticket:
                return pos
        return None
    
    async def place_order(
        self,
        symbol: str,
        order_type: str,
        volume: float,
        price: Optional[float] = None,
        sl: float = 0.0,
        tp: float = 0.0,
        magic: int = 0,
        comment: str = "",
        deviation: int = 20
    ) -> Dict[str, Any]:
        """
        Simulate placing a market order.
        
        Args:
            symbol: Trading symbol
            order_type: "buy" or "sell"
            volume: Lot size
            price: Price (ignored, uses market)
            sl: Stop loss price
            tp: Take profit price
            magic: Magic number
            comment: Order comment
            deviation: Slippage (ignored)
            
        Returns:
            Order result dictionary
        """
        tick = await self.get_tick(symbol)
        
        if order_type.lower() == "buy":
            fill_price = tick["ask"]
        else:
            fill_price = tick["bid"]
        
        ticket = self._next_ticket
        self._next_ticket += 1
        
        self._positions[ticket] = {
            "symbol": symbol,
            "type": order_type.lower(),
            "volume": Decimal(str(volume)),
            "price_open": fill_price,
            "sl": Decimal(str(sl)),
            "tp": Decimal(str(tp)),
            "magic": magic,
            "comment": f"[PAPER] {comment}",
            "time": datetime.utcnow(),
        }
        
        self.logger.info(
            "Paper order placed",
            ticket=ticket,
            symbol=symbol,
            type=order_type,
            volume=volume,
            price=float(fill_price)
        )
        
        return {
            "success": True,
            "ticket": ticket,
            "price": fill_price,
            "volume": Decimal(str(volume)),
        }
    
    async def close_position(
        self,
        ticket: int,
        volume: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Simulate closing a position.
        
        Args:
            ticket: Position ticket
            volume: Volume to close (None = full)
            
        Returns:
            Close result dictionary
        """
        if ticket not in self._positions:
            return {"success": False, "error": f"Position {ticket} not found"}
        
        pos = self._positions[ticket]
        tick = await self.get_tick(pos["symbol"])
        
        if pos["type"] == "buy":
            close_price = tick["bid"]
            profit = (close_price - pos["price_open"]) * pos["volume"] * Decimal("100")
        else:
            close_price = tick["ask"]
            profit = (pos["price_open"] - close_price) * pos["volume"] * Decimal("100")
        
        # Update balance
        self._balance += profit
        
        # Remove position
        del self._positions[ticket]
        
        self.logger.info(
            "Paper position closed",
            ticket=ticket,
            price=float(close_price),
            profit=float(profit)
        )
        
        return {
            "success": True,
            "ticket": ticket,
            "price": close_price,
            "volume": pos["volume"],
            "profit": profit,
        }
    
    async def modify_position(
        self,
        ticket: int,
        sl: Optional[float] = None,
        tp: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Modify position SL/TP.
        
        Args:
            ticket: Position ticket
            sl: New stop loss
            tp: New take profit
            
        Returns:
            Modify result dictionary
        """
        if ticket not in self._positions:
            return {"success": False, "error": f"Position {ticket} not found"}
        
        if sl is not None:
            self._positions[ticket]["sl"] = Decimal(str(sl))
        if tp is not None:
            self._positions[ticket]["tp"] = Decimal(str(tp))
        
        self.logger.info("Paper position modified", ticket=ticket, sl=sl, tp=tp)
        
        return {"success": True, "ticket": ticket}
    
    async def open_position(
        self,
        symbol: str,
        direction: str,
        volume: float,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        comment: str = ""
    ) -> Dict[str, Any]:
        """
        Open a position (alias for place_order).
        
        Args:
            symbol: Trading symbol
            direction: "long" or "short"
            volume: Lot size
            sl: Stop loss price
            tp: Take profit price
            comment: Order comment
            
        Returns:
            Order result dictionary
        """
        order_type = "buy" if direction.lower() == "long" else "sell"
        return await self.place_order(
            symbol=symbol,
            order_type=order_type,
            volume=volume,
            sl=sl or 0.0,
            tp=tp or 0.0,
            magic=self.settings.magic_number,
            comment=comment
        )
    
    def _update_equity(self):
        """Update equity based on open positions."""
        # This is a simplified calculation
        unrealized_pnl = Decimal("0")
        for pos in self._positions.values():
            prices = self._prices.get(pos["symbol"], {"bid": pos["price_open"], "ask": pos["price_open"]})
            if pos["type"] == "buy":
                pnl = (prices["bid"] - pos["price_open"]) * pos["volume"] * Decimal("100")
            else:
                pnl = (pos["price_open"] - prices["ask"]) * pos["volume"] * Decimal("100")
            unrealized_pnl += pnl
        
        self._equity = self._balance + unrealized_pnl
    
    def update_price(self, symbol: str, bid: float, ask: float):
        """
        Manually update price for a symbol (for testing).
        
        Args:
            symbol: Trading symbol
            bid: New bid price
            ask: New ask price
        """
        self._prices[symbol] = {
            "bid": Decimal(str(bid)),
            "ask": Decimal(str(ask))
        }

    async def get_price_data(
        self,
        symbol: str,
        timeframe: str,
        count: int,
        start_time: Optional[datetime] = None
    ) -> List[PriceData]:
        """
        Generate simulated historical price data for paper trading.
        
        Uses a simple random walk algorithm to generate realistic-looking
        price data based on the current simulated price.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe (M1, M5, M15, M30, H1, H4, D1, W1, MN1)
            count: Number of bars to retrieve
            start_time: Start time (optional, ignored for simulation)
            
        Returns:
            List of PriceData objects with simulated OHLCV
        """
        # Timeframe to minutes mapping
        tf_minutes = {
            "M1": 1, "M5": 5, "M15": 15, "M30": 30,
            "H1": 60, "H4": 240, "D1": 1440, "W1": 10080, "MN1": 43200
        }
        
        minutes = tf_minutes.get(timeframe, 15)  # Default to M15
        
        # Get base price for symbol
        prices = self._prices.get(symbol, self._prices.get("XAUUSD", {"bid": Decimal("2050.50"), "ask": Decimal("2050.70")}))
        current_price = float(prices["bid"])
        
        # Volatility based on symbol and timeframe
        if "XAU" in symbol or "GOLD" in symbol:
            base_volatility = 0.0005  # 0.05% per bar for gold
        else:
            base_volatility = 0.0003  # 0.03% for forex pairs
        
        # Scale volatility with timeframe
        volatility = base_volatility * math.sqrt(minutes / 15)
        
        # Generate bars backwards from now
        result = []
        now = datetime.now(timezone.utc)
        price = current_price
        
        # Random seed for reproducible results (based on symbol and timeframe)
        random.seed(hash(f"{symbol}_{timeframe}_{now.date()}"))
        
        for i in range(count - 1, -1, -1):
            bar_time = now - timedelta(minutes=minutes * i)
            
            # Generate OHLC using random walk
            change_pct = random.gauss(0, volatility)
            bar_open = price
            bar_close = price * (1 + change_pct)
            
            # High/Low with some randomness
            intra_volatility = volatility * 0.5
            bar_high = max(bar_open, bar_close) * (1 + abs(random.gauss(0, intra_volatility)))
            bar_low = min(bar_open, bar_close) * (1 - abs(random.gauss(0, intra_volatility)))
            
            # Volume proportional to timeframe
            base_volume = 100 * minutes
            tick_volume = int(base_volume * (1 + random.uniform(-0.3, 0.3)))
            
            result.append(PriceData(
                timestamp=bar_time,
                open=Decimal(str(round(bar_open, 2))),
                high=Decimal(str(round(bar_high, 2))),
                low=Decimal(str(round(bar_low, 2))),
                close=Decimal(str(round(bar_close, 2))),
                tick_volume=tick_volume,
                spread=20,  # Typical spread for gold
                real_volume=0
            ))
            
            # Move to next price
            price = bar_close
        
        self.logger.debug(
            "Generated simulated price data",
            symbol=symbol,
            timeframe=timeframe,
            count=len(result)
        )
        
        return result
