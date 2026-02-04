"""
MT5 API Client
==============
Client for connecting to the MT5 API Bridge from Docker or remote systems.
Implements the same interface as MT5Connector for seamless integration.
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Optional, Dict, Any, List
import httpx
import structlog

from src.core.config import Settings
from src.core.exceptions import (
    MT5ConnectionError,
    OrderExecutionError,
    PositionNotFoundError,
)

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


class MT5APIClient:
    """
    Client for MT5 API Bridge.
    
    Implements the same interface as MT5Connector so it can be used
    as a drop-in replacement when running in Docker or on non-Windows systems.
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize MT5 API client.
        
        Args:
            settings: Application settings
        """
        self.settings = settings
        self.base_url = settings.mt5_bridge_url
        self.api_key = settings.mt5_bridge_api_key
        self._client: Optional[httpx.AsyncClient] = None
        self._connected = False
        
        self.logger = logger.bind(
            component="mt5_api_client",
            bridge_url=self.base_url
        )
    
    @property
    def is_connected(self) -> bool:
        """Check if connected to the bridge."""
        return self._connected
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with API key if configured."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        return headers
    
    async def connect(self) -> bool:
        """
        Connect to the MT5 API bridge.
        
        Returns:
            True if connection successful
            
        Raises:
            MT5ConnectionError: If connection fails
        """
        self.logger.info("Connecting to MT5 API bridge...")
        
        try:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=self._get_headers(),
                timeout=30.0
            )
            
            # Check health
            response = await self._client.get("/health")
            response.raise_for_status()
            
            health = response.json()
            
            if not health.get("mt5_connected"):
                raise MT5ConnectionError("MT5 bridge is running but MT5 is not connected")
            
            self._connected = True
            self.logger.info(
                "Connected to MT5 API bridge",
                account_login=health.get("account_login"),
                server=health.get("server")
            )
            
            return True
            
        except httpx.RequestError as e:
            self._connected = False
            raise MT5ConnectionError(f"Failed to connect to MT5 bridge at {self.base_url}: {e}")
        except httpx.HTTPStatusError as e:
            self._connected = False
            raise MT5ConnectionError(f"MT5 bridge returned error: {e.response.status_code}")
    
    async def disconnect(self):
        """Disconnect from the bridge."""
        if self._client:
            await self._client.aclose()
            self._client = None
        self._connected = False
        self.logger.info("Disconnected from MT5 API bridge")
    
    async def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make a request to the bridge API."""
        if not self._client:
            raise MT5ConnectionError("Not connected to MT5 bridge")
        
        try:
            response = await self._client.request(method, endpoint, **kwargs)
            response.raise_for_status()
            return response.json()
        except httpx.RequestError as e:
            self.logger.error("Bridge request failed", endpoint=endpoint, error=str(e))
            raise MT5ConnectionError(f"Bridge request failed: {e}")
        except httpx.HTTPStatusError as e:
            error_detail = e.response.json().get("detail", str(e))
            self.logger.error("Bridge returned error", endpoint=endpoint, error=error_detail)
            raise MT5ConnectionError(f"Bridge error: {error_detail}")
    
    async def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information.
        
        Returns:
            Account info dictionary
        """
        data = await self._request("GET", "/account")
        return {
            "login": data["login"],
            "server": data["server"],
            "balance": Decimal(str(data["balance"])),
            "equity": Decimal(str(data["equity"])),
            "margin": Decimal(str(data["margin"])),
            "free_margin": Decimal(str(data["free_margin"])),
            "leverage": data["leverage"],
            "currency": data["currency"],
            "profit": Decimal(str(data["profit"])),
        }
    
    async def get_tick(self, symbol: str) -> Dict[str, Any]:
        """
        Get current tick for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Tick data dictionary with bid, ask, time
        """
        data = await self._request("GET", f"/tick/{symbol}")
        return {
            "symbol": data["symbol"],
            "bid": Decimal(str(data["bid"])),
            "ask": Decimal(str(data["ask"])),
            "time": datetime.fromisoformat(data["time"]),
            "spread": Decimal(str(data["ask"])) - Decimal(str(data["bid"])),
        }
    
    async def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get symbol information.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Symbol info dictionary
        """
        data = await self._request("GET", f"/symbol/{symbol}")
        return {
            "name": data["name"],
            "digits": data["digits"],
            "point": Decimal(str(data["point"])),
            "trade_contract_size": Decimal(str(data["trade_contract_size"])),
            "volume_min": Decimal(str(data["volume_min"])),
            "volume_max": Decimal(str(data["volume_max"])),
            "volume_step": Decimal(str(data["volume_step"])),
            "spread": data["spread"],
            "bid": Decimal(str(data["bid"])),
            "ask": Decimal(str(data["ask"])),
        }
    
    async def get_price_data(
        self,
        symbol: str,
        timeframe: str,
        count: int,
        start_time: Optional[datetime] = None
    ) -> List[PriceData]:
        """
        Get historical price data.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe (M1, M5, M15, M30, H1, H4, D1, W1, MN1)
            count: Number of bars to retrieve
            start_time: Start time (optional)
            
        Returns:
            List of PriceData objects with OHLCV
        """
        params = {
            "timeframe": timeframe,
            "count": count
        }
        if start_time:
            params["start_time"] = start_time.isoformat()
        
        data = await self._request("GET", f"/price_data/{symbol}", params=params)
        
        result = []
        for bar in data:
            result.append(PriceData(
                timestamp=datetime.fromisoformat(bar["timestamp"]),
                open=Decimal(str(bar["open"])),
                high=Decimal(str(bar["high"])),
                low=Decimal(str(bar["low"])),
                close=Decimal(str(bar["close"])),
                tick_volume=bar["tick_volume"],
                spread=bar.get("spread", 0),
                real_volume=bar.get("real_volume", 0),
            ))
        
        return result
    
    async def get_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get open positions.
        
        Args:
            symbol: Filter by symbol (optional)
            
        Returns:
            List of position dictionaries
        """
        params = {}
        if symbol:
            params["symbol"] = symbol
        
        data = await self._request("GET", "/positions", params=params)
        
        positions = []
        for pos in data:
            positions.append({
                "ticket": pos["ticket"],
                "symbol": pos["symbol"],
                "type": pos["type"],
                "volume": Decimal(str(pos["volume"])),
                "price_open": Decimal(str(pos["price_open"])),
                "price_current": Decimal(str(pos["price_current"])),
                "sl": Decimal(str(pos["sl"])),
                "tp": Decimal(str(pos["tp"])),
                "profit": Decimal(str(pos["profit"])),
                "swap": Decimal(str(pos["swap"])),
                "time": datetime.fromisoformat(pos["time"]),
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
            Position dictionary or None if not found
        """
        try:
            data = await self._request("GET", f"/positions/{ticket}")
            return {
                "ticket": data["ticket"],
                "symbol": data["symbol"],
                "type": data["type"],
                "volume": Decimal(str(data["volume"])),
                "price_open": Decimal(str(data["price_open"])),
                "price_current": Decimal(str(data["price_current"])),
                "sl": Decimal(str(data["sl"])),
                "tp": Decimal(str(data["tp"])),
                "profit": Decimal(str(data["profit"])),
                "swap": Decimal(str(data["swap"])),
                "time": datetime.fromisoformat(data["time"]),
                "magic": data["magic"],
                "comment": data["comment"],
            }
        except MT5ConnectionError:
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
        Place a market order.
        
        Args:
            symbol: Trading symbol
            order_type: "buy" or "sell"
            volume: Lot size
            price: Price (optional, uses market price)
            sl: Stop loss price
            tp: Take profit price
            magic: Magic number
            comment: Order comment
            deviation: Allowed slippage
            
        Returns:
            Order result dictionary
            
        Raises:
            OrderExecutionError: If order fails
        """
        request_data = {
            "symbol": symbol,
            "order_type": order_type.lower(),
            "volume": volume,
            "sl": sl,
            "tp": tp,
            "magic": magic,
            "comment": comment,
            "deviation": deviation
        }
        
        if price:
            request_data["price"] = price
        
        data = await self._request("POST", "/order", json=request_data)
        
        if not data.get("success"):
            raise OrderExecutionError(
                f"Order failed: {data.get('error', 'Unknown error')}",
                symbol=symbol,
                order_type=order_type
            )
        
        self.logger.info(
            "Order placed via bridge",
            ticket=data.get("ticket"),
            symbol=symbol,
            type=order_type,
            volume=volume,
            price=data.get("price")
        )
        
        return {
            "success": True,
            "ticket": data.get("ticket"),
            "price": Decimal(str(data["price"])) if data.get("price") else None,
            "volume": Decimal(str(data["volume"])) if data.get("volume") else None,
        }
    
    async def close_position(
        self,
        ticket: int,
        volume: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Close a position.
        
        Args:
            ticket: Position ticket
            volume: Volume to close (None = full position)
            
        Returns:
            Close result dictionary
            
        Raises:
            PositionNotFoundError: If position not found
            OrderExecutionError: If close fails
        """
        params = {}
        if volume:
            params["volume"] = volume
        
        try:
            data = await self._request("POST", f"/positions/{ticket}/close", params=params)
        except MT5ConnectionError as e:
            if "not found" in str(e).lower():
                raise PositionNotFoundError(f"Position {ticket} not found")
            raise
        
        if not data.get("success"):
            raise OrderExecutionError(
                f"Close failed: {data.get('error', 'Unknown error')}",
                ticket=ticket
            )
        
        self.logger.info(
            "Position closed via bridge",
            ticket=ticket,
            price=data.get("price"),
            volume=data.get("volume")
        )
        
        return {
            "success": True,
            "ticket": data.get("ticket"),
            "price": Decimal(str(data["price"])) if data.get("price") else None,
            "volume": Decimal(str(data["volume"])) if data.get("volume") else None,
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
            sl: New stop loss (None = keep current)
            tp: New take profit (None = keep current)
            
        Returns:
            Modify result dictionary
            
        Raises:
            PositionNotFoundError: If position not found
            OrderExecutionError: If modify fails
        """
        params = {}
        if sl is not None:
            params["sl"] = sl
        if tp is not None:
            params["tp"] = tp
        
        try:
            data = await self._request("PUT", f"/positions/{ticket}", params=params)
        except MT5ConnectionError as e:
            if "not found" in str(e).lower():
                raise PositionNotFoundError(f"Position {ticket} not found")
            raise
        
        if not data.get("success"):
            raise OrderExecutionError(
                f"Modify failed: {data.get('error', 'Unknown error')}",
                ticket=ticket
            )
        
        self.logger.info(
            "Position modified via bridge",
            ticket=ticket,
            sl=sl,
            tp=tp
        )
        
        return {"success": True, "ticket": ticket}
    
    # Alias methods to match MT5Connector interface
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
