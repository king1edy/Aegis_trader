"""
MT5 API Bridge Server
=====================
A FastAPI server that exposes MetaTrader 5 operations via REST API.
Run this on Windows where MT5 is installed, then connect from Docker.

Usage:
    python -m src.execution.mt5_api_bridge
    
Or:
    uvicorn src.execution.mt5_api_bridge:app --host 0.0.0.0 --port 8001
"""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import structlog

# Will be imported conditionally
mt5 = None
logger = structlog.get_logger()


# =============================================================================
# Pydantic Models for API
# =============================================================================

class OrderType(str, Enum):
    BUY = "buy"
    SELL = "sell"


class TickData(BaseModel):
    """Real-time tick data."""
    symbol: str
    bid: float
    ask: float
    time: datetime
    volume: float = 0
    last: float = 0


class AccountInfo(BaseModel):
    """Trading account information."""
    login: int
    server: str
    balance: float
    equity: float
    margin: float
    free_margin: float
    leverage: int
    currency: str
    profit: float = 0


class PositionInfo(BaseModel):
    """Open position information."""
    ticket: int
    symbol: str
    type: str
    volume: float
    price_open: float
    price_current: float
    sl: float
    tp: float
    profit: float
    swap: float
    time: datetime
    magic: int = 0
    comment: str = ""


class OrderRequest(BaseModel):
    """Order placement request."""
    symbol: str
    order_type: OrderType
    volume: float
    price: Optional[float] = None
    sl: float = 0.0
    tp: float = 0.0
    magic: int = 0
    comment: str = ""
    deviation: int = 20


class OrderResponse(BaseModel):
    """Order execution response."""
    success: bool
    ticket: Optional[int] = None
    price: Optional[float] = None
    volume: Optional[float] = None
    error: Optional[str] = None


class ClosePositionRequest(BaseModel):
    """Close position request."""
    ticket: int
    volume: Optional[float] = None  # None = close full position


class ModifyPositionRequest(BaseModel):
    """Modify position SL/TP request."""
    ticket: int
    sl: Optional[float] = None
    tp: Optional[float] = None


class SymbolInfo(BaseModel):
    """Symbol information."""
    name: str
    digits: int
    point: float
    trade_contract_size: float
    volume_min: float
    volume_max: float
    volume_step: float
    spread: int
    bid: float
    ask: float


class PriceDataItem(BaseModel):
    """Historical price data (OHLCV)."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    tick_volume: int
    spread: int = 0
    real_volume: int = 0


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    mt5_connected: bool
    account_login: Optional[int] = None
    server: Optional[str] = None
    timestamp: datetime


# =============================================================================
# MT5 Bridge Server
# =============================================================================

class MT5Bridge:
    """Bridge between REST API and MetaTrader 5."""
    
    def __init__(self):
        self.mt5 = None
        self.connected = False
        self.config = None
        
    async def initialize(self, login: int, password: str, server: str, path: str = None):
        """Initialize MT5 connection."""
        try:
            import MetaTrader5 as mt5_module
            self.mt5 = mt5_module
        except ImportError:
            raise RuntimeError("MetaTrader5 package not available. Run on Windows with MT5 installed.")
        
        # Initialize MT5
        init_kwargs = {}
        if path:
            init_kwargs["path"] = path
            
        if not self.mt5.initialize(**init_kwargs):
            error = self.mt5.last_error()
            raise RuntimeError(f"MT5 initialization failed: {error}")
        
        # Login
        if not self.mt5.login(login=login, password=password, server=server):
            error = self.mt5.last_error()
            self.mt5.shutdown()
            raise RuntimeError(f"MT5 login failed: {error}")
        
        self.connected = True
        logger.info("MT5 bridge connected", login=login, server=server)
        
    async def shutdown(self):
        """Shutdown MT5 connection."""
        if self.mt5 and self.connected:
            self.mt5.shutdown()
            self.connected = False
            logger.info("MT5 bridge disconnected")
    
    def ensure_connected(self):
        """Ensure MT5 is connected."""
        if not self.connected or not self.mt5:
            raise HTTPException(status_code=503, detail="MT5 not connected")
    
    def get_account_info(self) -> AccountInfo:
        """Get account information."""
        self.ensure_connected()
        info = self.mt5.account_info()
        if info is None:
            raise HTTPException(status_code=500, detail="Failed to get account info")
        
        return AccountInfo(
            login=info.login,
            server=info.server,
            balance=info.balance,
            equity=info.equity,
            margin=info.margin,
            free_margin=info.margin_free,
            leverage=info.leverage,
            currency=info.currency,
            profit=info.profit
        )
    
    def get_tick(self, symbol: str) -> TickData:
        """Get current tick for symbol."""
        self.ensure_connected()
        
        # Ensure symbol is in Market Watch
        if not self.mt5.symbol_select(symbol, True):
            raise HTTPException(status_code=400, detail=f"Symbol {symbol} not available")
        
        tick = self.mt5.symbol_info_tick(symbol)
        if tick is None:
            raise HTTPException(status_code=500, detail=f"Failed to get tick for {symbol}")
        
        return TickData(
            symbol=symbol,
            bid=tick.bid,
            ask=tick.ask,
            time=datetime.fromtimestamp(tick.time),
            volume=tick.volume,
            last=tick.last
        )
    
    def get_symbol_info(self, symbol: str) -> SymbolInfo:
        """Get symbol information."""
        self.ensure_connected()
        
        if not self.mt5.symbol_select(symbol, True):
            raise HTTPException(status_code=400, detail=f"Symbol {symbol} not available")
        
        info = self.mt5.symbol_info(symbol)
        if info is None:
            raise HTTPException(status_code=500, detail=f"Failed to get info for {symbol}")
        
        return SymbolInfo(
            name=info.name,
            digits=info.digits,
            point=info.point,
            trade_contract_size=info.trade_contract_size,
            volume_min=info.volume_min,
            volume_max=info.volume_max,
            volume_step=info.volume_step,
            spread=info.spread,
            bid=info.bid,
            ask=info.ask
        )
    
    # Timeframe mapping
    TIMEFRAME_MAP = {
        "M1": 1, "M5": 5, "M15": 15, "M30": 30,
        "H1": 60, "H4": 240,
        "D1": 1440, "W1": 10080, "MN1": 43200
    }
    
    def get_price_data(
        self, 
        symbol: str, 
        timeframe: str, 
        count: int,
        start_time: Optional[datetime] = None
    ) -> List[PriceDataItem]:
        """Get historical price data."""
        self.ensure_connected()
        
        # Ensure symbol is in Market Watch
        if not self.mt5.symbol_select(symbol, True):
            raise HTTPException(status_code=400, detail=f"Symbol {symbol} not available")
        
        tf = self.TIMEFRAME_MAP.get(timeframe.upper())
        if tf is None:
            raise HTTPException(status_code=400, detail=f"Invalid timeframe: {timeframe}")
        
        # Get the MT5 timeframe constant
        if tf == 1:
            tf_const = self.mt5.TIMEFRAME_M1
        elif tf == 5:
            tf_const = self.mt5.TIMEFRAME_M5
        elif tf == 15:
            tf_const = self.mt5.TIMEFRAME_M15
        elif tf == 30:
            tf_const = self.mt5.TIMEFRAME_M30
        elif tf == 60:
            tf_const = self.mt5.TIMEFRAME_H1
        elif tf == 240:
            tf_const = self.mt5.TIMEFRAME_H4
        elif tf == 1440:
            tf_const = self.mt5.TIMEFRAME_D1
        elif tf == 10080:
            tf_const = self.mt5.TIMEFRAME_W1
        elif tf == 43200:
            tf_const = self.mt5.TIMEFRAME_MN1
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported timeframe: {timeframe}")
        
        if start_time:
            rates = self.mt5.copy_rates_from(symbol, tf_const, start_time, count)
        else:
            rates = self.mt5.copy_rates_from_pos(symbol, tf_const, 0, count)
        
        if rates is None or len(rates) == 0:
            return []
        
        result = []
        for r in rates:
            result.append(PriceDataItem(
                timestamp=datetime.fromtimestamp(r['time']),
                open=float(r['open']),
                high=float(r['high']),
                low=float(r['low']),
                close=float(r['close']),
                tick_volume=int(r['tick_volume']),
                spread=int(r['spread']),
                real_volume=int(r['real_volume'])
            ))
        
        return result
    
    def get_positions(self, symbol: Optional[str] = None) -> List[PositionInfo]:
        """Get open positions."""
        self.ensure_connected()
        
        if symbol:
            positions = self.mt5.positions_get(symbol=symbol)
        else:
            positions = self.mt5.positions_get()
        
        if positions is None:
            return []
        
        result = []
        for pos in positions:
            pos_type = "buy" if pos.type == self.mt5.ORDER_TYPE_BUY else "sell"
            result.append(PositionInfo(
                ticket=pos.ticket,
                symbol=pos.symbol,
                type=pos_type,
                volume=pos.volume,
                price_open=pos.price_open,
                price_current=pos.price_current,
                sl=pos.sl,
                tp=pos.tp,
                profit=pos.profit,
                swap=pos.swap,
                time=datetime.fromtimestamp(pos.time),
                magic=pos.magic,
                comment=pos.comment
            ))
        
        return result
    
    def place_order(self, request: OrderRequest) -> OrderResponse:
        """Place a market order."""
        self.ensure_connected()
        
        # Get symbol info
        symbol_info = self.mt5.symbol_info(request.symbol)
        if symbol_info is None:
            return OrderResponse(success=False, error=f"Symbol {request.symbol} not found")
        
        # Get current price
        tick = self.mt5.symbol_info_tick(request.symbol)
        if tick is None:
            return OrderResponse(success=False, error="Failed to get current price")
        
        # Determine order type and price
        if request.order_type == OrderType.BUY:
            order_type = self.mt5.ORDER_TYPE_BUY
            price = tick.ask
        else:
            order_type = self.mt5.ORDER_TYPE_SELL
            price = tick.bid
        
        # Use provided price if given (for pending orders in future)
        if request.price:
            price = request.price
        
        # Build request
        mt5_request = {
            "action": self.mt5.TRADE_ACTION_DEAL,
            "symbol": request.symbol,
            "volume": request.volume,
            "type": order_type,
            "price": price,
            "sl": request.sl,
            "tp": request.tp,
            "deviation": request.deviation,
            "magic": request.magic,
            "comment": request.comment,
            "type_time": self.mt5.ORDER_TIME_GTC,
            "type_filling": self.mt5.ORDER_FILLING_IOC,
        }
        
        # Send order
        result = self.mt5.order_send(mt5_request)
        
        if result is None:
            error = self.mt5.last_error()
            return OrderResponse(success=False, error=f"Order send failed: {error}")
        
        if result.retcode != self.mt5.TRADE_RETCODE_DONE:
            return OrderResponse(
                success=False, 
                error=f"Order rejected: {result.retcode} - {result.comment}"
            )
        
        logger.info("Order placed", 
                    ticket=result.order, 
                    symbol=request.symbol,
                    type=request.order_type,
                    volume=result.volume,
                    price=result.price)
        
        return OrderResponse(
            success=True,
            ticket=result.order,
            price=result.price,
            volume=result.volume
        )
    
    def close_position(self, request: ClosePositionRequest) -> OrderResponse:
        """Close an open position."""
        self.ensure_connected()
        
        # Get the position
        positions = self.mt5.positions_get(ticket=request.ticket)
        if not positions:
            return OrderResponse(success=False, error=f"Position {request.ticket} not found")
        
        position = positions[0]
        
        # Determine close volume
        volume = request.volume if request.volume else position.volume
        
        # Determine close type and price
        if position.type == self.mt5.ORDER_TYPE_BUY:
            close_type = self.mt5.ORDER_TYPE_SELL
            tick = self.mt5.symbol_info_tick(position.symbol)
            price = tick.bid
        else:
            close_type = self.mt5.ORDER_TYPE_BUY
            tick = self.mt5.symbol_info_tick(position.symbol)
            price = tick.ask
        
        # Build close request
        mt5_request = {
            "action": self.mt5.TRADE_ACTION_DEAL,
            "symbol": position.symbol,
            "volume": volume,
            "type": close_type,
            "position": request.ticket,
            "price": price,
            "deviation": 20,
            "magic": position.magic,
            "comment": "API close",
            "type_time": self.mt5.ORDER_TIME_GTC,
            "type_filling": self.mt5.ORDER_FILLING_IOC,
        }
        
        result = self.mt5.order_send(mt5_request)
        
        if result is None:
            error = self.mt5.last_error()
            return OrderResponse(success=False, error=f"Close failed: {error}")
        
        if result.retcode != self.mt5.TRADE_RETCODE_DONE:
            return OrderResponse(
                success=False,
                error=f"Close rejected: {result.retcode} - {result.comment}"
            )
        
        logger.info("Position closed", ticket=request.ticket, volume=volume, price=result.price)
        
        return OrderResponse(
            success=True,
            ticket=result.order,
            price=result.price,
            volume=result.volume
        )
    
    def modify_position(self, request: ModifyPositionRequest) -> OrderResponse:
        """Modify position SL/TP."""
        self.ensure_connected()
        
        # Get the position
        positions = self.mt5.positions_get(ticket=request.ticket)
        if not positions:
            return OrderResponse(success=False, error=f"Position {request.ticket} not found")
        
        position = positions[0]
        
        # Use current values if not specified
        sl = request.sl if request.sl is not None else position.sl
        tp = request.tp if request.tp is not None else position.tp
        
        mt5_request = {
            "action": self.mt5.TRADE_ACTION_SLTP,
            "symbol": position.symbol,
            "position": request.ticket,
            "sl": sl,
            "tp": tp,
        }
        
        result = self.mt5.order_send(mt5_request)
        
        if result is None:
            error = self.mt5.last_error()
            return OrderResponse(success=False, error=f"Modify failed: {error}")
        
        if result.retcode != self.mt5.TRADE_RETCODE_DONE:
            return OrderResponse(
                success=False,
                error=f"Modify rejected: {result.retcode} - {result.comment}"
            )
        
        logger.info("Position modified", ticket=request.ticket, sl=sl, tp=tp)
        
        return OrderResponse(success=True, ticket=request.ticket)


# =============================================================================
# FastAPI Application
# =============================================================================

# Global bridge instance
bridge = MT5Bridge()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup: Initialize MT5 from environment
    import os
    
    login = int(os.getenv("MT5_LOGIN", "0"))
    password = os.getenv("MT5_PASSWORD", "")
    server = os.getenv("MT5_SERVER", "")
    path = os.getenv("MT5_PATH", "")
    
    if login and password and server:
        try:
            await bridge.initialize(login, password, server, path or None)
            logger.info("MT5 Bridge initialized on startup")
        except Exception as e:
            logger.error("Failed to initialize MT5 on startup", error=str(e))
            # Don't fail startup - allow manual connection
    else:
        logger.warning("MT5 credentials not provided, bridge not auto-connected")
    
    yield
    
    # Shutdown
    await bridge.shutdown()


app = FastAPI(
    title="MT5 API Bridge",
    description="REST API bridge for MetaTrader 5 operations",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# API Key authentication (optional)
async def verify_api_key(x_api_key: Optional[str] = Header(None)):
    """Optional API key verification."""
    import os
    expected_key = os.getenv("MT5_BRIDGE_API_KEY", "")
    if expected_key and x_api_key != expected_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    account = None
    if bridge.connected:
        try:
            account = bridge.get_account_info()
        except:
            pass
    
    return HealthResponse(
        status="healthy" if bridge.connected else "degraded",
        mt5_connected=bridge.connected,
        account_login=account.login if account else None,
        server=account.server if account else None,
        timestamp=datetime.utcnow()
    )


@app.post("/connect")
async def connect(
    login: int,
    password: str,
    server: str,
    path: Optional[str] = None,
    _: bool = Depends(verify_api_key)
):
    """Manually connect to MT5."""
    try:
        await bridge.initialize(login, password, server, path)
        return {"status": "connected", "login": login, "server": server}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/disconnect")
async def disconnect(_: bool = Depends(verify_api_key)):
    """Disconnect from MT5."""
    await bridge.shutdown()
    return {"status": "disconnected"}


@app.get("/account", response_model=AccountInfo)
async def get_account(_: bool = Depends(verify_api_key)):
    """Get account information."""
    return bridge.get_account_info()


@app.get("/tick/{symbol}", response_model=TickData)
async def get_tick(symbol: str, _: bool = Depends(verify_api_key)):
    """Get current tick for a symbol."""
    return bridge.get_tick(symbol)


@app.get("/symbol/{symbol}", response_model=SymbolInfo)
async def get_symbol_info(symbol: str, _: bool = Depends(verify_api_key)):
    """Get symbol information."""
    return bridge.get_symbol_info(symbol)


@app.get("/price_data/{symbol}", response_model=List[PriceDataItem])
async def get_price_data(
    symbol: str, 
    timeframe: str,
    count: int = 100,
    start_time: Optional[datetime] = None,
    _: bool = Depends(verify_api_key)
):
    """Get historical price data (OHLCV)."""
    return bridge.get_price_data(symbol, timeframe, count, start_time)


@app.get("/positions", response_model=List[PositionInfo])
async def get_positions(symbol: Optional[str] = None, _: bool = Depends(verify_api_key)):
    """Get open positions."""
    return bridge.get_positions(symbol)


@app.get("/positions/{ticket}", response_model=PositionInfo)
async def get_position(ticket: int, _: bool = Depends(verify_api_key)):
    """Get a specific position."""
    positions = bridge.get_positions()
    for pos in positions:
        if pos.ticket == ticket:
            return pos
    raise HTTPException(status_code=404, detail=f"Position {ticket} not found")


@app.post("/order", response_model=OrderResponse)
async def place_order(request: OrderRequest, _: bool = Depends(verify_api_key)):
    """Place a market order."""
    return bridge.place_order(request)


@app.post("/positions/{ticket}/close", response_model=OrderResponse)
async def close_position(ticket: int, volume: Optional[float] = None, _: bool = Depends(verify_api_key)):
    """Close a position."""
    return bridge.close_position(ClosePositionRequest(ticket=ticket, volume=volume))


@app.put("/positions/{ticket}", response_model=OrderResponse)
async def modify_position(
    ticket: int, 
    sl: Optional[float] = None, 
    tp: Optional[float] = None,
    _: bool = Depends(verify_api_key)
):
    """Modify position SL/TP."""
    return bridge.modify_position(ModifyPositionRequest(ticket=ticket, sl=sl, tp=tp))


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    import os
    
    host = os.getenv("MT5_BRIDGE_HOST", "0.0.0.0")
    port = int(os.getenv("MT5_BRIDGE_PORT", "8001"))
    
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║                    MT5 API Bridge Server                        ║
╠══════════════════════════════════════════════════════════════════╣
║  Starting on http://{host}:{port}                                   ║
║  Docs: http://{host}:{port}/docs                                    ║
║                                                                  ║
║  Set environment variables:                                      ║
║    MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, MT5_PATH                ║
║    MT5_BRIDGE_API_KEY (optional)                                ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(app, host=host, port=port)
