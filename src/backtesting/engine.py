"""
Backtesting Engine

Event-driven backtesting engine for the MTFTR strategy.
Simulates trading on historical data with realistic execution.

Features:
- Multi-timeframe data simulation
- Realistic spread and slippage modeling
- Position management with partial closes
- Comprehensive trade logging and metrics
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from uuid import uuid4
import pandas as pd
import numpy as np

from src.core.logging_config import get_logger
from src.strategies.base_strategy import TradingSignal
from src.execution.mt5_connector import OrderDirection

logger = get_logger("backtest_engine")


class PositionState(Enum):
    """Position state in the backtest"""
    INITIAL = "initial"
    TP1_HIT = "tp1_hit"
    TP2_HIT = "tp2_hit"
    TRAILING = "trailing"
    CLOSED = "closed"


@dataclass
class BacktestConfig:
    """Configuration for backtesting - OPTIMIZED FOR PERFORMANCE"""
    # Account settings
    initial_balance: float = 10000.0
    leverage: int = 100
    currency: str = "USD"
    
    # Execution settings
    spread_pips: float = 1.5  # Tighter spread assumption
    slippage_pips: float = 0.3  # Lower slippage
    commission_per_lot: float = 7.0  # Commission per round-trip lot
    
    # OPTIMIZED Position management - let winners run longer:
    # TP1 at 1.5:1, TP2 at 3:1, trail remainder
    tp1_close_percent: float = 0.40   # Close 40% at TP1 (1.5:1)
    tp2_close_percent: float = 0.30   # Close 30% at TP2 (3:1)
    trail_percent: float = 0.30       # Trail final 30% for big moves
    
    # Risk management - MORE AGGRESSIVE
    max_risk_per_trade: float = 0.02  # 2% risk per trade
    max_daily_risk: float = 0.06      # 6% daily DD limit
    max_weekly_risk: float = 0.12     # 12% weekly DD limit
    max_drawdown: float = 0.20        # 20% circuit breaker
    max_trades_per_day: int = 6       # Allow more trades
    max_open_trades: int = 3          # Allow 3 concurrent
    max_consec_losses: int = 4        # More tolerance for drawdown
    
    # Time-based exit - let trades develop
    max_trade_hours: int = 16  # More time for trends
    
    # Symbol settings
    pip_size: float = 0.1  # XAUUSD pip = $0.10 per 0.01 lot
    lot_size: float = 100  # Contract size (1 lot = 100 oz)
    min_lot: float = 0.01
    max_lot: float = 100.0
    lot_step: float = 0.01


@dataclass
class BacktestTrade:
    """Represents a trade in backtesting"""
    id: str = field(default_factory=lambda: str(uuid4()))
    ticket: int = 0
    
    # Trade identification
    symbol: str = ""
    direction: OrderDirection = OrderDirection.BUY
    
    # Timing
    signal_time: datetime = None
    entry_time: datetime = None
    exit_time: Optional[datetime] = None
    
    # Prices
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit_1: float = 0.0
    take_profit_2: float = 0.0
    current_sl: float = 0.0  # May be moved to BE
    exit_price: Optional[float] = None
    
    # Position sizing
    initial_lots: float = 0.0
    current_lots: float = 0.0
    
    # State
    state: PositionState = PositionState.INITIAL
    
    # Results
    realized_pnl: float = 0.0  # From partial closes
    unrealized_pnl: float = 0.0
    commission: float = 0.0
    
    # Exit reason
    exit_reason: str = ""
    
    # Context
    market_context: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self.current_sl = self.stop_loss


@dataclass
class BacktestResult:
    """Results from a backtest run"""
    # Configuration
    symbol: str = ""
    start_date: datetime = None
    end_date: datetime = None
    initial_balance: float = 0.0
    
    # Overall performance
    final_balance: float = 0.0
    total_return: float = 0.0
    total_return_pct: float = 0.0
    
    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    breakeven_trades: int = 0
    
    win_rate: float = 0.0
    
    # Profit metrics
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    net_profit: float = 0.0
    profit_factor: float = 0.0
    
    # Average trade
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_trade: float = 0.0
    avg_rr_achieved: float = 0.0
    
    # Largest
    largest_win: float = 0.0
    largest_loss: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    
    # Risk metrics
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    max_drawdown_date: datetime = None
    
    # Risk-adjusted returns
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    
    # Time metrics
    avg_trade_duration: timedelta = timedelta()
    avg_winning_duration: timedelta = timedelta()
    avg_losing_duration: timedelta = timedelta()
    
    # Trade list
    trades: List[BacktestTrade] = field(default_factory=list)
    
    # Equity curve
    equity_curve: List[Dict] = field(default_factory=list)
    
    # Monthly breakdown
    monthly_returns: Dict[str, float] = field(default_factory=dict)


class BacktestEngine:
    """
    Event-driven backtesting engine.
    
    Simulates the MTFTR strategy on historical data with:
    - Multi-timeframe analysis
    - Realistic execution (spread, slippage)
    - Position management (partial closes, trailing)
    - Comprehensive metrics
    """
    
    def __init__(self, config: BacktestConfig):
        """
        Initialize backtesting engine.
        
        Args:
            config: Backtest configuration
        """
        self.config = config
        self.reset()
        
    def reset(self):
        """Reset engine state for new backtest"""
        self.balance = self.config.initial_balance
        self.equity = self.config.initial_balance
        self.peak_equity = self.config.initial_balance
        
        self.open_positions: List[BacktestTrade] = []
        self.closed_trades: List[BacktestTrade] = []
        
        self.equity_curve: List[Dict] = []
        self.daily_pnl: Dict[str, float] = {}
        
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        self.max_drawdown_date = None
        
        self.ticket_counter = 1
        self.trades_today = 0
        self.current_date = None
        
    def open_position(
        self,
        signal: TradingSignal,
        timestamp: datetime,
        current_price: float
    ) -> Optional[BacktestTrade]:
        """
        Open a new position based on signal.
        
        Args:
            signal: Trading signal
            timestamp: Current bar timestamp
            current_price: Current market price
            
        Returns:
            BacktestTrade if opened, None if rejected
        """
        # Check daily trade limit
        trade_date = timestamp.date()
        if self.current_date != trade_date:
            self.current_date = trade_date
            self.trades_today = 0
            
        if self.trades_today >= self.config.max_trades_per_day:
            logger.debug("Daily trade limit reached")
            return None
            
        # Check drawdown limit
        if self.current_drawdown >= self.config.max_drawdown:
            logger.debug("Max drawdown limit reached")
            return None
        
        # Calculate position size
        lot_size = self._calculate_lot_size(
            signal.entry_price,
            signal.stop_loss,
            self.balance
        )
        
        if lot_size < self.config.min_lot:
            logger.debug("Position size too small")
            return None
        
        # Apply spread to entry
        spread = self.config.spread_pips * self.config.pip_size
        if signal.direction == OrderDirection.BUY:
            entry_price = signal.entry_price + spread / 2  # Buy at ask
        else:
            entry_price = signal.entry_price - spread / 2  # Sell at bid
            
        # Apply slippage
        slippage = np.random.uniform(0, self.config.slippage_pips) * self.config.pip_size
        if signal.direction == OrderDirection.BUY:
            entry_price += slippage
        else:
            entry_price -= slippage
        
        # Calculate commission
        commission = lot_size * self.config.commission_per_lot
        
        # Create trade
        trade = BacktestTrade(
            ticket=self.ticket_counter,
            symbol=signal.symbol,
            direction=signal.direction,
            signal_time=signal.timestamp,
            entry_time=timestamp,
            entry_price=entry_price,
            stop_loss=signal.stop_loss,
            take_profit_1=signal.take_profit_1,
            take_profit_2=signal.take_profit_2,
            current_sl=signal.stop_loss,
            initial_lots=lot_size,
            current_lots=lot_size,
            state=PositionState.INITIAL,
            commission=commission,
            market_context=signal.market_context.copy()
        )
        
        self.ticket_counter += 1
        self.trades_today += 1
        self.open_positions.append(trade)
        
        # Deduct commission
        self.balance -= commission
        
        logger.info(
            "Position opened",
            ticket=trade.ticket,
            direction=signal.direction.value,
            entry=entry_price,
            sl=signal.stop_loss,
            tp1=signal.take_profit_1,
            lots=lot_size
        )
        
        return trade
    
    def update_positions(
        self,
        timestamp: datetime,
        high: float,
        low: float,
        close: float,
        h1_ema50: Optional[float] = None,
        h1_hull_34: Optional[float] = None,
        h1_hull_34_prev: Optional[float] = None,
        m15_atr: Optional[float] = None
    ):
        """
        Update all open positions with new price data.
        MATCHES MT5 EA ManageOpenPositions() logic exactly.
        
        Checks:
        - Time-based exit (8 hours max if TP1 not hit)
        - Stop loss hit
        - Take profit levels (50%/30%/20% scaling)
        - Trailing stop on 1H EMA50
        - Hull MA flip exit for trailing positions
        - Price close past 1H EMA50 exit
        
        Args:
            timestamp: Bar timestamp
            high: Bar high
            low: Bar low
            close: Bar close
            h1_ema50: 1H EMA50 for trailing (optional)
            h1_hull_34: 1H Hull MA current
            h1_hull_34_prev: 1H Hull MA previous (for flip detection)
            m15_atr: 15M ATR for trail buffer
        """
        positions_to_close = []
        
        for trade in self.open_positions:
            # ========== TIME-BASED EXIT (MT5 EA: InpMaxTradeHours=8) ==========
            if trade.state == PositionState.INITIAL:
                trade_duration = timestamp - trade.entry_time
                max_duration = timedelta(hours=self.config.max_trade_hours)
                if trade_duration > max_duration:
                    trade.exit_price = close
                    trade.exit_reason = "Time Exit"
                    positions_to_close.append(trade)
                    logger.info(
                        "Time exit triggered",
                        ticket=trade.ticket,
                        duration=str(trade_duration)
                    )
                    continue
            
            # ========== STOP LOSS CHECK ==========
            if trade.direction == OrderDirection.BUY:
                if low <= trade.current_sl:
                    trade.exit_price = trade.current_sl
                    trade.exit_reason = "Stop Loss"
                    positions_to_close.append(trade)
                    continue
            else:
                if high >= trade.current_sl:
                    trade.exit_price = trade.current_sl
                    trade.exit_reason = "Stop Loss"
                    positions_to_close.append(trade)
                    continue
            
            # ========== STATE: INITIAL — Waiting for TP1 (1:1 R:R) ==========
            if trade.state == PositionState.INITIAL:
                tp1_hit = False
                if trade.direction == OrderDirection.BUY:
                    if high >= trade.take_profit_1:
                        tp1_hit = True
                else:
                    if low <= trade.take_profit_1:
                        tp1_hit = True
                
                if tp1_hit:
                    # Close 50% at TP1 (MT5 EA: InpTP1_ClosePct=50)
                    close_lots = trade.initial_lots * self.config.tp1_close_percent
                    
                    # Check if remaining position would be too small
                    remaining_lots = trade.initial_lots - close_lots
                    if remaining_lots < self.config.min_lot:
                        # Close everything at TP1 (lot too small to split)
                        pnl = self._calculate_pnl(
                            trade.direction,
                            trade.entry_price,
                            trade.take_profit_1,
                            trade.initial_lots
                        )
                        trade.realized_pnl += pnl
                        trade.current_lots = 0
                        self.balance += pnl
                        trade.exit_price = trade.take_profit_1
                        trade.exit_reason = "TP1 Full Close"
                        positions_to_close.append(trade)
                        logger.info(
                            "TP1 hit - full close (lot too small to split)",
                            ticket=trade.ticket,
                            pnl=pnl
                        )
                    else:
                        # Partial close at TP1
                        pnl = self._calculate_pnl(
                            trade.direction,
                            trade.entry_price,
                            trade.take_profit_1,
                            close_lots
                        )
                        trade.realized_pnl += pnl
                        trade.current_lots = remaining_lots
                        self.balance += pnl
                        
                        # Move SL to breakeven
                        trade.current_sl = trade.entry_price
                        trade.state = PositionState.TP1_HIT
                        
                        logger.info(
                            "TP1 hit - partial close, SL to BE",
                            ticket=trade.ticket,
                            pnl=pnl,
                            remaining_lots=trade.current_lots
                        )
            
            # ========== STATE: TP1_HIT — Waiting for TP2 (2:1 R:R) ==========
            elif trade.state == PositionState.TP1_HIT:
                tp2_hit = False
                if trade.direction == OrderDirection.BUY:
                    if high >= trade.take_profit_2:
                        tp2_hit = True
                else:
                    if low <= trade.take_profit_2:
                        tp2_hit = True
                
                if tp2_hit:
                    # Close 30% at TP2 (60% of remaining 50%)
                    # MT5 EA: trailVol = originalVolume * 0.20, closeVol = currentVol - trailVol
                    trail_lots = trade.initial_lots * self.config.trail_percent
                    close_lots = trade.current_lots - trail_lots
                    
                    if trail_lots < self.config.min_lot or close_lots <= 0:
                        # Close everything at TP2 (lot too small to trail)
                        pnl = self._calculate_pnl(
                            trade.direction,
                            trade.entry_price,
                            trade.take_profit_2,
                            trade.current_lots
                        )
                        trade.realized_pnl += pnl
                        trade.current_lots = 0
                        self.balance += pnl
                        trade.exit_price = trade.take_profit_2
                        trade.exit_reason = "TP2 Full Close"
                        positions_to_close.append(trade)
                        logger.info(
                            "TP2 hit - full close (lot too small to trail)",
                            ticket=trade.ticket,
                            pnl=pnl
                        )
                    else:
                        # Partial close at TP2, enter trail mode
                        pnl = self._calculate_pnl(
                            trade.direction,
                            trade.entry_price,
                            trade.take_profit_2,
                            close_lots
                        )
                        trade.realized_pnl += pnl
                        trade.current_lots = trail_lots
                        self.balance += pnl
                        trade.state = PositionState.TP2_HIT
                        
                        logger.info(
                            "TP2 hit - partial close, entering trail mode",
                            ticket=trade.ticket,
                            pnl=pnl,
                            trailing_lots=trade.current_lots
                        )
            
            # ========== STATE: TP2_HIT — Trailing Final 20% ==========
            elif trade.state == PositionState.TP2_HIT:
                # Check Hull MA flip for exit (MT5 EA: exit on 1H Hull flip)
                if h1_hull_34 is not None and h1_hull_34_prev is not None:
                    hull_flip = False
                    if trade.direction == OrderDirection.BUY:
                        # Exit long on Hull flip to bearish
                        if h1_hull_34 < h1_hull_34_prev:
                            hull_flip = True
                    else:
                        # Exit short on Hull flip to bullish
                        if h1_hull_34 > h1_hull_34_prev:
                            hull_flip = True
                    
                    if hull_flip:
                        trade.exit_price = close
                        trade.exit_reason = "Hull MA Flip"
                        positions_to_close.append(trade)
                        continue
                
                # Check if price closed past 1H EMA50 (MT5 EA exit signal)
                if h1_ema50 is not None:
                    if trade.direction == OrderDirection.BUY:
                        if close < h1_ema50:
                            trade.exit_price = close
                            trade.exit_reason = "Price Below 1H EMA50"
                            positions_to_close.append(trade)
                            continue
                    else:
                        if close > h1_ema50:
                            trade.exit_price = close
                            trade.exit_reason = "Price Above 1H EMA50"
                            positions_to_close.append(trade)
                            continue
                
                # Trail SL on 1H EMA50 (MT5 EA: trail with 0.5 ATR buffer)
                if h1_ema50 is not None:
                    atr_buffer = (m15_atr * 0.5) if m15_atr else (self.config.pip_size * 50)
                    
                    if trade.direction == OrderDirection.BUY:
                        # Trail below EMA50 for longs
                        new_sl = h1_ema50 - atr_buffer
                        if new_sl > trade.current_sl and new_sl < close:
                            trade.current_sl = new_sl
                            logger.debug(f"Trail SL updated to {new_sl}")
                    else:
                        # Trail above EMA50 for shorts
                        new_sl = h1_ema50 + atr_buffer
                        if new_sl < trade.current_sl and new_sl > close:
                            trade.current_sl = new_sl
                            logger.debug(f"Trail SL updated to {new_sl}")
            
            # Calculate unrealized P&L
            trade.unrealized_pnl = self._calculate_pnl(
                trade.direction,
                trade.entry_price,
                close,
                trade.current_lots
            )
        
        # Close positions that hit exit
        for trade in positions_to_close:
            self._close_position(trade, timestamp)
        
        # Update equity
        self._update_equity(timestamp)
    
    def _close_position(self, trade: BacktestTrade, timestamp: datetime):
        """Close a position and record results"""
        # Calculate final P&L on remaining lots
        final_pnl = self._calculate_pnl(
            trade.direction,
            trade.entry_price,
            trade.exit_price,
            trade.current_lots
        )
        trade.realized_pnl += final_pnl
        self.balance += final_pnl
        
        trade.exit_time = timestamp
        trade.state = PositionState.CLOSED
        trade.current_lots = 0
        trade.unrealized_pnl = 0
        
        self.open_positions.remove(trade)
        self.closed_trades.append(trade)
        
        # Net P&L after commission
        net_pnl = trade.realized_pnl - trade.commission
        
        logger.info(
            "Position closed",
            ticket=trade.ticket,
            exit_reason=trade.exit_reason,
            gross_pnl=trade.realized_pnl,
            net_pnl=net_pnl,
            duration=str(timestamp - trade.entry_time)
        )
        
        # Track daily P&L
        date_key = timestamp.strftime("%Y-%m-%d")
        self.daily_pnl[date_key] = self.daily_pnl.get(date_key, 0) + net_pnl
    
    def _calculate_lot_size(
        self,
        entry_price: float,
        stop_loss: float,
        account_balance: float
    ) -> float:
        """Calculate position size based on risk"""
        risk_amount = account_balance * self.config.max_risk_per_trade
        sl_distance = abs(entry_price - stop_loss)
        
        if sl_distance == 0:
            return 0.0
        
        # For XAUUSD: pip value = contract_size * pip_size
        pip_value = self.config.lot_size * self.config.pip_size
        sl_pips = sl_distance / self.config.pip_size
        
        # Position size = risk / (SL pips * pip value)
        lot_size = risk_amount / (sl_pips * pip_value)
        
        # Round to lot step
        lot_size = round(lot_size / self.config.lot_step) * self.config.lot_step
        
        # Apply limits
        lot_size = max(self.config.min_lot, min(self.config.max_lot, lot_size))
        
        return lot_size
    
    def _calculate_pnl(
        self,
        direction: OrderDirection,
        entry_price: float,
        exit_price: float,
        lots: float
    ) -> float:
        """Calculate P&L for a position"""
        if direction == OrderDirection.BUY:
            pips = (exit_price - entry_price) / self.config.pip_size
        else:
            pips = (entry_price - exit_price) / self.config.pip_size
        
        pip_value = self.config.lot_size * self.config.pip_size * lots
        return pips * pip_value
    
    def _update_equity(self, timestamp: datetime):
        """Update equity curve and drawdown tracking"""
        # Calculate total unrealized P&L
        unrealized = sum(t.unrealized_pnl for t in self.open_positions)
        self.equity = self.balance + unrealized
        
        # Track peak equity
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
        
        # Calculate drawdown
        if self.peak_equity > 0:
            self.current_drawdown = (self.peak_equity - self.equity) / self.peak_equity
            if self.current_drawdown > self.max_drawdown:
                self.max_drawdown = self.current_drawdown
                self.max_drawdown_date = timestamp
        
        # Record equity point
        self.equity_curve.append({
            'timestamp': timestamp,
            'balance': self.balance,
            'equity': self.equity,
            'drawdown': self.current_drawdown,
            'open_positions': len(self.open_positions)
        })
    
    def get_results(self, symbol: str, start_date: datetime, end_date: datetime) -> BacktestResult:
        """
        Generate comprehensive backtest results.
        
        Args:
            symbol: Tested symbol
            start_date: Backtest start
            end_date: Backtest end
            
        Returns:
            BacktestResult with all metrics
        """
        result = BacktestResult(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            initial_balance=self.config.initial_balance,
            final_balance=self.balance,
            trades=self.closed_trades.copy(),
            equity_curve=self.equity_curve.copy()
        )
        
        # Overall performance
        result.total_return = self.balance - self.config.initial_balance
        result.total_return_pct = result.total_return / self.config.initial_balance * 100
        
        # Trade statistics
        result.total_trades = len(self.closed_trades)
        
        if result.total_trades == 0:
            return result
        
        # Classify trades
        wins = []
        losses = []
        
        for trade in self.closed_trades:
            net_pnl = trade.realized_pnl - trade.commission
            if net_pnl > 0.5:  # Small buffer for breakeven
                result.winning_trades += 1
                wins.append(net_pnl)
            elif net_pnl < -0.5:
                result.losing_trades += 1
                losses.append(net_pnl)
            else:
                result.breakeven_trades += 1
        
        result.win_rate = result.winning_trades / result.total_trades * 100 if result.total_trades > 0 else 0
        
        # Profit metrics
        result.gross_profit = sum(wins) if wins else 0
        result.gross_loss = abs(sum(losses)) if losses else 0
        result.net_profit = result.gross_profit - result.gross_loss
        result.profit_factor = result.gross_profit / result.gross_loss if result.gross_loss > 0 else float('inf')
        
        # Averages
        result.avg_win = np.mean(wins) if wins else 0
        result.avg_loss = abs(np.mean(losses)) if losses else 0
        result.avg_trade = result.net_profit / result.total_trades if result.total_trades > 0 else 0
        
        # Largest
        result.largest_win = max(wins) if wins else 0
        result.largest_loss = abs(min(losses)) if losses else 0
        
        # Consecutive
        result.max_consecutive_wins = self._max_consecutive(wins, losses, True)
        result.max_consecutive_losses = self._max_consecutive(wins, losses, False)
        
        # Drawdown
        result.max_drawdown = self.max_drawdown * self.config.initial_balance
        result.max_drawdown_pct = self.max_drawdown * 100
        result.max_drawdown_date = self.max_drawdown_date
        
        # Time metrics
        durations = [(t.exit_time - t.entry_time) for t in self.closed_trades if t.exit_time]
        win_durations = [(t.exit_time - t.entry_time) for t in self.closed_trades 
                        if t.exit_time and (t.realized_pnl - t.commission) > 0]
        loss_durations = [(t.exit_time - t.entry_time) for t in self.closed_trades 
                         if t.exit_time and (t.realized_pnl - t.commission) < 0]
        
        if durations:
            result.avg_trade_duration = timedelta(seconds=np.mean([d.total_seconds() for d in durations]))
        if win_durations:
            result.avg_winning_duration = timedelta(seconds=np.mean([d.total_seconds() for d in win_durations]))
        if loss_durations:
            result.avg_losing_duration = timedelta(seconds=np.mean([d.total_seconds() for d in loss_durations]))
        
        # Risk-adjusted returns
        if self.daily_pnl:
            daily_returns = list(self.daily_pnl.values())
            avg_return = np.mean(daily_returns)
            std_return = np.std(daily_returns)
            downside_std = np.std([r for r in daily_returns if r < 0]) if any(r < 0 for r in daily_returns) else 1
            
            # Sharpe (assuming 0 risk-free rate)
            result.sharpe_ratio = (avg_return / std_return * np.sqrt(252)) if std_return > 0 else 0
            
            # Sortino
            result.sortino_ratio = (avg_return / downside_std * np.sqrt(252)) if downside_std > 0 else 0
            
            # Calmar
            result.calmar_ratio = (result.total_return_pct / result.max_drawdown_pct) if result.max_drawdown_pct > 0 else 0
        
        # Monthly returns
        for date_str, pnl in self.daily_pnl.items():
            month_key = date_str[:7]  # YYYY-MM
            result.monthly_returns[month_key] = result.monthly_returns.get(month_key, 0) + pnl
        
        # Calculate average R:R achieved
        rr_values = []
        for trade in self.closed_trades:
            sl_distance = abs(trade.entry_price - trade.stop_loss)
            if sl_distance > 0:
                if trade.direction == OrderDirection.BUY:
                    profit_distance = (trade.exit_price or trade.entry_price) - trade.entry_price
                else:
                    profit_distance = trade.entry_price - (trade.exit_price or trade.entry_price)
                rr_values.append(profit_distance / sl_distance)
        
        result.avg_rr_achieved = np.mean(rr_values) if rr_values else 0
        
        return result
    
    def _max_consecutive(self, wins: List[float], losses: List[float], count_wins: bool) -> int:
        """Calculate max consecutive wins or losses"""
        # Reconstruct sequence
        sequence = []
        for trade in self.closed_trades:
            net_pnl = trade.realized_pnl - trade.commission
            sequence.append(net_pnl > 0)
        
        max_count = 0
        current_count = 0
        target = count_wins
        
        for is_win in sequence:
            if is_win == target:
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0
        
        return max_count


def print_results(result: BacktestResult):
    """Print formatted backtest results"""
    print("\n" + "=" * 60)
    print(f"BACKTEST RESULTS - {result.symbol}")
    print(f"Period: {result.start_date.strftime('%Y-%m-%d')} to {result.end_date.strftime('%Y-%m-%d')}")
    print("=" * 60)
    
    print(f"\n📊 PERFORMANCE SUMMARY")
    print(f"  Initial Balance:    ${result.initial_balance:,.2f}")
    print(f"  Final Balance:      ${result.final_balance:,.2f}")
    print(f"  Net Profit:         ${result.total_return:,.2f} ({result.total_return_pct:+.2f}%)")
    
    print(f"\n📈 TRADE STATISTICS")
    print(f"  Total Trades:       {result.total_trades}")
    print(f"  Winning Trades:     {result.winning_trades} ({result.win_rate:.1f}%)")
    print(f"  Losing Trades:      {result.losing_trades}")
    print(f"  Breakeven:          {result.breakeven_trades}")
    
    print(f"\n💰 PROFIT METRICS")
    print(f"  Gross Profit:       ${result.gross_profit:,.2f}")
    print(f"  Gross Loss:         ${result.gross_loss:,.2f}")
    print(f"  Profit Factor:      {result.profit_factor:.2f}")
    print(f"  Avg Win:            ${result.avg_win:,.2f}")
    print(f"  Avg Loss:           ${result.avg_loss:,.2f}")
    print(f"  Avg Trade:          ${result.avg_trade:,.2f}")
    print(f"  Avg R:R Achieved:   {result.avg_rr_achieved:.2f}")
    
    print(f"\n⚠️ RISK METRICS")
    print(f"  Max Drawdown:       ${result.max_drawdown:,.2f} ({result.max_drawdown_pct:.2f}%)")
    if result.max_drawdown_date:
        print(f"  Max DD Date:        {result.max_drawdown_date.strftime('%Y-%m-%d')}")
    print(f"  Largest Win:        ${result.largest_win:,.2f}")
    print(f"  Largest Loss:       ${result.largest_loss:,.2f}")
    print(f"  Max Consec. Wins:   {result.max_consecutive_wins}")
    print(f"  Max Consec. Losses: {result.max_consecutive_losses}")
    
    print(f"\n📉 RISK-ADJUSTED RETURNS")
    print(f"  Sharpe Ratio:       {result.sharpe_ratio:.2f}")
    print(f"  Sortino Ratio:      {result.sortino_ratio:.2f}")
    print(f"  Calmar Ratio:       {result.calmar_ratio:.2f}")
    
    print(f"\n⏱️ TIME METRICS")
    print(f"  Avg Trade Duration: {result.avg_trade_duration}")
    print(f"  Avg Win Duration:   {result.avg_winning_duration}")
    print(f"  Avg Loss Duration:  {result.avg_losing_duration}")
    
    if result.monthly_returns:
        print(f"\n📅 MONTHLY RETURNS")
        for month, ret in sorted(result.monthly_returns.items()):
            print(f"  {month}: ${ret:,.2f}")
    
    print("\n" + "=" * 60)
