"""
Background Risk Monitor Service
Always-on monitoring for drawdown protection.
"""

import asyncio
from datetime import datetime, timezone
from typing import Optional, Callable, Awaitable
from dataclasses import dataclass

from src.core.config import settings
from src.core.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class RiskMonitorConfig:
    """Configuration for risk monitoring thresholds."""
    check_interval_seconds: float = 18.0  # ~18 second checks
    max_balance_drawdown_pct: float = 10.0  # Max % drawdown from peak balance
    max_equity_drawdown_pct: float = 15.0  # Max % drawdown from peak equity
    max_daily_loss_pct: float = 5.0  # Max daily loss %
    max_weekly_loss_pct: float = 10.0  # Max weekly loss %
    min_margin_level_pct: float = 150.0  # Minimum margin level before forced close
    emergency_margin_level_pct: float = 100.0  # Emergency close all level


@dataclass 
class RiskState:
    """Current risk state snapshot."""
    peak_balance: float
    peak_equity: float
    daily_start_balance: float
    weekly_start_balance: float
    current_balance: float
    current_equity: float
    margin_level: float
    balance_drawdown_pct: float
    equity_drawdown_pct: float
    daily_loss_pct: float
    weekly_loss_pct: float
    is_trading_blocked: bool = False
    block_reason: Optional[str] = None


class RiskMonitor:
    """
    Background risk monitor that runs independently.
    
    Enforces drawdown limits early - deeper losses require
    disproportionately larger returns to recover.
    """
    
    def __init__(
        self,
        config: RiskMonitorConfig,
        broker,
        on_threshold_breach: Optional[Callable[[str, RiskState], Awaitable[None]]] = None,
        on_emergency_close: Optional[Callable[[str], Awaitable[None]]] = None
    ):
        self.config = config
        self.broker = broker
        self.on_threshold_breach = on_threshold_breach
        self.on_emergency_close = on_emergency_close
        
        self._running = False
        self._trading_blocked = False
        self._block_reason: Optional[str] = None
        
        # Peak tracking
        self._peak_balance: float = 0.0
        self._peak_equity: float = 0.0
        self._daily_start_balance: float = 0.0
        self._weekly_start_balance: float = 0.0
        self._last_daily_reset: Optional[datetime] = None
        self._last_weekly_reset: Optional[datetime] = None
    
    @property
    def is_trading_blocked(self) -> bool:
        return self._trading_blocked
    
    @property
    def block_reason(self) -> Optional[str]:
        return self._block_reason
    
    async def start(self) -> None:
        """Start the background monitoring task."""
        self._running = True
        
        # Initialize peaks from current account
        account = await self.broker.get_account_info()
        balance = self._get_balance(account)
        equity = self._get_equity(account)
        
        self._peak_balance = balance
        self._peak_equity = equity
        self._daily_start_balance = balance
        self._weekly_start_balance = balance
        self._last_daily_reset = datetime.now(timezone.utc)
        self._last_weekly_reset = datetime.now(timezone.utc)
        
        logger.info(
            "Risk monitor started",
            peak_balance=self._peak_balance,
            check_interval=self.config.check_interval_seconds
        )
        
        asyncio.create_task(self._monitor_loop())
    
    async def stop(self) -> None:
        """Stop the monitoring task."""
        self._running = False
        logger.info("Risk monitor stopped")
    
    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                await self._check_risk_state()
                await asyncio.sleep(self.config.check_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception("Error in risk monitor", error=str(e))
                await asyncio.sleep(5)
    
    async def _check_risk_state(self) -> None:
        """Check current risk state and enforce limits."""
        account = await self.broker.get_account_info()
        
        balance = self._get_balance(account)
        equity = self._get_equity(account)
        margin_level = self._get_margin_level(account)
        
        # Update peaks (only when increasing)
        if balance > self._peak_balance:
            self._peak_balance = balance
        if equity > self._peak_equity:
            self._peak_equity = equity
        
        # Check for daily/weekly reset
        await self._check_period_resets(balance)
        
        # Calculate drawdowns
        balance_dd = ((self._peak_balance - balance) / self._peak_balance * 100) if self._peak_balance > 0 else 0
        equity_dd = ((self._peak_equity - equity) / self._peak_equity * 100) if self._peak_equity > 0 else 0
        daily_loss = ((self._daily_start_balance - balance) / self._daily_start_balance * 100) if self._daily_start_balance > 0 else 0
        weekly_loss = ((self._weekly_start_balance - balance) / self._weekly_start_balance * 100) if self._weekly_start_balance > 0 else 0
        
        state = RiskState(
            peak_balance=self._peak_balance,
            peak_equity=self._peak_equity,
            daily_start_balance=self._daily_start_balance,
            weekly_start_balance=self._weekly_start_balance,
            current_balance=balance,
            current_equity=equity,
            margin_level=margin_level,
            balance_drawdown_pct=balance_dd,
            equity_drawdown_pct=equity_dd,
            daily_loss_pct=daily_loss,
            weekly_loss_pct=weekly_loss,
            is_trading_blocked=self._trading_blocked,
            block_reason=self._block_reason
        )
        
        # Check thresholds
        await self._enforce_limits(state)
    
    async def _enforce_limits(self, state: RiskState) -> None:
        """Enforce risk limits and take action if breached."""
        breach_reason = None
        emergency = False
        
        # Emergency margin check
        if state.margin_level > 0 and state.margin_level < self.config.emergency_margin_level_pct:
            breach_reason = f"EMERGENCY: Margin level {state.margin_level:.1f}% below {self.config.emergency_margin_level_pct}%"
            emergency = True
        
        # Balance drawdown
        elif state.balance_drawdown_pct >= self.config.max_balance_drawdown_pct:
            breach_reason = f"Balance drawdown {state.balance_drawdown_pct:.2f}% exceeds limit {self.config.max_balance_drawdown_pct}%"
        
        # Equity drawdown
        elif state.equity_drawdown_pct >= self.config.max_equity_drawdown_pct:
            breach_reason = f"Equity drawdown {state.equity_drawdown_pct:.2f}% exceeds limit {self.config.max_equity_drawdown_pct}%"
        
        # Daily loss
        elif state.daily_loss_pct >= self.config.max_daily_loss_pct:
            breach_reason = f"Daily loss {state.daily_loss_pct:.2f}% exceeds limit {self.config.max_daily_loss_pct}%"
        
        # Weekly loss
        elif state.weekly_loss_pct >= self.config.max_weekly_loss_pct:
            breach_reason = f"Weekly loss {state.weekly_loss_pct:.2f}% exceeds limit {self.config.max_weekly_loss_pct}%"
        
        # Margin warning
        elif state.margin_level > 0 and state.margin_level < self.config.min_margin_level_pct:
            breach_reason = f"Margin level {state.margin_level:.1f}% below {self.config.min_margin_level_pct}%"
        
        if breach_reason:
            logger.warning("Risk threshold breached", reason=breach_reason)
            self._trading_blocked = True
            self._block_reason = breach_reason
            
            if self.on_threshold_breach:
                await self.on_threshold_breach(breach_reason, state)
            
            if emergency and self.on_emergency_close:
                await self.on_emergency_close(breach_reason)
    
    async def _check_period_resets(self, current_balance: float) -> None:
        """Reset daily/weekly tracking at period boundaries."""
        now = datetime.now(timezone.utc)
        
        # Daily reset (at midnight UTC)
        if self._last_daily_reset and now.date() > self._last_daily_reset.date():
            self._daily_start_balance = current_balance
            self._last_daily_reset = now
            logger.info("Daily period reset", new_start_balance=current_balance)
        
        # Weekly reset (Monday midnight UTC)
        if self._last_weekly_reset:
            days_since_reset = (now - self._last_weekly_reset).days
            if days_since_reset >= 7 or (now.weekday() == 0 and self._last_weekly_reset.weekday() != 0):
                self._weekly_start_balance = current_balance
                self._last_weekly_reset = now
                logger.info("Weekly period reset", new_start_balance=current_balance)
    
    def reset_block(self) -> None:
        """Manually reset trading block (use with caution)."""
        self._trading_blocked = False
        self._block_reason = None
        logger.info("Trading block manually reset")
    
    def _get_balance(self, account) -> float:
        """Extract balance from account object."""
        if isinstance(account, dict):
            return float(account["balance"])
        return float(account.balance)
    
    def _get_equity(self, account) -> float:
        """Extract equity from account object."""
        if isinstance(account, dict):
            return float(account["equity"])
        return float(account.equity)
    
    def _get_margin_level(self, account) -> float:
        """Extract margin level from account object."""
        if isinstance(account, dict):
            return float(account.get("margin_level", 0))
        return float(getattr(account, "margin_level", 0))