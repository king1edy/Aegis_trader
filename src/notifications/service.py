"""
Notification Service
====================

Central facade for all notification channels.
Currently supports Telegram, designed for extensibility.
"""

from typing import Optional, Dict, Any

from src.core.config import Settings, settings as default_settings
from src.core.logging_config import get_logger
from src.notifications.telegram import TelegramNotifier

logger = get_logger("notification_service")


class NotificationService:
    """
    Central notification service facade.
    
    Coordinates all notification channels (currently Telegram).
    Provides a single interface for the trading system to send notifications.
    
    Usage:
        from src.notifications import NotificationService
        
        # Get singleton instance
        notifier = NotificationService.get_instance()
        
        # Send notifications
        await notifier.notify_trade_opened(...)
        await notifier.notify_system_status("started", {...})
    """
    
    _instance: Optional["NotificationService"] = None
    
    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize notification service.
        
        Args:
            settings: Application settings (uses global settings if not provided)
        """
        self.settings = settings or default_settings
        
        # Initialize Telegram notifier
        self.telegram = TelegramNotifier(self.settings)
        
        # Track enabled channels
        self._channels_enabled = []
        if self.telegram.enabled:
            self._channels_enabled.append("telegram")
        
        if self._channels_enabled:
            logger.info(
                "Notification service initialized",
                channels=self._channels_enabled
            )
        else:
            logger.info("All notification channels disabled")
    
    @classmethod
    def get_instance(cls, settings: Optional[Settings] = None) -> "NotificationService":
        """
        Get or create singleton instance.
        
        Args:
            settings: Application settings (only used on first call)
            
        Returns:
            NotificationService singleton instance
        """
        if cls._instance is None:
            cls._instance = cls(settings)
        return cls._instance
    
    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (for testing)."""
        cls._instance = None
    
    @property
    def is_enabled(self) -> bool:
        """Check if any notification channel is enabled."""
        return len(self._channels_enabled) > 0
    
    # =========================================================================
    # Trade Notifications
    # =========================================================================
    
    async def notify_trade_opened(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        take_profit_1: float,
        take_profit_2: Optional[float] = None,
        lot_size: float = 0.0,
        confidence: float = 0.0,
        reason: str = ""
    ) -> bool:
        """
        Notify when a trade is opened.
        
        Returns:
            True if at least one channel succeeded
        """
        results = []
        
        if self.telegram.enabled:
            result = await self.telegram.notify_trade_opened(
                symbol=symbol,
                direction=direction,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit_1=take_profit_1,
                take_profit_2=take_profit_2,
                lot_size=lot_size,
                confidence=confidence,
                reason=reason
            )
            results.append(result)
        
        return any(results)
    
    async def notify_trade_closed(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        exit_price: float,
        profit_loss: float,
        lot_size: float,
        duration_minutes: int,
        exit_reason: str = "",
        is_partial: bool = False,
        partial_percent: int = 0
    ) -> bool:
        """
        Notify when a trade is closed (full or partial).
        
        Returns:
            True if at least one channel succeeded
        """
        results = []
        
        if self.telegram.enabled:
            result = await self.telegram.notify_trade_closed(
                symbol=symbol,
                direction=direction,
                entry_price=entry_price,
                exit_price=exit_price,
                profit_loss=profit_loss,
                lot_size=lot_size,
                duration_minutes=duration_minutes,
                exit_reason=exit_reason,
                is_partial=is_partial,
                partial_percent=partial_percent
            )
            results.append(result)
        
        return any(results)
    
    # =========================================================================
    # Signal Notifications
    # =========================================================================
    
    async def notify_signal_generated(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        confidence: float,
        timeframe: str = "15M",
        reason: str = "",
        was_executed: bool = False
    ) -> bool:
        """
        Notify when a signal is generated.
        
        Returns:
            True if at least one channel succeeded
        """
        results = []
        
        if self.telegram.enabled:
            result = await self.telegram.notify_signal_generated(
                symbol=symbol,
                direction=direction,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                confidence=confidence,
                timeframe=timeframe,
                reason=reason,
                was_executed=was_executed
            )
            results.append(result)
        
        return any(results)
    
    async def notify_signal_rejected(
        self,
        symbol: str,
        direction: str,
        reason: str
    ) -> bool:
        """
        Notify when a signal is rejected by risk checks.
        
        Returns:
            True if at least one channel succeeded
        """
        results = []
        
        if self.telegram.enabled:
            result = await self.telegram.notify_signal_rejected(
                symbol=symbol,
                direction=direction,
                reason=reason
            )
            results.append(result)
        
        return any(results)
    
    # =========================================================================
    # Risk Notifications
    # =========================================================================
    
    async def notify_risk_warning(
        self,
        warning_type: str,
        current_value: float,
        limit_value: float,
        symbol: Optional[str] = None,
        additional_info: str = ""
    ) -> bool:
        """
        Notify on risk warning (drawdown, daily loss, etc.).
        
        Returns:
            True if at least one channel succeeded
        """
        results = []
        
        if self.telegram.enabled:
            result = await self.telegram.notify_risk_warning(
                warning_type=warning_type,
                current_value=current_value,
                limit_value=limit_value,
                symbol=symbol,
                additional_info=additional_info
            )
            results.append(result)
        
        return any(results)
    
    # =========================================================================
    # System Notifications
    # =========================================================================
    
    async def notify_system_status(
        self,
        status: str,
        details: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Notify on system status changes.
        
        Args:
            status: Status type (started, stopped, reconnected, etc.)
            details: Additional details
            
        Returns:
            True if at least one channel succeeded
        """
        results = []
        
        if self.telegram.enabled:
            result = await self.telegram.notify_system_status(
                status=status,
                details=details
            )
            results.append(result)
        
        return any(results)
    
    async def notify_daily_summary(
        self,
        date: str,
        total_trades: int,
        winning_trades: int,
        losing_trades: int,
        gross_profit: float,
        gross_loss: float,
        net_pnl: float,
        win_rate: float,
        starting_balance: float,
        ending_balance: float
    ) -> bool:
        """
        Notify daily trading summary.
        
        Returns:
            True if at least one channel succeeded
        """
        results = []
        
        if self.telegram.enabled:
            result = await self.telegram.notify_daily_summary(
                date=date,
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                gross_profit=gross_profit,
                gross_loss=gross_loss,
                net_pnl=net_pnl,
                win_rate=win_rate,
                starting_balance=starting_balance,
                ending_balance=ending_balance
            )
            results.append(result)
        
        return any(results)
    
    # =========================================================================
    # Custom Messages
    # =========================================================================
    
    async def send_custom_message(self, message: str) -> bool:
        """
        Send a custom message to all enabled channels.
        
        Args:
            message: Message text (Markdown formatted)
            
        Returns:
            True if at least one channel succeeded
        """
        results = []
        
        if self.telegram.enabled:
            result = await self.telegram.send_custom_message(message)
            results.append(result)
        
        return any(results)
