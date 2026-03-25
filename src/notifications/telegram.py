"""
Telegram Notifier
=================

Async Telegram notification sender using python-telegram-bot library.
Sends formatted messages for trading events.
"""

import asyncio
from typing import Optional, Dict, Any

from telegram import Bot
from telegram.error import TelegramError
from telegram.constants import ParseMode

from src.core.config import Settings
from src.core.logging_config import get_logger
from src.notifications.message_formatter import MessageFormatter

logger = get_logger("telegram_notifier")


class TelegramNotifier:
    """
    Async Telegram notification sender.
    
    Uses python-telegram-bot library to send formatted messages.
    Gracefully handles errors without interrupting trading operations.
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize Telegram notifier.
        
        Args:
            settings: Application settings containing Telegram configuration
        """
        self.settings = settings
        self.enabled = settings.telegram_enabled
        self.chat_id = settings.telegram_chat_id
        self.formatter = MessageFormatter
        
        # Initialize bot if enabled
        self._bot: Optional[Bot] = None
        if self.enabled and settings.telegram_bot_token:
            self._bot = Bot(token=settings.telegram_bot_token)
            logger.info(
                "Telegram notifier initialized",
                chat_id=self.chat_id[:4] + "***" if self.chat_id else "not set"
            )
        elif self.enabled:
            logger.warning("Telegram enabled but bot token not configured")
            self.enabled = False
        else:
            logger.info("Telegram notifications disabled")
    
    async def _send_message(
        self, 
        text: str, 
        parse_mode: str = ParseMode.MARKDOWN
    ) -> bool:
        """
        Send a message to the configured Telegram chat.
        
        Args:
            text: Message text (Markdown formatted)
            parse_mode: Parse mode (default: Markdown)
            
        Returns:
            True if message sent successfully, False otherwise
        """
        if not self.enabled or not self._bot:
            return False
        
        if not self.chat_id:
            logger.warning("Telegram chat_id not configured")
            return False
        
        try:
            await self._bot.send_message(
                chat_id=self.chat_id,
                text=text,
                parse_mode=parse_mode,
                disable_web_page_preview=True
            )
            logger.debug("Telegram message sent successfully")
            return True
            
        except TelegramError as e:
            logger.error(
                "Failed to send Telegram message",
                error=str(e),
                error_type=type(e).__name__
            )
            return False
        except Exception as e:
            logger.error(
                "Unexpected error sending Telegram message",
                error=str(e),
                error_type=type(e).__name__
            )
            return False
    
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
        Send trade opened notification.
        
        Args:
            symbol: Trading symbol
            direction: BUY or SELL
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit_1: First take profit target
            take_profit_2: Second take profit target
            lot_size: Position size
            confidence: Signal confidence
            reason: Trade reason
            
        Returns:
            True if notification sent successfully
        """
        if not self.settings.notify_on_trade_open:
            return False
        
        message = self.formatter.format_trade_opened(
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
        
        return await self._send_message(message)
    
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
        Send trade closed notification.
        
        Args:
            symbol: Trading symbol
            direction: BUY or SELL
            entry_price: Entry price
            exit_price: Exit price
            profit_loss: Profit/loss amount
            lot_size: Position size closed
            duration_minutes: Trade duration
            exit_reason: Reason for exit
            is_partial: Whether partial close
            partial_percent: Percentage closed
            
        Returns:
            True if notification sent successfully
        """
        if not self.settings.notify_on_trade_close:
            return False
        
        message = self.formatter.format_trade_closed(
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
        
        return await self._send_message(message)
    
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
        Send signal generated notification.
        
        Returns:
            True if notification sent successfully
        """
        if not self.settings.notify_on_signal_generated:
            return False
        
        message = self.formatter.format_signal_generated(
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
        
        return await self._send_message(message)
    
    async def notify_signal_rejected(
        self,
        symbol: str,
        direction: str,
        reason: str
    ) -> bool:
        """
        Send signal rejected notification.
        
        Returns:
            True if notification sent successfully
        """
        message = self.formatter.format_signal_rejected(
            symbol=symbol,
            direction=direction,
            reason=reason
        )
        
        return await self._send_message(message)
    
    async def notify_risk_warning(
        self,
        warning_type: str,
        current_value: float,
        limit_value: float,
        symbol: Optional[str] = None,
        additional_info: str = ""
    ) -> bool:
        """
        Send risk warning notification.
        
        Args:
            warning_type: Type of warning
            current_value: Current value
            limit_value: Limit value
            symbol: Related symbol
            additional_info: Additional context
            
        Returns:
            True if notification sent successfully
        """
        if not self.settings.notify_on_drawdown_warning:
            return False
        
        message = self.formatter.format_risk_warning(
            warning_type=warning_type,
            current_value=current_value,
            limit_value=limit_value,
            symbol=symbol,
            additional_info=additional_info
        )
        
        return await self._send_message(message)
    
    async def notify_system_status(
        self,
        status: str,
        details: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Send system status notification.
        
        Args:
            status: Status type (started, stopped, reconnected, etc.)
            details: Additional details
            
        Returns:
            True if notification sent successfully
        """
        message = self.formatter.format_system_status(
            status=status,
            details=details
        )
        
        return await self._send_message(message)
    
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
        Send daily summary notification.
        
        Returns:
            True if notification sent successfully
        """
        if not self.settings.notify_on_daily_summary:
            return False
        
        message = self.formatter.format_daily_summary(
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
        
        return await self._send_message(message)
    
    async def send_custom_message(self, message: str) -> bool:
        """
        Send a custom message.
        
        Args:
            message: Message text (Markdown formatted)
            
        Returns:
            True if notification sent successfully
        """
        return await self._send_message(message)
