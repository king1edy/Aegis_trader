"""
Notifications Module
====================

Centralized notification system for the trading automation platform.
Currently supports Telegram notifications with rich formatting.

Usage:
    from src.notifications import NotificationService
    
    # Get singleton instance
    notifier = NotificationService.get_instance()
    
    # Send trade notification
    await notifier.notify_trade_opened(
        symbol="XAUUSD",
        direction="BUY",
        entry_price=2050.50,
        stop_loss=2040.00,
        take_profit_1=2075.00,
        lot_size=0.1,
        confidence=85.0
    )
    
    # Send system status
    await notifier.notify_system_status("started", {"version": "1.0.0", "balance": 10000})
"""

from src.notifications.telegram import TelegramNotifier
from src.notifications.service import NotificationService
from src.notifications.message_formatter import MessageFormatter

__all__ = [
    "TelegramNotifier",
    "NotificationService", 
    "MessageFormatter"
]
