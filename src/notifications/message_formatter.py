"""
Message Formatter
=================

Formats notification messages with rich emoji styling for Telegram.
All messages use Markdown formatting.
"""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional, Dict, Any


class MessageFormatter:
    """
    Format notification messages with emojis and Markdown.
    
    All methods return Markdown-formatted strings ready for Telegram.
    """
    
    # Emoji constants
    BUY_EMOJI = "🟢"
    SELL_EMOJI = "🔴"
    PROFIT_EMOJI = "💰"
    LOSS_EMOJI = "💸"
    WARNING_EMOJI = "⚠️"
    SIGNAL_EMOJI = "📊"
    START_EMOJI = "🚀"
    STOP_EMOJI = "🛑"
    RECONNECT_EMOJI = "🔄"
    CHART_EMOJI = "📈"
    CLOCK_EMOJI = "🕐"
    TARGET_EMOJI = "🎯"
    SHIELD_EMOJI = "🛡️"
    MONEY_EMOJI = "💵"
    
    @classmethod
    def format_trade_opened(
        cls,
        symbol: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        take_profit_1: float,
        take_profit_2: Optional[float] = None,
        lot_size: float = 0.0,
        confidence: float = 0.0,
        reason: str = ""
    ) -> str:
        """
        Format trade opened notification.
        
        Args:
            symbol: Trading symbol
            direction: BUY or SELL
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit_1: First take profit target
            take_profit_2: Second take profit target (optional)
            lot_size: Position size in lots
            confidence: Signal confidence (0-100)
            reason: Trade reason/setup
            
        Returns:
            Formatted Markdown message
        """
        emoji = cls.BUY_EMOJI if direction.upper() == "BUY" else cls.SELL_EMOJI
        
        # Calculate risk in pips (approximate for display)
        risk_pips = abs(entry_price - stop_loss)
        reward_pips = abs(take_profit_1 - entry_price)
        rr_ratio = reward_pips / risk_pips if risk_pips > 0 else 0
        
        message = f"""
{emoji} *{direction.upper()} {symbol}*

{cls.MONEY_EMOJI} *Entry:* `{entry_price:.5g}`
{cls.SHIELD_EMOJI} *SL:* `{stop_loss:.5g}`
{cls.TARGET_EMOJI} *TP1:* `{take_profit_1:.5g}`"""
        
        if take_profit_2:
            message += f"\n{cls.TARGET_EMOJI} *TP2:* `{take_profit_2:.5g}`"
        
        message += f"""

📦 *Size:* `{lot_size:.2f}` lots
📊 *R:R:* `1:{rr_ratio:.1f}`"""
        
        if confidence > 0:
            message += f"\n🎯 *Confidence:* `{confidence:.0f}%`"
        
        if reason:
            message += f"\n\n💡 _{reason}_"
        
        message += f"\n\n{cls.CLOCK_EMOJI} _{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC_"
        
        return message.strip()
    
    @classmethod
    def format_trade_closed(
        cls,
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
    ) -> str:
        """
        Format trade closed notification.
        
        Args:
            symbol: Trading symbol
            direction: BUY or SELL
            entry_price: Entry price
            exit_price: Exit price
            profit_loss: Profit/loss amount
            lot_size: Position size closed
            duration_minutes: Trade duration in minutes
            exit_reason: Reason for exit (TP1, TP2, SL, etc.)
            is_partial: Whether this is a partial close
            partial_percent: Percentage closed if partial
            
        Returns:
            Formatted Markdown message
        """
        is_profit = profit_loss >= 0
        pnl_emoji = cls.PROFIT_EMOJI if is_profit else cls.LOSS_EMOJI
        direction_emoji = cls.BUY_EMOJI if direction.upper() == "BUY" else cls.SELL_EMOJI
        
        close_type = "PARTIAL CLOSE" if is_partial else "CLOSED"
        pnl_sign = "+" if is_profit else ""
        
        # Format duration
        if duration_minutes >= 60:
            hours = duration_minutes // 60
            mins = duration_minutes % 60
            duration_str = f"{hours}h {mins}m"
        else:
            duration_str = f"{duration_minutes}m"
        
        message = f"""
{pnl_emoji} *{close_type}* {direction_emoji} {symbol}

💵 *P/L:* `{pnl_sign}${profit_loss:.2f}`
📍 *Entry:* `{entry_price:.5g}`
📍 *Exit:* `{exit_price:.5g}`"""
        
        if is_partial:
            message += f"\n📦 *Closed:* `{partial_percent}%` ({lot_size:.2f} lots)"
        else:
            message += f"\n📦 *Size:* `{lot_size:.2f}` lots"
        
        message += f"""
{cls.CLOCK_EMOJI} *Duration:* `{duration_str}`"""
        
        if exit_reason:
            message += f"\n🏷️ *Reason:* `{exit_reason}`"
        
        message += f"\n\n_{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC_"
        
        return message.strip()
    
    @classmethod
    def format_signal_generated(
        cls,
        symbol: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        confidence: float,
        timeframe: str = "15M",
        reason: str = "",
        was_executed: bool = False
    ) -> str:
        """
        Format signal generated notification.
        
        Args:
            symbol: Trading symbol
            direction: BUY or SELL
            entry_price: Proposed entry price
            stop_loss: Proposed stop loss
            take_profit: Proposed take profit
            confidence: Signal confidence (0-100)
            timeframe: Timeframe of the signal
            reason: Signal reason/setup
            was_executed: Whether the signal was executed
            
        Returns:
            Formatted Markdown message
        """
        emoji = cls.BUY_EMOJI if direction.upper() == "BUY" else cls.SELL_EMOJI
        status = "✅ EXECUTED" if was_executed else "📋 GENERATED"
        
        message = f"""
{cls.SIGNAL_EMOJI} *SIGNAL {status}*

{emoji} *{direction.upper()} {symbol}* ({timeframe})

📍 *Entry:* `{entry_price:.5g}`
{cls.SHIELD_EMOJI} *SL:* `{stop_loss:.5g}`
{cls.TARGET_EMOJI} *TP:* `{take_profit:.5g}`
🎯 *Confidence:* `{confidence:.0f}%`"""
        
        if reason:
            message += f"\n\n💡 _{reason}_"
        
        message += f"\n\n{cls.CLOCK_EMOJI} _{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC_"
        
        return message.strip()
    
    @classmethod
    def format_risk_warning(
        cls,
        warning_type: str,
        current_value: float,
        limit_value: float,
        symbol: Optional[str] = None,
        additional_info: str = ""
    ) -> str:
        """
        Format risk warning notification.
        
        Args:
            warning_type: Type of warning (drawdown, daily_loss, consecutive_losses)
            current_value: Current value that triggered warning
            limit_value: The limit that was approached/exceeded
            symbol: Related symbol if applicable
            additional_info: Additional context
            
        Returns:
            Formatted Markdown message
        """
        warning_titles = {
            "drawdown": "DRAWDOWN WARNING",
            "daily_loss": "DAILY LOSS LIMIT",
            "consecutive_losses": "CONSECUTIVE LOSSES",
            "max_positions": "MAX POSITIONS",
            "margin_warning": "LOW MARGIN"
        }
        
        title = warning_titles.get(warning_type, warning_type.upper())
        
        message = f"""
{cls.WARNING_EMOJI} *{title}*

📊 *Current:* `{current_value:.2f}%`
🚫 *Limit:* `{limit_value:.2f}%`"""
        
        if symbol:
            message += f"\n📈 *Symbol:* `{symbol}`"
        
        if additional_info:
            message += f"\n\n_{additional_info}_"
        
        message += f"\n\n{cls.CLOCK_EMOJI} _{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC_"
        
        return message.strip()
    
    @classmethod
    def format_system_status(
        cls,
        status: str,
        details: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Format system status notification.
        
        Args:
            status: Status type (started, stopped, reconnected, error)
            details: Additional details (balance, version, etc.)
            
        Returns:
            Formatted Markdown message
        """
        status_config = {
            "started": (cls.START_EMOJI, "SYSTEM STARTED", "Trading system is now online"),
            "stopped": (cls.STOP_EMOJI, "SYSTEM STOPPED", "Trading system has been shut down"),
            "reconnected": (cls.RECONNECT_EMOJI, "RECONNECTED", "Connection restored"),
            "disconnected": (cls.WARNING_EMOJI, "DISCONNECTED", "Connection lost"),
            "paused": (cls.WARNING_EMOJI, "TRADING PAUSED", "Trading has been paused"),
            "resumed": (cls.START_EMOJI, "TRADING RESUMED", "Trading has been resumed")
        }
        
        emoji, title, description = status_config.get(
            status.lower(), 
            (cls.SIGNAL_EMOJI, status.upper(), "")
        )
        
        message = f"""
{emoji} *{title}*

_{description}_"""
        
        if details:
            message += "\n"
            if "version" in details:
                message += f"\n📋 *Version:* `{details['version']}`"
            if "balance" in details:
                message += f"\n💵 *Balance:* `${details['balance']:.2f}`"
            if "equity" in details:
                message += f"\n💰 *Equity:* `${details['equity']:.2f}`"
            if "broker" in details:
                message += f"\n🏦 *Broker:* `{details['broker']}`"
            if "symbol" in details:
                message += f"\n📈 *Symbol:* `{details['symbol']}`"
            if "reason" in details:
                message += f"\n📝 *Reason:* `{details['reason']}`"
        
        message += f"\n\n{cls.CLOCK_EMOJI} _{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC_"
        
        return message.strip()
    
    @classmethod
    def format_daily_summary(
        cls,
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
    ) -> str:
        """
        Format daily summary notification.
        
        Args:
            date: Summary date (YYYY-MM-DD)
            total_trades: Total number of trades
            winning_trades: Number of winning trades
            losing_trades: Number of losing trades
            gross_profit: Total profit from winning trades
            gross_loss: Total loss from losing trades
            net_pnl: Net profit/loss
            win_rate: Win rate percentage
            starting_balance: Balance at start of day
            ending_balance: Balance at end of day
            
        Returns:
            Formatted Markdown message
        """
        is_profit = net_pnl >= 0
        pnl_emoji = cls.PROFIT_EMOJI if is_profit else cls.LOSS_EMOJI
        pnl_sign = "+" if is_profit else ""
        
        balance_change = ending_balance - starting_balance
        balance_change_pct = (balance_change / starting_balance * 100) if starting_balance > 0 else 0
        balance_sign = "+" if balance_change >= 0 else ""
        
        message = f"""
{cls.CHART_EMOJI} *DAILY SUMMARY* - {date}

{pnl_emoji} *Net P/L:* `{pnl_sign}${net_pnl:.2f}`

📊 *Performance:*
├ Trades: `{total_trades}`
├ Winners: `{winning_trades}` | Losers: `{losing_trades}`
├ Win Rate: `{win_rate:.1f}%`
├ Gross Profit: `+${gross_profit:.2f}`
└ Gross Loss: `-${abs(gross_loss):.2f}`

💵 *Account:*
├ Starting: `${starting_balance:.2f}`
├ Ending: `${ending_balance:.2f}`
└ Change: `{balance_sign}${balance_change:.2f}` ({balance_sign}{balance_change_pct:.2f}%)"""
        
        message += f"\n\n{cls.CLOCK_EMOJI} _{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC_"
        
        return message.strip()
    
    @classmethod
    def format_signal_rejected(
        cls,
        symbol: str,
        direction: str,
        reason: str
    ) -> str:
        """
        Format signal rejected notification.
        
        Args:
            symbol: Trading symbol
            direction: BUY or SELL
            reason: Rejection reason
            
        Returns:
            Formatted Markdown message
        """
        emoji = cls.BUY_EMOJI if direction.upper() == "BUY" else cls.SELL_EMOJI
        
        message = f"""
🚫 *SIGNAL REJECTED*

{emoji} {direction.upper()} {symbol}
📝 *Reason:* _{reason}_

{cls.CLOCK_EMOJI} _{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC_"""
        
        return message.strip()
