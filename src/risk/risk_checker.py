"""
Risk Checker

Validates trading signals against all risk limits before execution.
Enforces:
- Max daily trades
- Max open positions
- Daily loss limits
- Drawdown limits
- Consecutive loss pauses
- Trading pause status
"""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Tuple, Optional

from src.strategies.base_strategy import TradingSignal
from src.execution.mt5_connector import AccountInfo
from src.database.repository import TradeRepository, PerformanceRepository, SystemRepository
from src.core.config import Settings
from src.core.logging_config import get_logger

logger = get_logger("risk_checker")


class RiskChecker:
    """
    Validate signals against all risk management rules.

    Checks performed:
    1. Max trades per day limit
    2. Max open positions limit
    3. Daily loss limit
    4. Max drawdown limit
    5. Consecutive losses (triggers pause)
    6. Active trading pause
    7. Sufficient margin

    Returns (allowed, rejection_reason) tuple.
    """

    def __init__(
        self,
        settings: Settings,
        trade_repo: TradeRepository,
        perf_repo: PerformanceRepository,
        system_repo: SystemRepository
    ):
        """
        Initialize risk checker.

        Args:
            settings: Application settings
            trade_repo: Trade repository
            perf_repo: Performance repository
            system_repo: System events repository
        """
        self.settings = settings
        self.trade_repo = trade_repo
        self.perf_repo = perf_repo
        self.system_repo = system_repo

        logger.info(
            "Risk checker initialized",
            max_daily_trades=settings.max_daily_trades,
            max_open_positions=settings.max_open_positions,
            max_daily_loss_percent=settings.max_daily_loss_percent,
            max_drawdown_percent=settings.max_drawdown_percent,
            max_consecutive_losses=settings.max_consecutive_losses
        )

    async def check_all_limits(
        self,
        signal: TradingSignal,
        account_info: AccountInfo
    ) -> Tuple[bool, Optional[str]]:
        """
        Run all risk checks on a signal.

        Args:
            signal: Trading signal to validate
            account_info: Current account information

        Returns:
            Tuple of (allowed, rejection_reason)
            - (True, None) if all checks pass
            - (False, reason) if any check fails
        """
        # 1. Check for active trading pause
        check_passed, reason = await self._check_trading_pause()
        if not check_passed:
            return False, reason

        # 2. Check max trades per day
        check_passed, reason = await self._check_daily_trade_limit()
        if not check_passed:
            return False, reason

        # 3. Check max open positions
        check_passed, reason = await self._check_open_positions_limit(signal.symbol)
        if not check_passed:
            return False, reason

        # 4. Check daily loss limit
        check_passed, reason = await self._check_daily_loss_limit(account_info)
        if not check_passed:
            return False, reason

        # 5. Check max drawdown
        check_passed, reason = await self._check_drawdown_limit(account_info)
        if not check_passed:
            return False, reason

        # 6. Check consecutive losses (may trigger pause)
        check_passed, reason = await self._check_consecutive_losses()
        if not check_passed:
            return False, reason

        # 7. Check margin availability
        check_passed, reason = await self._check_margin_availability(account_info)
        if not check_passed:
            return False, reason

        # All checks passed
        logger.info(
            "All risk checks passed",
            symbol=signal.symbol,
            direction=signal.direction.value,
            confidence=signal.confidence
        )
        return True, None

    async def _check_trading_pause(self) -> Tuple[bool, Optional[str]]:
        """
        Check if trading is currently paused.

        Returns:
            (allowed, rejection_reason)
        """
        active_pause = await self.system_repo.get_active_pause()

        if active_pause:
            logger.warning(
                "Trading is paused",
                reason=active_pause.reason,
                start_time=active_pause.start_time
            )
            return False, f"Trading paused: {active_pause.reason}"

        return True, None

    async def _check_daily_trade_limit(self) -> Tuple[bool, Optional[str]]:
        """
        Check if daily trade limit has been reached.

        Returns:
            (allowed, rejection_reason)
        """
        today_count = await self.trade_repo.count_today_trades()

        if today_count >= self.settings.max_daily_trades:
            logger.warning(
                "Daily trade limit reached",
                today_count=today_count,
                max_daily_trades=self.settings.max_daily_trades
            )
            return False, f"Daily trade limit reached: {today_count}/{self.settings.max_daily_trades}"

        logger.debug(
            "Daily trade check passed",
            today_count=today_count,
            limit=self.settings.max_daily_trades
        )
        return True, None

    async def _check_open_positions_limit(
        self,
        symbol: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if max open positions limit has been reached.

        Args:
            symbol: Trading symbol

        Returns:
            (allowed, rejection_reason)
        """
        open_trades = await self.trade_repo.get_open_trades(symbol=symbol)
        open_count = len(open_trades)

        if open_count >= self.settings.max_open_positions:
            logger.warning(
                "Max open positions limit reached",
                open_count=open_count,
                max_open_positions=self.settings.max_open_positions,
                symbol=symbol
            )
            return False, f"Max open positions reached: {open_count}/{self.settings.max_open_positions}"

        logger.debug(
            "Open positions check passed",
            open_count=open_count,
            limit=self.settings.max_open_positions
        )
        return True, None

    async def _check_daily_loss_limit(
        self,
        account_info: AccountInfo
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if daily loss limit has been exceeded.

        Args:
            account_info: Account information

        Returns:
            (allowed, rejection_reason)
        """
        daily_perf = await self.perf_repo.get_daily_performance()

        if daily_perf:
            balance = float(account_info.balance)
            daily_pnl = float(daily_perf.total_pnl)
            daily_loss_percent = (abs(daily_pnl) / balance * 100) if balance > 0 else 0

            max_loss_percent = self.settings.max_daily_loss_percent

            if daily_pnl < 0 and daily_loss_percent >= max_loss_percent:
                logger.warning(
                    "Daily loss limit reached",
                    daily_pnl=daily_pnl,
                    daily_loss_percent=daily_loss_percent,
                    max_loss_percent=max_loss_percent
                )

                # Trigger automatic pause
                await self.system_repo.start_trading_pause(
                    reason="daily_loss_limit",
                    trigger_value=Decimal(str(daily_loss_percent)),
                    threshold_value=Decimal(str(max_loss_percent)),
                    was_automatic=True,
                    notes=f"Daily loss: ${daily_pnl:.2f} ({daily_loss_percent:.2f}%)"
                )

                return False, f"Daily loss limit exceeded: {daily_loss_percent:.2f}% (limit: {max_loss_percent}%)"

        logger.debug("Daily loss check passed")
        return True, None

    async def _check_drawdown_limit(
        self,
        account_info: AccountInfo
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if max drawdown limit has been exceeded.

        Args:
            account_info: Account information

        Returns:
            (allowed, rejection_reason)
        """
        drawdown_amount, drawdown_percent = await self.perf_repo.calculate_current_drawdown()

        max_dd_percent = self.settings.max_drawdown_percent

        if drawdown_percent >= Decimal(str(max_dd_percent)):
            logger.warning(
                "Max drawdown limit reached",
                drawdown_percent=float(drawdown_percent),
                max_drawdown_percent=max_dd_percent
            )

            # Trigger automatic pause
            await self.system_repo.start_trading_pause(
                reason="max_drawdown",
                trigger_value=drawdown_percent,
                threshold_value=Decimal(str(max_dd_percent)),
                was_automatic=True,
                notes=f"Drawdown: ${float(drawdown_amount):.2f} ({float(drawdown_percent):.2f}%)"
            )

            return False, f"Max drawdown exceeded: {float(drawdown_percent):.2f}% (limit: {max_dd_percent}%)"

        logger.debug(
            "Drawdown check passed",
            current_drawdown_percent=float(drawdown_percent)
        )
        return True, None

    async def _check_consecutive_losses(self) -> Tuple[bool, Optional[str]]:
        """
        Check consecutive losses and trigger pause if threshold reached.

        Returns:
            (allowed, rejection_reason)
        """
        consecutive_losses = await self.trade_repo.get_consecutive_losses()

        max_consecutive = self.settings.max_consecutive_losses

        if consecutive_losses >= max_consecutive:
            logger.warning(
                "Max consecutive losses reached",
                consecutive_losses=consecutive_losses,
                max_consecutive_losses=max_consecutive
            )

            # Trigger automatic pause
            await self.system_repo.start_trading_pause(
                reason="consecutive_losses",
                trigger_value=Decimal(str(consecutive_losses)),
                threshold_value=Decimal(str(max_consecutive)),
                was_automatic=True,
                notes=f"{consecutive_losses} consecutive losses"
            )

            return False, f"Max consecutive losses reached: {consecutive_losses} (limit: {max_consecutive})"

        logger.debug(
            "Consecutive losses check passed",
            consecutive_losses=consecutive_losses,
            limit=max_consecutive
        )
        return True, None

    async def _check_margin_availability(
        self,
        account_info: AccountInfo
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if sufficient margin is available.

        Args:
            account_info: Account information

        Returns:
            (allowed, rejection_reason)
        """
        # Check margin level (percentage)
        if account_info.margin_level is not None:
            min_margin_level = self.settings.min_margin_level

            if account_info.margin_level < min_margin_level:
                logger.warning(
                    "Insufficient margin level",
                    margin_level=account_info.margin_level,
                    min_margin_level=min_margin_level
                )
                return False, f"Insufficient margin: {account_info.margin_level:.0f}% (min: {min_margin_level}%)"

        # Check free margin (absolute value)
        free_margin = float(account_info.free_margin)
        if free_margin < 100:  # Minimum $100 free margin
            logger.warning(
                "Low free margin",
                free_margin=free_margin
            )
            return False, f"Insufficient free margin: ${free_margin:.2f}"

        logger.debug(
            "Margin check passed",
            margin_level=account_info.margin_level,
            free_margin=free_margin
        )
        return True, None

    async def get_risk_status(self, account_info: AccountInfo) -> dict:
        """
        Get current risk status summary.

        Args:
            account_info: Account information

        Returns:
            Dictionary with risk status details
        """
        today_count = await self.trade_repo.count_today_trades()
        open_trades = await self.trade_repo.get_open_trades()
        consecutive_losses = await self.trade_repo.get_consecutive_losses()
        active_pause = await self.system_repo.get_active_pause()
        drawdown_amount, drawdown_percent = await self.perf_repo.calculate_current_drawdown()
        daily_perf = await self.perf_repo.get_daily_performance()

        daily_pnl = float(daily_perf.total_pnl) if daily_perf else 0.0
        balance = float(account_info.balance)
        daily_loss_percent = (abs(daily_pnl) / balance * 100) if balance > 0 and daily_pnl < 0 else 0.0

        return {
            "is_paused": active_pause is not None,
            "pause_reason": active_pause.reason if active_pause else None,
            "today_trades": today_count,
            "max_daily_trades": self.settings.max_daily_trades,
            "open_positions": len(open_trades),
            "max_open_positions": self.settings.max_open_positions,
            "consecutive_losses": consecutive_losses,
            "max_consecutive_losses": self.settings.max_consecutive_losses,
            "daily_pnl": daily_pnl,
            "daily_loss_percent": daily_loss_percent,
            "max_daily_loss_percent": self.settings.max_daily_loss_percent,
            "drawdown_amount": float(drawdown_amount),
            "drawdown_percent": float(drawdown_percent),
            "max_drawdown_percent": self.settings.max_drawdown_percent,
            "margin_level": account_info.margin_level,
            "min_margin_level": self.settings.min_margin_level,
            "free_margin": float(account_info.free_margin),
            "can_trade": active_pause is None
        }
