"""
Position Sizer

Calculates lot sizes based on fixed percentage risk per trade.
Ensures proper risk management by limiting capital at risk.
"""

from typing import Optional
from dataclasses import dataclass

from src.execution.mt5_connector import SymbolInfo
from src.core.logging_config import get_logger
from src.core.config import Settings

logger = get_logger("position_sizer")


@dataclass
class PositionSizeResult:
    """Result of position size calculation"""
    lot_size: float
    risk_amount: float
    sl_distance_pips: float
    risk_percent_actual: float
    normalized_lot: float


class PositionSizer:
    """
    Calculate position sizes based on risk percentage.

    Formula:
    risk_amount = account_balance × risk_percent
    sl_distance_pips = abs(entry - stop_loss) / point
    lot_size = risk_amount / (sl_distance_pips × tick_value)

    Then normalize to broker's min/max/step.
    """

    def __init__(self, settings: Settings):
        """
        Initialize position sizer.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self.default_risk_percent = settings.max_risk_per_trade

        logger.info(
            "Position sizer initialized",
            default_risk_percent=self.default_risk_percent
        )

    async def calculate_lot_size(
        self,
        symbol: str,
        entry_price: float,
        stop_loss: float,
        account_balance: float,
        risk_percent: Optional[float] = None,
        symbol_info: Optional[SymbolInfo] = None
    ) -> float:
        """
        Calculate lot size for specified risk.

        Args:
            symbol: Trading symbol
            entry_price: Entry price
            stop_loss: Stop loss price
            account_balance: Account balance in account currency
            risk_percent: Risk percentage (default: from settings)
            symbol_info: Symbol information (required)

        Returns:
            Normalized lot size

        Raises:
            ValueError: If inputs are invalid
        """
        if symbol_info is None:
            raise ValueError("symbol_info is required")

        if risk_percent is None:
            risk_percent = self.default_risk_percent

        # Validate inputs
        if account_balance <= 0:
            raise ValueError(f"Invalid account balance: {account_balance}")

        if entry_price <= 0 or stop_loss <= 0:
            raise ValueError(f"Invalid prices: entry={entry_price}, sl={stop_loss}")

        if stop_loss == entry_price:
            raise ValueError("Stop loss cannot equal entry price")

        if risk_percent <= 0 or risk_percent > 0.1:  # Max 10%
            raise ValueError(f"Invalid risk percent: {risk_percent}")

        # Calculate risk amount
        risk_amount = account_balance * risk_percent

        # Calculate SL distance in pips
        sl_distance = abs(entry_price - stop_loss)
        sl_distance_pips = sl_distance / symbol_info.point

        # Calculate lot size
        # risk_amount = lot_size × pip_risk × tick_value
        # lot_size = risk_amount / (pip_risk × tick_value)
        if symbol_info.tick_value == 0:
            raise ValueError(f"Invalid tick_value for {symbol}: {symbol_info.tick_value}")

        raw_lot_size = risk_amount / (sl_distance_pips * symbol_info.tick_value)

        # Normalize to broker's lot step
        normalized_lot = symbol_info.normalize_lot(raw_lot_size)

        # Check min/max limits
        if normalized_lot < symbol_info.min_lot:
            logger.warning(
                "Lot size below minimum",
                calculated=normalized_lot,
                min_lot=symbol_info.min_lot,
                symbol=symbol
            )
            normalized_lot = symbol_info.min_lot

        if normalized_lot > symbol_info.max_lot:
            logger.warning(
                "Lot size above maximum",
                calculated=normalized_lot,
                max_lot=symbol_info.max_lot,
                symbol=symbol
            )
            normalized_lot = symbol_info.max_lot

        # Calculate actual risk with normalized lot
        actual_risk = normalized_lot * sl_distance_pips * symbol_info.tick_value
        actual_risk_percent = actual_risk / account_balance

        logger.info(
            "Position size calculated",
            symbol=symbol,
            entry=entry_price,
            sl=stop_loss,
            account_balance=account_balance,
            target_risk_percent=risk_percent * 100,
            target_risk_amount=risk_amount,
            raw_lot=raw_lot_size,
            normalized_lot=normalized_lot,
            actual_risk_amount=actual_risk,
            actual_risk_percent=actual_risk_percent * 100,
            sl_distance_pips=sl_distance_pips
        )

        return normalized_lot

    async def calculate_detailed(
        self,
        symbol: str,
        entry_price: float,
        stop_loss: float,
        account_balance: float,
        risk_percent: Optional[float] = None,
        symbol_info: Optional[SymbolInfo] = None
    ) -> PositionSizeResult:
        """
        Calculate lot size with detailed result.

        Args:
            symbol: Trading symbol
            entry_price: Entry price
            stop_loss: Stop loss price
            account_balance: Account balance
            risk_percent: Risk percentage (optional)
            symbol_info: Symbol information (required)

        Returns:
            PositionSizeResult with all details
        """
        if symbol_info is None:
            raise ValueError("symbol_info is required")

        if risk_percent is None:
            risk_percent = self.default_risk_percent

        # Calculate
        normalized_lot = await self.calculate_lot_size(
            symbol=symbol,
            entry_price=entry_price,
            stop_loss=stop_loss,
            account_balance=account_balance,
            risk_percent=risk_percent,
            symbol_info=symbol_info
        )

        # Calculate details
        sl_distance = abs(entry_price - stop_loss)
        sl_distance_pips = sl_distance / symbol_info.point
        risk_amount = account_balance * risk_percent
        actual_risk = normalized_lot * sl_distance_pips * symbol_info.tick_value
        actual_risk_percent = actual_risk / account_balance

        return PositionSizeResult(
            lot_size=normalized_lot,
            risk_amount=risk_amount,
            sl_distance_pips=sl_distance_pips,
            risk_percent_actual=actual_risk_percent,
            normalized_lot=normalized_lot
        )

    async def calculate_for_fixed_lot(
        self,
        entry_price: float,
        stop_loss: float,
        lot_size: float,
        account_balance: float,
        symbol_info: SymbolInfo
    ) -> float:
        """
        Calculate risk percentage for a given lot size.

        Useful for validating or reverse-engineering lot sizes.

        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            lot_size: Lot size to check
            account_balance: Account balance
            symbol_info: Symbol information

        Returns:
            Risk percentage (e.g., 0.01 for 1%)
        """
        sl_distance = abs(entry_price - stop_loss)
        sl_distance_pips = sl_distance / symbol_info.point

        risk_amount = lot_size * sl_distance_pips * symbol_info.tick_value
        risk_percent = risk_amount / account_balance

        logger.debug(
            "Risk calculated for fixed lot",
            lot_size=lot_size,
            risk_amount=risk_amount,
            risk_percent=risk_percent * 100
        )

        return risk_percent

    async def validate_lot_size(
        self,
        lot_size: float,
        symbol_info: SymbolInfo
    ) -> tuple[bool, str]:
        """
        Validate that lot size meets broker requirements.

        Args:
            lot_size: Lot size to validate
            symbol_info: Symbol information

        Returns:
            Tuple of (is_valid, reason)
        """
        if lot_size < symbol_info.min_lot:
            return False, f"Below minimum lot: {lot_size} < {symbol_info.min_lot}"

        if lot_size > symbol_info.max_lot:
            return False, f"Above maximum lot: {lot_size} > {symbol_info.max_lot}"

        # Check if it's a valid step
        steps = round(lot_size / symbol_info.lot_step)
        expected = steps * symbol_info.lot_step

        if abs(lot_size - expected) > 0.001:  # Small tolerance for floating point
            return False, f"Invalid lot step: {lot_size} (step: {symbol_info.lot_step})"

        return True, "Valid"

    def get_risk_stats(
        self,
        account_balance: float,
        risk_percent: Optional[float] = None
    ) -> dict:
        """
        Get risk statistics for given account balance.

        Args:
            account_balance: Account balance
            risk_percent: Risk percentage (optional)

        Returns:
            Dictionary with risk statistics
        """
        if risk_percent is None:
            risk_percent = self.default_risk_percent

        risk_amount = account_balance * risk_percent

        return {
            "account_balance": account_balance,
            "risk_percent": risk_percent * 100,
            "risk_amount_per_trade": risk_amount,
            "max_consecutive_losses": int(1 / risk_percent) if risk_percent > 0 else 0,
            "balance_after_10_losses": account_balance * ((1 - risk_percent) ** 10)
        }
