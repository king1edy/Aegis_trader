"""
Integration Tests — Position Sizer
====================================
Tests for: src/risk/position_sizer.py

Integration because PositionSizer depends on Settings + SymbolInfo working
together. All broker I/O is avoided (we pass SymbolInfo directly).
"""

import pytest

from src.risk.position_sizer import PositionSizer

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Lot‑Size Calculation
# ---------------------------------------------------------------------------

class TestCalculateLotSize:
    """Tests for PositionSizer.calculate_lot_size"""

    @pytest.mark.asyncio
    async def test_basic_lot_calculation(self, mock_settings, mock_symbol_info):
        """1% risk on $10k with 100-pip SL should yield a reasonable lot."""
        sizer = PositionSizer(mock_settings)
        lot = await sizer.calculate_lot_size(
            symbol="XAUUSD",
            entry_price=2050.0,
            stop_loss=2040.0,      # 10 USD = 1000 pips at point=0.01
            account_balance=10000.0,
            risk_percent=0.01,     # 1% → $100 risk
            symbol_info=mock_symbol_info,
        )
        assert lot > 0
        assert lot <= mock_symbol_info.max_lot

    @pytest.mark.asyncio
    async def test_rejects_zero_balance(self, mock_settings, mock_symbol_info):
        sizer = PositionSizer(mock_settings)
        with pytest.raises(ValueError, match="Invalid account balance"):
            await sizer.calculate_lot_size(
                symbol="XAUUSD",
                entry_price=2050.0,
                stop_loss=2040.0,
                account_balance=0,
                symbol_info=mock_symbol_info,
            )

    # TODO: test_rejects_equal_entry_and_sl (ValueError)
    # TODO: test_lot_clamped_to_min_lot — very small account → min_lot returned
    # TODO: test_lot_clamped_to_max_lot — huge account → max_lot capped


# ---------------------------------------------------------------------------
# Lot Validation
# ---------------------------------------------------------------------------

class TestValidateLotSize:
    """Tests for PositionSizer.validate_lot_size"""

    @pytest.mark.asyncio
    async def test_valid_lot(self, mock_settings, mock_symbol_info):
        sizer = PositionSizer(mock_settings)
        is_valid, reason = await sizer.validate_lot_size(0.05, mock_symbol_info)
        assert is_valid is True

    # TODO: test_below_min_lot_invalid
    # TODO: test_above_max_lot_invalid
    # TODO: test_wrong_step_invalid (e.g. 0.015 when step is 0.01)


# ---------------------------------------------------------------------------
# Risk Stats
# ---------------------------------------------------------------------------

class TestGetRiskStats:
    """Tests for PositionSizer.get_risk_stats"""

    def test_returns_expected_keys(self, mock_settings):
        sizer = PositionSizer(mock_settings)
        stats = sizer.get_risk_stats(account_balance=10000.0)
        assert "risk_amount_per_trade" in stats
        assert stats["risk_amount_per_trade"] == pytest.approx(100.0)

    # TODO: test_max_consecutive_losses_calculation
    # TODO: test_balance_after_10_losses