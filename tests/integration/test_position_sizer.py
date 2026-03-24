"""
Integration Tests — Position Sizer
====================================
Tests for: src/risk/position_sizer.py

Integration because PositionSizer depends on Settings + SymbolInfo working
together. All broker I/O is avoided (we pass SymbolInfo directly).
"""

import pytest
from decimal import Decimal

from src.risk.position_sizer import PositionSizer
from src.execution.mt5_connector import SymbolInfo

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_settings():
    """Create mock settings for testing"""
    class MockSettings:
        def __init__(self):
            self.max_risk_per_trade = 0.01  # 1% default risk
            self.magic_number = 123456
    
    return MockSettings()


@pytest.fixture
def mock_symbol_info():
    """Create mock symbol info for XAUUSD"""
    class MockSymbolInfo:
        def __init__(self):
            self.name = "XAUUSD"
            self.point = 0.01  # 1 pip = 0.01 USD for gold
            self.tick_value = 0.1  # $0.10 per pip per 0.01 lot
            self.min_lot = 0.01
            self.max_lot = 100.0
            self.lot_step = 0.01
        
        def normalize_lot(self, lot):
            """Normalize lot to step"""
            steps = round(lot / self.lot_step)
            return steps * self.lot_step
    
    return MockSymbolInfo()


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

    # test_rejects_equal_entry_and_sl (ValueError)#

    @pytest.mark.asyncio
    async def test_rejects_equal_entry_and_sl(self, mock_settings, mock_symbol_info):
        """Test that equal entry and stop loss raises ValueError"""
        sizer = PositionSizer(mock_settings)
        with pytest.raises(ValueError, match="Stop loss cannot equal entry price"):
            await sizer.calculate_lot_size(
                symbol="XAUUSD",
                entry_price=2050.0,
                stop_loss=2050.0,  # Equal to entry price
                account_balance=10000.0,
                symbol_info=mock_symbol_info,
            )
    # test_lot_clamped_to_min_lot — very small account → min_lot returned#

    @pytest.mark.asyncio
    async def test_lot_clamped_to_min_lot(self, mock_settings, mock_symbol_info):
        """Test that very small account results in minimum lot size"""
        sizer = PositionSizer(mock_settings)
        
        # Very small account with large SL distance
        lot = await sizer.calculate_lot_size(
            symbol="XAUUSD",
            entry_price=2050.0,
            stop_loss=2000.0,      # Large SL distance
            account_balance=100.0,  # Very small balance
            risk_percent=0.01,      # 1% risk = $1
            symbol_info=mock_symbol_info,
        )
        
        # Should be clamped to minimum lot
        assert lot == mock_symbol_info.min_lot
        assert lot == 0.01
    # test_lot_clamped_to_max_lot — huge account → max_lot capped#

    @pytest.mark.asyncio
    async def test_lot_clamped_to_max_lot(self, mock_settings, mock_symbol_info):
        """Test that huge account results in maximum lot size"""
        sizer = PositionSizer(mock_settings)
        
        # Very large account with tiny SL distance
        lot = await sizer.calculate_lot_size(
            symbol="XAUUSD",
            entry_price=2050.0,
            stop_loss=2049.5,      # Very small SL distance
            account_balance=1000000.0,  # $1M balance
            risk_percent=0.01,      # 1% risk = $10,000
            symbol_info=mock_symbol_info,
        )
        
        # Should be clamped to maximum lot
        assert lot == mock_symbol_info.max_lot
        assert lot == 100.0


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
        assert reason == "Valid"

    # test_below_min_lot_invalid#

    @pytest.mark.asyncio
    async def test_below_min_lot_invalid(self, mock_settings, mock_symbol_info):
        """Test that lot size below minimum is invalid"""
        sizer = PositionSizer(mock_settings)
        is_valid, reason = await sizer.validate_lot_size(0.005, mock_symbol_info)
        assert is_valid is False
        assert "Below minimum lot" in reason

    # test_above_max_lot_invalid#

    @pytest.mark.asyncio
    async def test_above_max_lot_invalid(self, mock_settings, mock_symbol_info):
        """Test that lot size above maximum is invalid"""
        sizer = PositionSizer(mock_settings)
        is_valid, reason = await sizer.validate_lot_size(150.0, mock_symbol_info)
        assert is_valid is False
        assert "Above maximum lot" in reason

    # test_wrong_step_invalid (e.g. 0.015 when step is 0.01)#

    @pytest.mark.asyncio
    async def test_wrong_step_invalid(self, mock_settings, mock_symbol_info):
        """Test that lot size not on step is invalid (e.g. 0.015 when step is 0.01)"""
        sizer = PositionSizer(mock_settings)
        is_valid, reason = await sizer.validate_lot_size(0.015, mock_symbol_info)
        assert is_valid is False
        assert "Invalid lot step" in reason


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

    # test_max_consecutive_losses_calculation#

    def test_max_consecutive_losses_calculation(self, mock_settings):
        """Test that max_consecutive_losses is calculated correctly"""
        sizer = PositionSizer(mock_settings)
        
        # With 1% risk per trade, can lose 100 trades before zero
        stats = sizer.get_risk_stats(account_balance=10000.0, risk_percent=0.01)
        assert stats["max_consecutive_losses"] == 100
        
        # With 2% risk per trade, can lose 50 trades before zero
        stats = sizer.get_risk_stats(account_balance=10000.0, risk_percent=0.02)
        assert stats["max_consecutive_losses"] == 50
        
        # With 5% risk per trade, can lose 20 trades before zero
        stats = sizer.get_risk_stats(account_balance=10000.0, risk_percent=0.05)
        assert stats["max_consecutive_losses"] == 20


    # test_balance_after_10_losses#

    def test_balance_after_10_losses(self, mock_settings):
        """Test that balance after 10 consecutive losses is calculated correctly"""
        sizer = PositionSizer(mock_settings)
        
        initial_balance = 10000.0
        risk_percent = 0.01  # 1% risk per trade
        
        stats = sizer.get_risk_stats(
            account_balance=initial_balance,
            risk_percent=risk_percent
        )
        
        # Expected: balance * (1 - risk)^10
        expected = initial_balance * ((1 - risk_percent) ** 10)
        assert stats["balance_after_10_losses"] == pytest.approx(expected)
        
        # Verify with manual calculation
        balance = initial_balance
        for _ in range(10):
            risk_amount = balance * risk_percent
            balance -= risk_amount
        
        assert stats["balance_after_10_losses"] == pytest.approx(balance)

    def test_custom_risk_percent(self, mock_settings):
        """Test that custom risk percent overrides default"""
        sizer = PositionSizer(mock_settings)
        
        # Default is 1%
        stats_default = sizer.get_risk_stats(account_balance=10000.0)
        assert stats_default["risk_amount_per_trade"] == 100.0
        
        # Custom 2% risk
        stats_custom = sizer.get_risk_stats(account_balance=10000.0, risk_percent=0.02)
        assert stats_custom["risk_amount_per_trade"] == 200.0

# ---------------------------------------------------------------------------
# Additional Tests (Comprehensive Coverage)
# ---------------------------------------------------------------------------
# I added these extra tests to ensure complete coverage of the PositionSizer class.
# The class has multiple public methods (calculate_detailed, calculate_for_fixed_lot)
# that should be verified for correctness. Also added edge case tests to ensure
# the code handles invalid inputs gracefully. This provides more robust test
# coverage beyond the initial requirements.

class TestCalculateDetailed:
    """Tests for calculate_detailed method"""

    @pytest.mark.asyncio
    async def test_calculate_detailed_returns_full_result(self, mock_settings, mock_symbol_info):
        """Test that calculate_detailed returns complete PositionSizeResult"""
        sizer = PositionSizer(mock_settings)
        
        result = await sizer.calculate_detailed(
            symbol="XAUUSD",
            entry_price=2050.0,
            stop_loss=2040.0,
            account_balance=10000.0,
            risk_percent=0.01,
            symbol_info=mock_symbol_info
        )
        
        assert result.lot_size > 0
        assert result.risk_amount == 100.0
        assert result.sl_distance_pips > 0
        assert 0 < result.risk_percent_actual <= 0.01
        assert result.normalized_lot == result.lot_size

    @pytest.mark.asyncio
    async def test_calculate_detailed_without_risk_percent(self, mock_settings, mock_symbol_info):
        """Test that calculate_detailed uses default risk percent"""
        sizer = PositionSizer(mock_settings)
        
        result = await sizer.calculate_detailed(
            symbol="XAUUSD",
            entry_price=2050.0,
            stop_loss=2040.0,
            account_balance=10000.0,
            symbol_info=mock_symbol_info
        )
        
        # Should use default 1% risk
        assert result.risk_amount == 100.0


class TestCalculateForFixedLot:
    """Tests for calculate_for_fixed_lot method"""

    @pytest.mark.asyncio
    async def test_calculate_risk_for_fixed_lot(self, mock_settings, mock_symbol_info):
        """Test that risk percentage is correctly calculated for a fixed lot"""
        sizer = PositionSizer(mock_settings)
        
        risk_percent = await sizer.calculate_for_fixed_lot(
            entry_price=2050.0,
            stop_loss=2040.0,
            lot_size=0.1,
            account_balance=10000.0,
            symbol_info=mock_symbol_info
        )
        
        # Expected risk = lot * pips * tick_value / balance
        # 0.1 lot * 1000 pips * 0.1 = $10 risk
        # $10 / $10000 = 0.001 (0.1%)
        assert risk_percent == pytest.approx(0.001)


class TestEdgeCases:
    """Test edge cases and error conditions"""

    @pytest.mark.asyncio
    async def test_rejects_negative_balance(self, mock_settings, mock_symbol_info):
        """Test that negative balance raises ValueError"""
        sizer = PositionSizer(mock_settings)
        with pytest.raises(ValueError, match="Invalid account balance"):
            await sizer.calculate_lot_size(
                symbol="XAUUSD",
                entry_price=2050.0,
                stop_loss=2040.0,
                account_balance=-1000.0,
                symbol_info=mock_symbol_info,
            )

    @pytest.mark.asyncio
    async def test_rejects_invalid_risk_percent(self, mock_settings, mock_symbol_info):
        """Test that invalid risk percentage raises ValueError"""
        sizer = PositionSizer(mock_settings)
        
        # Risk > 10%
        with pytest.raises(ValueError, match="Invalid risk percent"):
            await sizer.calculate_lot_size(
                symbol="XAUUSD",
                entry_price=2050.0,
                stop_loss=2040.0,
                account_balance=10000.0,
                risk_percent=0.15,  # 15% risk
                symbol_info=mock_symbol_info,
            )
        
        # Risk <= 0
        with pytest.raises(ValueError, match="Invalid risk percent"):
            await sizer.calculate_lot_size(
                symbol="XAUUSD",
                entry_price=2050.0,
                stop_loss=2040.0,
                account_balance=10000.0,
                risk_percent=0.0,
                symbol_info=mock_symbol_info,
            )

    @pytest.mark.asyncio
    async def test_rejects_missing_symbol_info(self, mock_settings):
        """Test that missing symbol_info raises ValueError"""
        sizer = PositionSizer(mock_settings)
        with pytest.raises(ValueError, match="symbol_info is required"):
            await sizer.calculate_lot_size(
                symbol="XAUUSD",
                entry_price=2050.0,
                stop_loss=2040.0,
                account_balance=10000.0,
                symbol_info=None,
            )

    @pytest.mark.asyncio
    async def test_rejects_invalid_prices(self, mock_settings, mock_symbol_info):
        """Test that invalid prices raise ValueError"""
        sizer = PositionSizer(mock_settings)
        
        # Negative entry price
        with pytest.raises(ValueError, match="Invalid prices"):
            await sizer.calculate_lot_size(
                symbol="XAUUSD",
                entry_price=-100.0,
                stop_loss=2040.0,
                account_balance=10000.0,
                symbol_info=mock_symbol_info,
            )
        
        # Negative stop loss
        with pytest.raises(ValueError, match="Invalid prices"):
            await sizer.calculate_lot_size(
                symbol="XAUUSD",
                entry_price=2050.0,
                stop_loss=-100.0,
                account_balance=10000.0,
                symbol_info=mock_symbol_info,
            )