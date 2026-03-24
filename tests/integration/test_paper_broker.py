"""
Integration Tests — Paper Trading Broker
==========================================
Tests for: src/execution/paper_broker.py

Integration because PaperTradingBroker coordinates Settings, price
simulation, and position management together.
"""

import pytest
from decimal import Decimal
from datetime import datetime, timedelta
import asyncio

from src.execution.paper_broker import PaperTradingBroker

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_settings():
    """Create mock settings for testing"""
    class MockSettings:
        def __init__(self):
            self.initial_balance = Decimal("10000.00")
            self.magic_number = 123456
            self.symbol = "XAUUSD"
    
    return MockSettings()


@pytest.fixture
async def connected_broker(mock_settings):
    """Create a connected broker instance"""
    broker = PaperTradingBroker(mock_settings)
    await broker.connect()
    yield broker
    await broker.disconnect()


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

class TestConnection:
    """Tests for connect/disconnect lifecycle."""

    @pytest.mark.asyncio
    async def test_connect_and_disconnect(self, mock_settings):
        broker = PaperTradingBroker(mock_settings)
        assert broker.is_connected is False

        result = await broker.connect()
        assert result is True
        assert broker.is_connected is True

        await broker.disconnect()
        assert broker.is_connected is False

    # test_double_connect_is_idempotent#
    @pytest.mark.asyncio
    async def test_double_connect_is_idempotent(self, mock_settings):
        """Test that connecting twice doesn't cause issues"""
        broker = PaperTradingBroker(mock_settings)
        
        result1 = await broker.connect()
        assert result1 is True
        assert broker.is_connected is True
        
        result2 = await broker.connect()
        assert result2 is True
        assert broker.is_connected is True
        
        await broker.disconnect()
        assert broker.is_connected is False


# ---------------------------------------------------------------------------
# Account Info
# ---------------------------------------------------------------------------

class TestAccountInfo:
    """Tests for get_account_info."""

    @pytest.mark.asyncio
    async def test_initial_balance(self, mock_settings):
        broker = PaperTradingBroker(mock_settings)
        await broker.connect()
        info = await broker.get_account_info()
        assert info["balance"] == Decimal("10000.00")

    #test_equity_matches_balance_with_no_positions#

    @pytest.mark.asyncio
    async def test_equity_matches_balance_with_no_positions(self, connected_broker):
        """Test that equity equals balance when no positions"""
        info = await connected_broker.get_account_info()
        assert info["equity"] == info["balance"]
        assert info["balance"] == Decimal("10000.00")


# ---------------------------------------------------------------------------
# Order Placement
# ---------------------------------------------------------------------------

class TestPlaceOrder:
    """Tests for place_order and open_position."""

    @pytest.mark.asyncio
    async def test_buy_order_creates_position(self, mock_settings):
        broker = PaperTradingBroker(mock_settings)
        await broker.connect()

        result = await broker.place_order(
            symbol="XAUUSD",
            order_type="buy",
            volume=0.01,
            sl=2040.0,
            tp=2060.0,
        )
        assert result["success"] is True
        assert "ticket" in result

        positions = await broker.get_positions()
        assert len(positions) == 1

    # test_sell_order_creates_position#

    @pytest.mark.asyncio
    async def test_sell_order_creates_position(self, connected_broker):
        """Test that sell order creates a position"""
        result = await connected_broker.place_order(
            symbol="XAUUSD",
            order_type="sell",
            volume=0.01,
            sl=2060.0,
            tp=2040.0,
        )
        assert result["success"] is True
        assert "ticket" in result

        positions = await connected_broker.get_positions()
        assert len(positions) == 1
        assert positions[0]["type"] == "sell"

    # test_multiple_orders_generate_unique_tickets#

    @pytest.mark.asyncio
    async def test_multiple_orders_generate_unique_tickets(self, connected_broker):
        """Test that multiple orders generate unique ticket numbers"""
        tickets = set()
        
        for i in range(5):
            result = await connected_broker.place_order(
                symbol="XAUUSD",
                order_type="buy" if i % 2 == 0 else "sell",
                volume=0.01
            )
            assert result["success"] is True
            tickets.add(result["ticket"])
        
        assert len(tickets) == 5
        
        positions = await connected_broker.get_positions()
        assert len(positions) == 5



# ---------------------------------------------------------------------------
# Close / Modify
# ---------------------------------------------------------------------------

class TestCloseAndModify:
    """Tests for close_position and modify_position."""

    @pytest.mark.asyncio
    async def test_close_removes_position(self, mock_settings):
        broker = PaperTradingBroker(mock_settings)
        await broker.connect()
        order = await broker.place_order("XAUUSD", "buy", 0.01)
        ticket = order["ticket"]

        result = await broker.close_position(ticket)
        assert result["success"] is True

        positions = await broker.get_positions()
        assert len(positions) == 0

    # test_close_nonexistent_ticket_returns_failure#

    @pytest.mark.asyncio
    async def test_close_nonexistent_ticket_returns_failure(self, connected_broker):
        """Test closing a non-existent ticket returns failure"""
        result = await connected_broker.close_position(99999)
        assert result["success"] is False
        assert "not found" in result["error"].lower()

    # test_modify_updates_sl_and_tp#

    @pytest.mark.asyncio
    async def test_modify_updates_sl_and_tp(self, connected_broker):
        """Test that modify_position updates stop loss and take profit"""
        order = await connected_broker.place_order(
            symbol="XAUUSD",
            order_type="buy",
            volume=0.01,
            sl=2040.0,
            tp=2060.0
        )
        ticket = order["ticket"]
        
        # Modify SL and TP
        result = await connected_broker.modify_position(
            ticket=ticket,
            sl=2045.0,
            tp=2065.0
        )
        assert result["success"] is True
        
        # Verify changes
        position = await connected_broker.get_position(ticket)
        assert position["sl"] == Decimal("2045.0")
        assert position["tp"] == Decimal("2065.0")

    # test_update_price_affects_pnl#

    @pytest.mark.asyncio
    async def test_update_price_affects_pnl(self, connected_broker):
        """Test that price updates affect unrealized P&L"""
        # Open a buy position
        order = await connected_broker.place_order(
            symbol="XAUUSD",
            order_type="buy",
            volume=0.1,
            sl=2040.0,
            tp=2060.0
        )
        
        # Get initial P&L
        position = await connected_broker.get_position(order["ticket"])
        initial_pnl = position["profit"]
        
        # Update price higher
        connected_broker.update_price("XAUUSD", 2055.00, 2055.20)
        
        # Get updated P&L (should be positive)
        position = await connected_broker.get_position(order["ticket"])
        assert position["profit"] > initial_pnl

# ---------------------------------------------------------------------------
# Price Data
# ---------------------------------------------------------------------------

class TestPriceData:
    """Tests for get_price_data (simulated bars)."""

    # test_returns_requested_number_of_bars#

    @pytest.mark.asyncio
    async def test_returns_requested_number_of_bars(self, connected_broker):
        """Test that get_price_data returns the requested number of bars"""
        count = 100
        bars = await connected_broker.get_price_data(
            symbol="XAUUSD",
            timeframe="M15",
            count=count
        )
        
        assert len(bars) == count


    # test_bars_have_ohlcv_fields#

    @pytest.mark.asyncio
    async def test_bars_have_ohlcv_fields(self, connected_broker):
        """Test that bars have all required OHLCV fields"""
        bars = await connected_broker.get_price_data(
            symbol="XAUUSD",
            timeframe="M1",
            count=10
        )
        
        for bar in bars:
            assert hasattr(bar, 'timestamp')
            assert hasattr(bar, 'open')
            assert hasattr(bar, 'high')
            assert hasattr(bar, 'low')
            assert hasattr(bar, 'close')
            assert hasattr(bar, 'tick_volume')
            assert bar.open <= bar.high
            assert bar.low <= bar.close
            assert bar.low <= bar.high
