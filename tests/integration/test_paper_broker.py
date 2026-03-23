"""
Integration Tests — Paper Trading Broker
==========================================
Tests for: src/execution/paper_broker.py

Integration because PaperTradingBroker coordinates Settings, price
simulation, and position management together.
"""

import pytest
from decimal import Decimal

from src.execution.paper_broker import PaperTradingBroker

pytestmark = pytest.mark.integration


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

    # TODO: test_double_connect_is_idempotent


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

    # TODO: test_equity_matches_balance_with_no_positions


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

    # TODO: test_sell_order_creates_position
    # TODO: test_multiple_orders_generate_unique_tickets


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

    # TODO: test_close_nonexistent_ticket_returns_failure
    # TODO: test_modify_updates_sl_and_tp
    # TODO: test_update_price_affects_pnl


# ---------------------------------------------------------------------------
# Price Data
# ---------------------------------------------------------------------------

class TestPriceData:
    """Tests for get_price_data (simulated bars)."""

    # TODO: test_returns_requested_number_of_bars
    # TODO: test_bars_have_ohlcv_fields
    pass
