"""
Shared Test Fixtures
====================
Reusable fixtures available to all test files automatically.

Add new fixtures here whenever multiple test modules need the same setup.
"""

import os
import numpy as np
import pandas as pd
import pytest
from decimal import Decimal
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Prevent Settings from loading a real .env during tests
# ---------------------------------------------------------------------------
os.environ.setdefault("APP_ENV", "development")
os.environ.setdefault("MT5_LOGIN", "0")
os.environ.setdefault("MT5_PASSWORD", "test")

from src.core.config import Settings
from src.execution.mt5_connector import SymbolInfo, AccountInfo


# ---------------------------------------------------------------------------
# Configuration Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_settings() -> Settings:
    """
    Return a Settings instance with safe defaults.

    Overrides are passed as keyword arguments so the real .env
    file is never required.
    """
    return Settings(
        app_env="development",
        mt5_login=0,
        mt5_password="test",
        mt5_server="TestServer",
        broker_mode="paper",
        telegram_enabled=False,
    )


# ---------------------------------------------------------------------------
# Market‑Data Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """
    Generate a 300‑bar OHLCV DataFrame with realistic XAUUSD prices.

    The data uses a simple random‑walk so indicator calculations
    behave deterministically (seeded).
    """
    np.random.seed(42)
    n = 300
    base_price = 2050.0
    returns = np.random.normal(0, 0.002, n)
    close = base_price * np.cumprod(1 + returns)

    high = close * (1 + np.abs(np.random.normal(0, 0.001, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.001, n)))
    open_ = np.roll(close, 1)
    open_[0] = base_price

    df = pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": np.random.randint(100, 5000, n),
    })
    return df


# ---------------------------------------------------------------------------
# Broker / Symbol Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_symbol_info() -> SymbolInfo:
    """SymbolInfo for XAUUSD with typical broker parameters."""
    return SymbolInfo(
        name="XAUUSD",
        description="Gold vs US Dollar",
        digits=2,
        point=0.01,
        spread=20,
        tick_size=0.01,
        tick_value=1.0,
        min_lot=0.01,
        max_lot=100.0,
        lot_step=0.01,
        contract_size=100.0,
        margin_required=1000.0,
        trade_allowed=True,
    )


@pytest.fixture
def mock_account_info() -> AccountInfo:
    """AccountInfo representing a $10 000 demo account."""
    return AccountInfo(
        login=12345,
        name="Test Account",
        server="TestServer",
        currency="USD",
        balance=Decimal("10000.00"),
        equity=Decimal("10000.00"),
        margin=Decimal("0.00"),
        free_margin=Decimal("10000.00"),
        margin_level=None,
        profit=Decimal("0.00"),
        leverage=100,
    )
