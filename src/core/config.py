"""
Configuration Management
========================
Centralized settings management using Pydantic Settings.
All configuration is loaded from environment variables with validation.
"""

from functools import lru_cache
from typing import List, Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from enum import Enum


class Environment(str, Enum):
    """Application environment."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class TradingSession(str, Enum):
    """Trading session identifiers."""
    ASIAN = "asian"
    LONDON = "london"
    NEWYORK = "newyork"


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    All settings are validated on startup. Invalid configuration
    will prevent the application from starting.
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # -------------------------------------------------------------------------
    # Application Settings
    # -------------------------------------------------------------------------
    app_name: str = "AegisAutomatedTradingSystem"
    app_env: Environment = Environment.DEVELOPMENT
    debug: bool = False
    log_level: str = "INFO"
    
    # -------------------------------------------------------------------------
    # MetaTrader 5 Configuration
    # -------------------------------------------------------------------------
    mt5_login: int = Field(default=0, description="MT5 account number")
    mt5_password: str = Field(default="", description="MT5 password")
    mt5_server: str = Field(default="Exness-MT5Trial", description="MT5 server name")
    mt5_path: str = Field(
        default="C:/Program Files/MetaTrader 5/terminal64.exe",
        description="Path to MT5 terminal"
    )
    
    # Broker connection mode: "direct" (Windows), "bridge" (Docker/remote), "paper" (simulation)
    broker_mode: str = Field(
        default="auto",
        description="Broker mode: 'auto' (detect), 'direct' (MT5 on Windows), 'bridge' (API bridge), 'paper' (simulation)"
    )
    
    # MT5 API Bridge settings (for running in Docker or non-Windows)
    mt5_bridge_url: str = Field(
        default="http://host.docker.internal:8001",
        description="URL of the MT5 API bridge server"
    )
    mt5_bridge_api_key: str = Field(
        default="",
        description="Optional API key for MT5 bridge authentication"
    )
    mt5_bridge_host: str = Field(default="0.0.0.0", description="Host for MT5 bridge server")
    mt5_bridge_port: int = Field(default=8001, description="Port for MT5 bridge server")
    
    # Legacy settings (kept for compatibility)
    mt5_api_host: str = "localhost"
    mt5_api_port: int = 8001
    use_mt5: bool = Field(default=False, description="Force using real MT5 connector instead of demo")
    
    # -------------------------------------------------------------------------
    # Trading Parameters
    # -------------------------------------------------------------------------
    default_symbol: str = "XAUUSD"
    
    # Risk Management
    max_risk_per_trade: float = Field(
        default=0.01,
        ge=0.001,
        le=0.05,
        description="Maximum risk per trade (1% = 0.01)"
    )
    max_daily_risk: float = Field(
        default=0.03,
        ge=0.01,
        le=0.10,
        description="Maximum daily risk (3% = 0.03)"
    )
    max_drawdown_percent: float = Field(
        default=0.10,
        ge=0.05,
        le=0.25,
        description="Maximum drawdown before shutdown"
    )
    max_trades_per_day: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum number of trades per day"
    )
    min_trade_interval_minutes: int = Field(
        default=60,
        ge=5,
        description="Minimum minutes between trades"
    )
    
    # Position Sizing
    default_lot_size: float = Field(default=0.01, ge=0.01, le=1.0)
    max_lot_size: float = Field(default=0.05, ge=0.01, le=1.0)
    use_atr_sizing: bool = True
    atr_multiplier: float = Field(default=1.5, ge=1.0, le=3.0)
    
    # Session Filtering (GMT times as strings)
    london_session_start: str = "07:00"
    london_session_end: str = "12:00"
    ny_session_start: str = "13:00"
    ny_session_end: str = "16:00"
    trade_sessions: str = "london,newyork"

    # -------------------------------------------------------------------------
    # MTFTR Strategy Configuration
    # -------------------------------------------------------------------------
    # Strategy enable/disable
    mtftr_enabled: bool = True

    # Indicator periods
    mtftr_ema_200: int = Field(default=200, ge=50, le=400)
    mtftr_ema_50: int = Field(default=50, ge=20, le=100)
    mtftr_ema_21: int = Field(default=21, ge=10, le=50)
    mtftr_hull_55: int = Field(default=55, ge=20, le=100)
    mtftr_hull_34: int = Field(default=34, ge=10, le=60)
    mtftr_rsi_period: int = Field(default=14, ge=5, le=30)
    mtftr_atr_period: int = Field(default=14, ge=5, le=30)
    mtftr_swing_lookback: int = Field(default=5, ge=3, le=10)

    # Risk/Reward ratios
    mtftr_tp1_rr: float = Field(default=1.0, ge=0.5, le=2.0, description="TP1 risk:reward")
    mtftr_tp2_rr: float = Field(default=2.0, ge=1.0, le=5.0, description="TP2 risk:reward")

    # Scaled exit percentages
    mtftr_tp1_close_percent: float = Field(default=0.50, ge=0.3, le=0.7, description="Close 50% at TP1")
    mtftr_tp2_close_percent: float = Field(default=0.30, ge=0.2, le=0.5, description="Close 30% at TP2")
    mtftr_trail_percent: float = Field(default=0.20, ge=0.1, le=0.5, description="Trail 20%")

    # Entry filters
    mtftr_min_rsi_long: float = Field(default=40.0, ge=20.0, le=50.0)
    mtftr_max_rsi_long: float = Field(default=55.0, ge=50.0, le=70.0)
    mtftr_min_rsi_short: float = Field(default=45.0, ge=30.0, le=55.0)
    mtftr_max_rsi_short: float = Field(default=60.0, ge=50.0, le=80.0)

    # Stop loss limits (in ATR multiples)
    mtftr_min_sl_atr: float = Field(default=1.0, ge=0.5, le=2.0)
    mtftr_max_sl_atr: float = Field(default=3.0, ge=2.0, le=5.0)
    mtftr_sl_buffer_atr: float = Field(default=0.5, ge=0.2, le=1.0)

    # Position limits
    max_open_positions: int = Field(default=2, ge=1, le=5, description="Max concurrent positions")
    max_daily_trades: int = Field(default=3, ge=1, le=10, description="Max trades per day")
    mtftr_max_trade_hours: int = Field(default=8, ge=4, le=24, description="Max hours per trade")

    # Risk limits
    max_daily_loss_percent: float = Field(default=0.03, ge=0.01, le=0.10, description="Max 3% daily loss")
    min_margin_level: float = Field(default=200.0, ge=100.0, le=500.0, description="Min margin level %")

    # -------------------------------------------------------------------------
    # Database Configuration
    # -------------------------------------------------------------------------
    postgres_user: str = "trading_user"
    postgres_password: str = "trading_pass"
    postgres_db: str = "trading_db"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    database_url: Optional[str] = None
    
    # -------------------------------------------------------------------------
    # Redis Configuration
    # -------------------------------------------------------------------------
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_url: Optional[str] = None
    
    # -------------------------------------------------------------------------
    # Telegram Notifications
    # -------------------------------------------------------------------------
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    telegram_enabled: bool = False
    notify_on_trade_open: bool = True
    notify_on_trade_close: bool = True
    notify_on_daily_summary: bool = True
    notify_on_error: bool = True
    notify_on_drawdown_warning: bool = True
    
    # -------------------------------------------------------------------------
    # API Configuration
    # -------------------------------------------------------------------------
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_secret_key: str = "change-this-in-production"
    tradingview_webhook_secret: str = ""
    
    # -------------------------------------------------------------------------
    # Monitoring
    # -------------------------------------------------------------------------
    grafana_user: str = "admin"
    grafana_password: str = "admin"
    enable_metrics: bool = True
    metrics_port: int = 9100
    
    # -------------------------------------------------------------------------
    # Behavioral Safeguards
    # -------------------------------------------------------------------------
    enable_manual_override: bool = Field(
        default=False,
        description="DANGER: Set to False in production to prevent manual intervention"
    )
    require_confirmation_for_manual: bool = True
    cooldown_after_loss_minutes: int = Field(default=30, ge=0)
    max_consecutive_losses: int = Field(default=3, ge=1, le=10)
    pause_duration_hours: int = Field(default=4, ge=1, le=24)
    
    # -------------------------------------------------------------------------
    # Backtesting
    # -------------------------------------------------------------------------
    backtest_data_path: str = "/app/data/historical"
    backtest_start_date: str = "2023-01-01"
    backtest_end_date: str = "2024-12-31"
    
    # -------------------------------------------------------------------------
    # Computed Properties
    # -------------------------------------------------------------------------
    @property
    def db_url(self) -> str:
        """Get the database URL, constructing it if not provided."""
        if self.database_url:
            return self.database_url
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )
    
    @property
    def async_db_url(self) -> str:
        """Get async database URL for asyncpg."""
        return self.db_url.replace("postgresql://", "postgresql+asyncpg://")
    
    @property
    def redis_connection_url(self) -> str:
        """Get the Redis URL, constructing it if not provided."""
        if self.redis_url:
            return self.redis_url
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"
    
    @property
    def active_sessions(self) -> List[TradingSession]:
        """Parse active trading sessions from comma-separated string."""
        sessions = []
        for session in self.trade_sessions.split(","):
            session = session.strip().lower()
            if session in [s.value for s in TradingSession]:
                sessions.append(TradingSession(session))
        return sessions
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.app_env == Environment.PRODUCTION
    
    # -------------------------------------------------------------------------
    # Validators
    # -------------------------------------------------------------------------
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Ensure log level is valid."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v = v.upper()
        if v not in valid_levels:
            raise ValueError(f"Invalid log level. Must be one of: {valid_levels}")
        return v
    
    @field_validator("max_lot_size")
    @classmethod
    def validate_max_lot_size(cls, v: float, info) -> float:
        """Ensure max_lot_size >= default_lot_size."""
        # Note: In Pydantic v2, we can't easily access other fields in validators
        # This validation should be done in model_validator instead
        return v
    
    def validate_trading_config(self) -> List[str]:
        """
        Validate trading configuration and return list of warnings.
        Call this after loading settings to check for potential issues.
        """
        warnings = []
        
        if self.max_lot_size < self.default_lot_size:
            warnings.append(
                f"max_lot_size ({self.max_lot_size}) < default_lot_size ({self.default_lot_size})"
            )
        
        if self.is_production and self.enable_manual_override:
            warnings.append(
                "CRITICAL: Manual override is enabled in production! "
                "This defeats the purpose of automated trading."
            )
        
        if self.is_production and self.debug:
            warnings.append("Debug mode is enabled in production")
        
        if self.max_risk_per_trade > 0.02:
            warnings.append(
                f"Risk per trade ({self.max_risk_per_trade*100}%) exceeds recommended 2%"
            )
        
        if not self.telegram_enabled and self.is_production:
            warnings.append("Telegram notifications are disabled in production")
        
        if not self.mt5_login or not self.mt5_password:
            warnings.append("MT5 credentials not configured")
        
        return warnings


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Using lru_cache ensures settings are only loaded once.
    """
    return Settings()


# Create a global settings instance for easy import
settings = get_settings()
