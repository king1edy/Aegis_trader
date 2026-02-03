"""
Base Strategy Interface

Abstract base class defining the interface for all trading strategies.
Ensures consistency across different strategy implementations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any

from src.execution.mt5_connector import BrokerInterface, OrderDirection
from src.core.logging_config import get_logger

logger = get_logger("base_strategy")


@dataclass
class StrategyConfig:
    """Base configuration for trading strategies"""
    name: str
    symbol: str
    enabled: bool = True
    max_positions: int = 1
    max_daily_trades: int = 3
    risk_per_trade: float = 0.01  # 1% risk per trade

    # Additional strategy-specific config can be added in subclasses
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TradingSignal:
    """
    Standardized trading signal output from strategy analysis.

    Contains all information needed to execute a trade:
    - Entry/exit levels
    - Risk parameters
    - Confidence and reasoning
    - Market context for debugging/analysis
    """
    # Basic info
    timestamp: datetime
    symbol: str
    direction: OrderDirection  # BUY or SELL

    # Entry and exit levels
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_final: Optional[float] = None

    # Signal quality
    confidence: float = 0.0  # 0.0 to 1.0
    reason: str = ""

    # Context for analysis and debugging
    market_context: Dict[str, Any] = field(default_factory=dict)
    strategy_data: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate signal after initialization"""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")

        if self.direction not in [OrderDirection.BUY, OrderDirection.SELL]:
            raise ValueError(f"Invalid direction: {self.direction}")

        # Validate SL/TP relationship
        if self.direction == OrderDirection.BUY:
            if self.stop_loss >= self.entry_price:
                raise ValueError("BUY: Stop loss must be below entry")
            if self.take_profit_1 <= self.entry_price:
                raise ValueError("BUY: Take profit must be above entry")
        else:
            if self.stop_loss <= self.entry_price:
                raise ValueError("SELL: Stop loss must be above entry")
            if self.take_profit_1 >= self.entry_price:
                raise ValueError("SELL: Take profit must be below entry")

    def get_risk_reward_ratio(self) -> float:
        """
        Calculate risk:reward ratio for TP1.

        Returns:
            Risk:reward ratio (e.g., 2.0 means 1:2 R:R)
        """
        sl_distance = abs(self.entry_price - self.stop_loss)
        tp_distance = abs(self.take_profit_1 - self.entry_price)

        if sl_distance == 0:
            return 0.0

        return tp_distance / sl_distance

    def get_sl_distance_pips(self, pip_size: float = 0.0001) -> float:
        """
        Get stop loss distance in pips.

        Args:
            pip_size: Pip size for the symbol (default: 0.0001 for FX)

        Returns:
            SL distance in pips
        """
        return abs(self.entry_price - self.stop_loss) / pip_size

    def to_dict(self) -> Dict[str, Any]:
        """Convert signal to dictionary for serialization"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "direction": self.direction.value,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit_1": self.take_profit_1,
            "take_profit_2": self.take_profit_2,
            "take_profit_final": self.take_profit_final,
            "confidence": self.confidence,
            "reason": self.reason,
            "risk_reward": self.get_risk_reward_ratio(),
            "market_context": self.market_context,
            "strategy_data": self.strategy_data
        }


@dataclass
class StrategyState:
    """
    Track strategy state for persistence across restarts.

    Stores last analysis times, bar timestamps, etc.
    """
    strategy_name: str
    last_analysis_time: Optional[datetime] = None
    last_signal_time: Optional[datetime] = None
    analysis_count: int = 0
    signal_count: int = 0
    state_data: Dict[str, Any] = field(default_factory=dict)


class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies.

    All strategies must implement:
    - analyze(): Generate trading signals
    - get_required_timeframes(): Specify data requirements

    The strategy should:
    - Be stateless (state stored externally)
    - Handle errors gracefully
    - Log decisions comprehensively
    - Validate outputs

    Usage:
        class MyStrategy(BaseStrategy):
            async def analyze(self, symbol: str) -> Optional[TradingSignal]:
                # Implementation here
                pass

            async def get_required_timeframes(self) -> List[str]:
                return ["H4", "H1", "M15"]
    """

    def __init__(
        self,
        config: StrategyConfig,
        broker: BrokerInterface
    ):
        """
        Initialize base strategy.

        Args:
            config: Strategy configuration
            broker: Broker interface for data access
        """
        self.config = config
        self.broker = broker

        self.logger = get_logger(f"strategy.{config.name}")

        self.logger.info(
            "Strategy initialized",
            name=config.name,
            symbol=config.symbol,
            max_positions=config.max_positions,
            risk_per_trade=config.risk_per_trade
        )

    @abstractmethod
    async def analyze(self, symbol: str) -> Optional[TradingSignal]:
        """
        Analyze market and generate trading signal.

        This is the main entry point for strategy logic.
        Should return None if no signal, or TradingSignal if conditions met.

        Args:
            symbol: Trading symbol to analyze

        Returns:
            TradingSignal if conditions met, None otherwise

        Raises:
            Should handle exceptions internally and return None on error
        """
        pass

    @abstractmethod
    async def get_required_timeframes(self) -> List[str]:
        """
        Get list of timeframes required by this strategy.

        Returns:
            List of timeframe strings (e.g., ["H4", "H1", "M15"])
        """
        pass

    async def validate_signal(
        self,
        signal: TradingSignal
    ) -> tuple[bool, str]:
        """
        Validate signal before execution.

        Checks:
        - Price levels are reasonable
        - Risk:reward is acceptable
        - Symbol is tradeable

        Args:
            signal: Signal to validate

        Returns:
            Tuple of (is_valid, reason)
        """
        try:
            # Check symbol is enabled
            if not self.config.enabled:
                return False, f"Strategy {self.config.name} is disabled"

            # Validate confidence
            if signal.confidence < 0.5:
                return False, f"Confidence too low: {signal.confidence}"

            # Validate risk:reward
            rr = signal.get_risk_reward_ratio()
            if rr < 1.0:
                return False, f"Risk:reward too low: {rr:.2f}"

            # Check symbol matches config
            if signal.symbol != self.config.symbol:
                self.logger.warning(
                    "Signal symbol mismatch",
                    expected=self.config.symbol,
                    got=signal.symbol
                )
                return False, "Symbol mismatch"

            # Check SL is not too wide (> 5% of entry)
            sl_percent = abs(signal.entry_price - signal.stop_loss) / signal.entry_price * 100
            if sl_percent > 5.0:
                return False, f"Stop loss too wide: {sl_percent:.2f}%"

            # All checks passed
            return True, "Signal validated"

        except Exception as e:
            self.logger.exception("Error validating signal", error=str(e))
            return False, f"Validation error: {str(e)}"

    async def on_trade_opened(self, signal: TradingSignal, ticket: int) -> None:
        """
        Hook called after trade is opened.

        Can be overridden for strategy-specific post-trade actions.

        Args:
            signal: Original signal
            ticket: Broker ticket number
        """
        self.logger.info(
            "Trade opened",
            strategy=self.config.name,
            ticket=ticket,
            symbol=signal.symbol,
            direction=signal.direction.value
        )

    async def on_trade_closed(
        self,
        ticket: int,
        profit: float,
        outcome: str
    ) -> None:
        """
        Hook called after trade is closed.

        Can be overridden for strategy-specific learning/adaptation.

        Args:
            ticket: Broker ticket number
            profit: Realized profit/loss
            outcome: "WIN", "LOSS", or "BREAKEVEN"
        """
        self.logger.info(
            "Trade closed",
            strategy=self.config.name,
            ticket=ticket,
            profit=profit,
            outcome=outcome
        )

    async def get_state(self) -> StrategyState:
        """
        Get current strategy state for persistence.

        Returns:
            StrategyState object
        """
        return StrategyState(
            strategy_name=self.config.name,
            state_data={}
        )

    async def load_state(self, state: StrategyState) -> None:
        """
        Load strategy state from persistence.

        Args:
            state: Previously saved state
        """
        self.logger.info(
            "Strategy state loaded",
            strategy=self.config.name,
            analysis_count=state.analysis_count,
            signal_count=state.signal_count
        )

    def get_info(self) -> Dict[str, Any]:
        """
        Get strategy information for monitoring.

        Returns:
            Dictionary with strategy details
        """
        return {
            "name": self.config.name,
            "symbol": self.config.symbol,
            "enabled": self.config.enabled,
            "max_positions": self.config.max_positions,
            "max_daily_trades": self.config.max_daily_trades,
            "risk_per_trade": self.config.risk_per_trade,
            "required_timeframes": []  # Will be populated by subclass
        }
