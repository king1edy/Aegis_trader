"""
Session Filter for MTFTR Strategy

Ensures trades only occur during active trading sessions:
- London Session: 07:00-12:00 GMT
- New York Session: 13:00-16:00 GMT

Critical for risk management - avoids low-liquidity periods.
"""

from datetime import datetime, time, timezone
from typing import Optional, Tuple
from dataclasses import dataclass

from src.core.logging_config import get_logger

logger = get_logger("session_filter")


@dataclass
class SessionConfig:
    """Configuration for trading sessions"""
    london_start: str = "07:00"
    london_end: str = "12:00"
    ny_start: str = "13:00"
    ny_end: str = "16:00"


class SessionFilter:
    """
    Filter trades based on active trading sessions.

    Only allows trades during:
    - London Session: 07:00-12:00 GMT
    - New York Session: 13:00-16:00 GMT

    Usage:
        filter = SessionFilter(london_start="07:00", london_end="12:00", ...)
        is_tradeable, reason = await filter.is_tradeable_time()
        if is_tradeable:
            # Proceed with trade
    """

    def __init__(
        self,
        london_start: str = "07:00",
        london_end: str = "12:00",
        ny_start: str = "13:00",
        ny_end: str = "16:00"
    ):
        """
        Initialize session filter.

        Args:
            london_start: London session start time (HH:MM format)
            london_end: London session end time (HH:MM format)
            ny_start: NY session start time (HH:MM format)
            ny_end: NY session end time (HH:MM format)
        """
        self.config = SessionConfig(
            london_start=london_start,
            london_end=london_end,
            ny_start=ny_start,
            ny_end=ny_end
        )

        # Parse session times
        self.london_start = self._parse_time(london_start)
        self.london_end = self._parse_time(london_end)
        self.ny_start = self._parse_time(ny_start)
        self.ny_end = self._parse_time(ny_end)

        logger.info(
            "Session filter initialized",
            london_session=f"{london_start}-{london_end}",
            ny_session=f"{ny_start}-{ny_end}"
        )

    async def is_tradeable_time(
        self,
        current_time: Optional[datetime] = None
    ) -> Tuple[bool, str]:
        """
        Check if current time is within tradeable sessions.

        Args:
            current_time: Time to check (defaults to now in GMT/UTC)

        Returns:
            Tuple of (is_tradeable, session_name_or_reason)
            - (True, "london") if in London session
            - (True, "ny") if in NY session
            - (False, "outside_hours") if outside sessions
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)

        # Ensure timezone-aware
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=timezone.utc)
        else:
            current_time = current_time.astimezone(timezone.utc)

        current_time_only = current_time.time()

        # Check London session
        if self._is_time_in_range(
            current_time_only,
            self.london_start,
            self.london_end
        ):
            logger.debug(
                "Within London session",
                current_time=current_time.strftime("%Y-%m-%d %H:%M:%S %Z")
            )
            return True, "london"

        # Check NY session
        if self._is_time_in_range(
            current_time_only,
            self.ny_start,
            self.ny_end
        ):
            logger.debug(
                "Within NY session",
                current_time=current_time.strftime("%Y-%m-%d %H:%M:%S %Z")
            )
            return True, "ny"

        # Outside trading hours
        logger.debug(
            "Outside trading sessions",
            current_time=current_time.strftime("%Y-%m-%d %H:%M:%S %Z"),
            london_session=f"{self.config.london_start}-{self.config.london_end}",
            ny_session=f"{self.config.ny_start}-{self.config.ny_end}"
        )
        return False, "outside_hours"

    async def get_current_session(
        self,
        current_time: Optional[datetime] = None
    ) -> Optional[str]:
        """
        Get the current active session name.

        Args:
            current_time: Time to check (defaults to now in GMT/UTC)

        Returns:
            "london", "ny", or None if outside trading hours
        """
        is_tradeable, session = await self.is_tradeable_time(current_time)

        if is_tradeable:
            return session

        return None

    async def get_next_session_start(
        self,
        current_time: Optional[datetime] = None
    ) -> Tuple[datetime, str]:
        """
        Calculate when the next trading session starts.

        Args:
            current_time: Reference time (defaults to now in GMT/UTC)

        Returns:
            Tuple of (next_session_start_datetime, session_name)
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)

        # Ensure timezone-aware
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=timezone.utc)
        else:
            current_time = current_time.astimezone(timezone.utc)

        current_time_only = current_time.time()

        # Check if before London session today
        if current_time_only < self.london_start:
            next_start = datetime.combine(
                current_time.date(),
                self.london_start,
                tzinfo=timezone.utc
            )
            return next_start, "london"

        # Check if before NY session today
        if current_time_only < self.ny_start:
            next_start = datetime.combine(
                current_time.date(),
                self.ny_start,
                tzinfo=timezone.utc
            )
            return next_start, "ny"

        # After NY session - next is London tomorrow
        from datetime import timedelta
        next_day = current_time.date() + timedelta(days=1)
        next_start = datetime.combine(
            next_day,
            self.london_start,
            tzinfo=timezone.utc
        )
        return next_start, "london"

    def _parse_time(self, time_str: str) -> time:
        """
        Parse time string in HH:MM format.

        Args:
            time_str: Time string (e.g., "07:00", "13:30")

        Returns:
            datetime.time object

        Raises:
            ValueError: If format is invalid
        """
        try:
            hour, minute = time_str.split(":")
            return time(int(hour), int(minute))
        except (ValueError, AttributeError) as e:
            logger.error("Invalid time format", time_str=time_str, error=str(e))
            raise ValueError(f"Invalid time format: {time_str}. Expected HH:MM")

    def _is_time_in_range(
        self,
        check_time: time,
        start_time: time,
        end_time: time
    ) -> bool:
        """
        Check if time is within range (inclusive start, exclusive end).

        Args:
            check_time: Time to check
            start_time: Range start
            end_time: Range end

        Returns:
            True if check_time >= start_time and check_time < end_time
        """
        return start_time <= check_time < end_time

    def get_session_info(self) -> dict:
        """
        Get session configuration information.

        Returns:
            Dictionary with session details
        """
        return {
            "london_session": {
                "start": self.config.london_start,
                "end": self.config.london_end,
                "duration_hours": self._calculate_duration(
                    self.london_start,
                    self.london_end
                )
            },
            "ny_session": {
                "start": self.config.ny_start,
                "end": self.config.ny_end,
                "duration_hours": self._calculate_duration(
                    self.ny_start,
                    self.ny_end
                )
            },
            "total_tradeable_hours": (
                self._calculate_duration(self.london_start, self.london_end) +
                self._calculate_duration(self.ny_start, self.ny_end)
            )
        }

    def _calculate_duration(self, start: time, end: time) -> float:
        """
        Calculate duration between two times in hours.

        Args:
            start: Start time
            end: End time

        Returns:
            Duration in hours
        """
        start_minutes = start.hour * 60 + start.minute
        end_minutes = end.hour * 60 + end.minute

        if end_minutes < start_minutes:
            # Crosses midnight
            end_minutes += 24 * 60

        duration_minutes = end_minutes - start_minutes
        return duration_minutes / 60.0
