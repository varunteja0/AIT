"""Health checks for the platform."""

from __future__ import annotations

from autonomous_trading_researcher.core.models import RiskSnapshot


class SystemHealthCheck:
    """Evaluate overall system health for monitoring."""

    def assess(self, datasets_ready: int, risk_snapshot: RiskSnapshot) -> bool:
        """Return whether the platform is currently healthy."""

        return datasets_ready > 0 and not risk_snapshot.halted

