"""Live execution scaffold: broker adapter, paper broker, rate limit, risk, scheduler, manager.

A code-complete order/position lifecycle drivable against :class:`PaperBroker` for testing.
Wiring a real broker means implementing :class:`BrokerAdapter` and feeding a live event
stream through :class:`~bsealpha.features.StreamingFeatureEngine`; SEBI onboarding / Algo-ID
provisioning remain out-of-band (§8.1).
"""

from __future__ import annotations

from .broker import BrokerAdapter, Fill, Order, PaperBroker, Position
from .manager import ExecutionManager, StepResult
from .mt5_broker import MT5BrokerAdapter, connect_mt5
from .rate_limit import TokenBucket
from .risk import RiskLimits, RiskMonitor
from .safety import DeploymentGuard, DeploymentMode, PreflightResult
from .scheduler import ForcedFlattenSchedule

__all__ = [
    "BrokerAdapter",
    "PaperBroker",
    "Order",
    "Fill",
    "Position",
    "TokenBucket",
    "RiskLimits",
    "RiskMonitor",
    "ForcedFlattenSchedule",
    "ExecutionManager",
    "StepResult",
    "DeploymentGuard",
    "DeploymentMode",
    "PreflightResult",
    "MT5BrokerAdapter",
    "connect_mt5",
]
