from dataclasses import dataclass
from typing import Optional


@dataclass
class SimulationConfig:
    """Centralized configuration for the simulation"""

    run_until: int = 10000
    warmup_time: int = 1000
    log_interval: int = 100
    monitor_interval: int = 500
    seed: Optional[int] = None
