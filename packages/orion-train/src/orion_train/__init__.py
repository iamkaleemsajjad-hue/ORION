"""Training engine and safe checkpoint recovery for ORION models."""

from orion_train.checkpoint import CheckpointManager
from orion_train.config import TrainerConfig
from orion_train.distributed import DistributedContext, initialize_ddp
from orion_train.optim import build_adamw, build_parameter_groups
from orion_train.schedules import TrapezoidalSchedule, WarmupStableDecaySchedule
from orion_train.trainer import Trainer, TrainerState

__all__ = [
    "CheckpointManager",
    "DistributedContext",
    "Trainer",
    "TrainerConfig",
    "TrainerState",
    "TrapezoidalSchedule",
    "WarmupStableDecaySchedule",
    "build_adamw",
    "build_parameter_groups",
    "initialize_ddp",
]

__version__ = "0.1.0"
