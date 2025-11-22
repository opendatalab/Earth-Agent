"""
Training-Free GRPO for Earth-Agent

This module implements Training-Free Group Relative Policy Optimization
to improve Earth-Agent performance without model parameter updates.
"""

from .training_free_grpo import TrainingFreeGRPO
from .rollout_manager import RolloutManager
from .experience_updater import ExperienceUpdater
from .data_manager import DataManager
from .config import TrainingFreeGRPOConfig, PracticeArguments

__all__ = [
    'TrainingFreeGRPO',
    'RolloutManager',
    'ExperienceUpdater',
    'DataManager',
    'TrainingFreeGRPOConfig',
    'PracticeArguments'
]

__version__ = "1.0.0"
