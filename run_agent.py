"""Baseline controller and environment wrapper for `physics_engine.py`.

Полностью SAC здесь не включаем (требует torch + корректная типизация/нейросети).
Функционально: среда + PD-контроллер, использующий динамику из `physics_engine.py`.

Этот файл обеспечивает обратную совместимость, импортируя все компоненты из модуля agent.
"""

from __future__ import annotations

from agent import HoldConfig
from agent import PhysicsEnv
from agent import PDHoldAgent
from agent import compute_reward as _reward
from agent import train_sac_stable_baselines3, train_mlp, run_episode

__all__ = [
    "HoldConfig",
    "PhysicsEnv",
    "PDHoldAgent",
    "_reward",
    "train_sac_stable_baselines3",
    "train_mlp",
    "run_episode",
]

if __name__ == "__main__":
    # Quick training (уменьшено, чтобы не убивало процесс по времени/памяти)
    train_sac_stable_baselines3()
