"""Agent implementations for the cart+double-pendulum system."""

from __future__ import annotations

import numpy as np

from .config import HoldConfig


class PDHoldAgent:
    """PD feedback baseline controller for keeping θ1 and θ2 near upright position.
    
    This agent implements a proportional-derivative controller that attempts
    to stabilize both pendulums in the upright position.
    """
    
    def __init__(self, cfg: HoldConfig) -> None:
        """Initialize the PD controller.
        
        Args:
            cfg: Configuration containing PD gains.
        """
        self.cfg = cfg
    
    def act(self, obs: np.ndarray) -> float:
        """Compute control action from observation.
        
        Args:
            obs: Observation array [x, dx, th1, dth1, th2, dth2].
        
        Returns:
            Control force u.
        """
        x, dx, th1, dth1, th2, dth2 = [float(v) for v in obs.tolist()]
        u = (
            self.cfg.kx * x
            - self.cfg.vx * dx
            - self.cfg.k1 * th1
            - self.cfg.kw1 * dth1
            - self.cfg.k2 * th2
            - self.cfg.kw2 * dth2
        )
        return float(u)
