"""Configuration parameters for the cart+double-pendulum system and PID controller."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class PIDConfig:
    """Configuration for the PID controller.
    
    Attributes:
        Kp: Proportional gain for each state variable (6x1 vector).
        Ki: Integral gain for each state variable (6x1 vector).
        Kd: Derivative gain for each state variable (6x1 vector).
        u_max: Maximum control force (N).
        u_min: Minimum control force (N).
        integral_limit: Maximum integral term value (anti-windup).
    """
    # Gains for each state: [x, dx, theta1, dtheta1, theta2, dtheta2]
    Kp: Optional[np.ndarray] = None
    Ki: Optional[np.ndarray] = None
    Kd: Optional[np.ndarray] = None
    u_max: float = 100.0
    u_min: float = -100.0
    integral_limit: float = 1000.0
    
    def __post_init__(self) -> None:
        if self.Kp is None:
            self.Kp = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        if self.Ki is None:
            self.Ki = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        if self.Kd is None:
            self.Kd = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)


# Default PID gains for cart+double-pendulum stabilization
# These are tuned for the equilibrium point: x=0, dx=0, θ1=π, dθ1=0, θ2=π, dθ2=0
# State order: [x, dx, θ1, dθ1, θ2, dθ2]
DEFAULT_PID_CONFIG = PIDConfig(
    # Strong proportional gains for angles to pull pendulums upright
    # Small/zero gains for cart position (secondary objective)
    Kp=np.array([0.0, 0.0, 200.0, 0.0, 150.0, 0.0], dtype=np.float64),
    # Integral gains to eliminate steady-state error
    Ki=np.array([0.0, 0.0, 20.0, 0.0, 15.0, 0.0], dtype=np.float64),
    # Derivative gains for damping
    Kd=np.array([0.0, 50.0, 0.0, 30.0, 0.0, 25.0], dtype=np.float64),
    u_max=500.0,
    u_min=-500.0,
    integral_limit=10000.0,
)

# Alternative tuned gains (more aggressive)
AGGRESSIVE_PID_CONFIG = PIDConfig(
    Kp=np.array([0.0, 0.0, 300.0, 0.0, 250.0, 0.0], dtype=np.float64),
    Ki=np.array([0.0, 0.0, 50.0, 0.0, 40.0, 0.0], dtype=np.float64),
    Kd=np.array([0.0, 80.0, 0.0, 50.0, 0.0, 40.0], dtype=np.float64),
    u_max=1000.0,
    u_min=-1000.0,
    integral_limit=50000.0,
)

# Conservative gains (slower but more stable)
CONSERVATIVE_PID_CONFIG = PIDConfig(
    Kp=np.array([0.0, 0.0, 100.0, 0.0, 80.0, 0.0], dtype=np.float64),
    Ki=np.array([0.0, 0.0, 10.0, 0.0, 8.0, 0.0], dtype=np.float64),
    Kd=np.array([0.0, 30.0, 0.0, 20.0, 0.0, 15.0], dtype=np.float64),
    u_max=200.0,
    u_min=-200.0,
    integral_limit=5000.0,
)
