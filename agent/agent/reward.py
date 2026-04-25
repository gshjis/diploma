"""Reward function for the cart+double-pendulum environment."""

import numpy as np

import physics_engine as phys


def compute_reward(s: phys.State, u: float = 0.0) -> float:
    """Compute reward based on state and control action.
    
    The reward encourages raising the pendulums to the upright position
    while penalizing high velocities and cart displacement.
    
    Args:
        s: Current state of the system.
        u: Control force applied.
    
    Returns:
        Reward value (higher is better).
    """
    th1 = float(s.pos.Teta1)
    th2 = float(s.pos.Teta2)
    dth1 = float(s.vel.dTeta1)
    dth2 = float(s.vel.dTeta2)
    x = float(s.pos.x)
    
    # Direct reward for height (maximizing potential energy)
    # 0 at bottom, 40 at top (for both pendulums)
    angle_reward = 10.0 * ((1.0 - np.cos(th1)) + (1.0 - np.cos(th2)))
    
    # Penalty for angular velocity
    vel_penalty = 0.005 * (dth1 * dth1 + dth2 * dth2)
    
    # Penalty for cart displacement
    x_penalty = 0.5 * x * x
    
    # Return positive reward for height minus penalties
    return angle_reward - vel_penalty - x_penalty
