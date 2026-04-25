"""Physics environment for the cart+double-pendulum system."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

import physics_engine as phys
from .config import HoldConfig
from .reward import compute_reward


class PhysicsEnv:
    """Environment wrapper for the cart+double-pendulum physics simulation.
    
    This class provides a Gym-like interface for interacting with the physics
    simulation, including reset, step, and observation methods.
    """
    
    def __init__(self, p: phys.SystemParameters, cfg: HoldConfig) -> None:
        """Initialize the environment.
        
        Args:
            p: System parameters for the physics simulation.
            cfg: Configuration for the environment.
        """
        self.p = p
        self.cfg = cfg
        self._state: phys.State | None = None
        self._steps = 0
        self._prev_th1: float = 0.0
        self._prev_th2: float = 0.0
        self._total_steps: int = 0
    
    def reset(self, seed: int | None = None) -> np.ndarray:
        """Reset the environment to initial state.
        
        Args:
            seed: Optional random seed for reproducibility.
        
        Returns:
            Initial observation array.
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Random initial angles in [-0.5, 0.5] radians
        th1 = np.random.uniform(-0.5, 0.5)
        th2 = np.random.uniform(-0.5, 0.5)
        self._state = phys.State(
            pos=phys.Positions(x=np.float64(0.0), Teta1=np.float64(th1), Teta2=np.float64(th2)),
            vel=phys.Velocities(dx=np.float64(0.0), dTeta1=np.float64(0.0), dTeta2=np.float64(0.0)),
        )
        self._steps = 0
        self._total_steps = 0
        assert self._state is not None
        self._prev_th1 = float(self._state.pos.Teta1)
        self._prev_th2 = float(self._state.pos.Teta2)
        return self._obs(self._state)
    
    def _obs(self, s: phys.State) -> np.ndarray:
        """Create observation array from state.
        
        Args:
            s: Current state.
        
        Returns:
            Observation array [x, dx, th1, dth1, th2, dth2].
        """
        return np.array(
            [
                float(s.pos.x),
                float(s.vel.dx),
                float(s.pos.Teta1),
                float(s.vel.dTeta1),
                float(s.pos.Teta2),
                float(s.vel.dTeta2),
            ],
            dtype=np.float32,
        )
    
    def step(self, u: float) -> Tuple[np.ndarray, float, bool, dict]:
        """Execute one step in the environment.
        
        Args:
            u: Control force to apply.
        
        Returns:
            Tuple of (observation, reward, done, info).
        """
        assert self._state is not None
        self._steps += 1
        self._total_steps += 1
        
        # Clip control input
        u = float(np.clip(u, -self.cfg.u_limit, self.cfg.u_limit))
        # Fast integrator for original two-link dynamics.
        self._state = phys.rk4_step_fast(self._state, u=u, p=self.p, dt=self.cfg.dt)
        s = self._state
        
        prev_th1 = float(s.pos.Teta1)
        prev_th2 = float(s.pos.Teta2)
        
        # Check termination conditions
        out_of_bounds = abs(float(s.pos.x)) > self.cfg.x_limit
        
        # Adaptive fall threshold: curriculum learning
        adaptive_threshold = max(
            1.0,
            2.0 * (1.0 - min(1.0, self._total_steps / 50000.0)),
        )
        
        fallen = (
            abs(float(s.pos.Teta1)) > adaptive_threshold
            or abs(float(s.pos.Teta2)) > adaptive_threshold
        )
        
        state_arr = s.to_array()
        nan_state = bool(np.isnan(state_arr).any())
        done = bool(out_of_bounds or fallen or nan_state or self._steps >= self.cfg.max_steps)
        
        # Calculate rewards
        reward = self._calculate_reward(s, u, done, fallen)
        
        # Terminal penalty
        if done and (out_of_bounds or fallen or nan_state):
            reward -= 10.0
        
        # Save previous angles for next step
        self._prev_th1 = prev_th1
        self._prev_th2 = prev_th2
        
        obs = self._obs(s)
        info = {
            "fallen": fallen,
            "out_of_bounds": out_of_bounds,
            "nan_state": nan_state,
            "terminal_reason": (
                "out_of_bounds"
                if out_of_bounds
                else "fallen"
                if fallen
                else "nan_state"
                if nan_state
                else "max_steps"
                if self._steps >= self.cfg.max_steps
                else None
            ),
        }
        
        # Normalize reward (without hard clipping)
        reward = reward / 10.0
        return obs, reward, done, info
    
    def _calculate_reward(self, s: phys.State, u: float, done: bool, fallen: bool) -> float:
        """Calculate reward with shaping bonuses.
        
        Args:
            s: Current state.
            u: Control force.
            done: Whether the episode is done.
            fallen: Whether the pendulums have fallen.
        
        Returns:
            Total reward value.
        """
        # Base reward from state and control
        reward = compute_reward(s, u)
        
        # Alive bonus: small reward for staying alive
        alive_bonus = 0.15 if not done else 0.0
        
        # Swing complete bonus: reward for passing through the top
        th1_now = float(s.pos.Teta1)
        th2_now = float(s.pos.Teta2)
        swing_complete_bonus = 0.0
        if (self._prev_th1 < np.pi <= th1_now) or (self._prev_th1 > np.pi >= th1_now):
            swing_complete_bonus += 1.0
        if (self._prev_th2 < np.pi <= th2_now) or (self._prev_th2 > np.pi >= th2_now):
            swing_complete_bonus += 1.0
        swing_complete_bonus *= 50.0
        
        # Stabilization bonus: reward for holding upright position
        stabilization_bonus = 0.0
        if abs(th1_now - np.pi) < 0.2 and abs(th2_now - np.pi) < 0.2:
            stabilization_bonus = 1.0
        
        # Exploration bonus: encourage movement from bottom position
        exploration_bonus = 0.0
        if abs(th1_now) < 0.5 and abs(th2_now) < 0.5:
            exploration_bonus = 0.06 * abs(u)
        
        return reward + alive_bonus + swing_complete_bonus + stabilization_bonus + exploration_bonus
