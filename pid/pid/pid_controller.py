"""PID controller implementation for multivariable systems."""

from __future__ import annotations

import numpy as np

from config import PIDConfig


class PIDController:
    """Multivariable PID controller with anti-windup.
    
    This controller implements a PID algorithm for systems with multiple
    state variables. The control law is:
    
        u = -Kp * e - Ki * ∫e dt - Kd * de/dt
    
    where e = r - y is the error between reference and output.
    
    Attributes:
        config: PID configuration parameters.
        integral_state: Current integral state for each output.
        previous_output: Previous output for derivative computation.
    """
    
    def __init__(self, config: PIDConfig | None = None) -> None:
        """Initialize the PID controller.
        
        Args:
            config: PID configuration. If None, uses DEFAULT_PID_CONFIG.
        """
        from config import DEFAULT_PID_CONFIG
        self.config = config or DEFAULT_PID_CONFIG
        self.n_state = 6  # Number of state variables
        
        # Initialize integral state
        self.integral_state = np.zeros(self.n_state, dtype=np.float64)
        
        # For derivative computation (backwards Euler)
        self.previous_output: np.ndarray | None = None
        
        # For derivative computation (error derivative)
        self.previous_error: np.ndarray | None = None
    
    def reset(self) -> None:
        """Reset the controller state (clears integral and history)."""
        self.integral_state = np.zeros(self.n_state, dtype=np.float64)
        self.previous_output = None
        self.previous_error = None
    
    def compute(self, reference: np.ndarray, output: np.ndarray, dt: float) -> float:
        """Compute control action using PID algorithm.
        
        Args:
            reference: Reference signal (setpoint) [n_state x 1]
            output: Current output measurement [n_state x 1]
            dt: Time step
        
        Returns:
            Control action u (scalar force)
        """
        # Compute error
        error = reference - output
        
        # Update integral (trapezoidal integration)
        if self.previous_error is not None:
            integral_update = (error + self.previous_error) * dt / 2
        else:
            integral_update = error * dt
        
        self.integral_state += integral_update
        
        # Anti-windup: clamp integral
        self.integral_state = np.clip(
            self.integral_state,
            -self.config.integral_limit,
            self.config.integral_limit
        )
        
        # Compute derivative (backwards Euler)
        if self.previous_output is not None:
            derivative = (output - self.previous_output) / dt
        else:
            derivative = np.zeros(self.n_state, dtype=np.float64)
        
        # PID computation: u = -Kp*e - Ki*∫e - Kd*de/dt
        # Note: negative sign because we're stabilizing around equilibrium
        u = (
            -self.config.Kp @ error
            -self.config.Ki @ self.integral_state
            -self.config.Kd @ derivative
        )
        
        # Update history
        self.previous_error = error.copy()
        self.previous_output = output.copy()
        
        # Clamp output
        u = np.clip(u, self.config.u_min, self.config.u_max)
        
        return float(u)
    
    def compute_with_derivative_filter(
        self,
        reference: np.ndarray,
        output: np.ndarray,
        dt: float,
        filter_time_constant: float = 0.01,
    ) -> float:
        """Compute control action with derivative filter.
        
        The derivative term is filtered to reduce noise sensitivity:
            τ * dy_f/dt + y_f = y
        
        Args:
            reference: Reference signal [n_state x 1]
            output: Current output measurement [n_state x 1]
            dt: Time step
            filter_time_constant: Filter time constant τ
        
        Returns:
            Control action u (scalar force)
        """
        # Compute error
        error = reference - output
        
        # Update integral
        if self.previous_error is not None:
            integral_update = (error + self.previous_error) * dt / 2
        else:
            integral_update = error * dt
        
        self.integral_state += integral_update
        self.integral_state = np.clip(
            self.integral_state,
            -self.config.integral_limit,
            self.config.integral_limit
        )
        
        # Apply derivative filter
        if self.previous_output is not None:
            # Discrete filter: y_f[k] = (τ/(τ+dt)) * y_f[k-1] + (dt/(τ+dt)) * y[k]
            alpha = filter_time_constant / (filter_time_constant + dt)
            filtered_derivative = (output - self.previous_output) / dt
            # Simple first-order filter on derivative
            if hasattr(self, 'filtered_derivative'):
                filtered_derivative = alpha * self.filtered_derivative + (1 - alpha) * filtered_derivative
            self.filtered_derivative = filtered_derivative
        else:
            filtered_derivative = np.zeros(self.n_state, dtype=np.float64)
        
        # PID computation
        u = (
            -self.config.Kp @ error
            -self.config.Ki @ self.integral_state
            -self.config.Kd @ filtered_derivative
        )
        
        # Update history
        self.previous_error = error.copy()
        self.previous_output = output.copy()
        
        # Clamp output
        u = np.clip(u, self.config.u_min, self.config.u_max)
        
        return float(u)
