"""Feedback components: summator and unity feedback loop."""

from __future__ import annotations

import numpy as np


class Summator:
    """Summator (adder) for control systems.
    
    Computes the sum or difference of multiple input signals.
    Commonly used to compute error: e = r - y (reference - output)
    
    Attributes:
        signs: List of signs for each input (+1 or -1).
    """
    
    def __init__(self, signs: list[int] | None = None) -> None:
        """Initialize the summator.
        
        Args:
            signs: List of signs for each input. Default is [+1, -1] for e = r - y.
        """
        if signs is None:
            # Default: e = r - y (reference minus output)
            self.signs = np.array([1, -1], dtype=np.float64)
        else:
            self.signs = np.array(signs, dtype=np.float64)
    
    def compute(self, *inputs: np.ndarray) -> np.ndarray:
        """Compute the weighted sum of inputs.
        
        Args:
            *inputs: Input arrays to sum. Must have compatible shapes.
        
        Returns:
            Weighted sum of inputs.
        
        Example:
            >>> summator = Summator([1, -1])
            >>> error = summator.compute(reference, output)
        """
        if len(inputs) != len(self.signs):
            raise ValueError(
                f"Number of inputs ({len(inputs)}) must match number of signs ({len(self.signs)})"
            )
        
        result = np.zeros_like(inputs[0])
        for sign, inp in zip(self.signs, inputs):
            result += sign * inp
        
        return result
    
    def compute_error(self, reference: np.ndarray, output: np.ndarray) -> np.ndarray:
        """Compute error signal e = r - y.
        
        Args:
            reference: Reference signal (setpoint)
            output: Measured output
        
        Returns:
            Error signal e = reference - output
        """
        return self.compute(reference, output)


class UnityFeedback:
    """Unity feedback loop wrapper.
    
    Implements a standard feedback control structure:
    
        ┌─────────┐     ┌─────────┐     ┌─────────┐
        │  Plant  │────▶│  Output │────▶│         │
        └─────────┘     └─────────┘     │         │
                ▲                       ▼         │
                │              ┌─────────┐        │
                │              │ Summator│        │
                │              └─────────┘        │
                │                   │             │
                │                   ▼             │
        ┌───────┴────────┐  ┌─────────┐          │
        │   Controller   │◀─│  Error  │◀─────────┘
        └────────────────┘  └─────────┘
                │
                ▼
               u (control)
    
    Attributes:
        controller: The controller to use.
        summator: The summator for computing error.
    """
    
    def __init__(
        self,
        controller,
        summator: Summator | None = None,
    ) -> None:
        """Initialize the unity feedback system.
        
        Args:
            controller: Controller instance with a compute method.
            summator: Summator for error computation. If None, uses default.
        """
        self.controller = controller
        self.summator = summator or Summator([1, -1])
        
        # History for plotting/debugging
        self.reference_history: list[np.ndarray] = []
        self.output_history: list[np.ndarray] = []
        self.error_history: list[np.ndarray] = []
        self.control_history: list[float] = []
    
    def reset(self) -> None:
        """Reset the feedback system state."""
        self.controller.reset()
        self.reference_history.clear()
        self.output_history.clear()
        self.error_history.clear()
        self.control_history.clear()
    
    def step(
        self,
        reference: np.ndarray,
        output: np.ndarray,
        dt: float,
    ) -> float:
        """Perform one control step.
        
        Args:
            reference: Reference signal (setpoint) [n_state x 1]
            output: Current output measurement [n_state x 1]
            dt: Time step
        
        Returns:
            Control action u (scalar)
        """
        # Compute error: e = r - y
        error = self.summator.compute_error(reference, output)
        
        # Compute control action
        u = self.controller.compute(reference, output, dt)
        
        # Record history
        self.reference_history.append(reference.copy())
        self.output_history.append(output.copy())
        self.error_history.append(error.copy())
        self.control_history.append(u)
        
        return u
    
    def step_with_filter(
        self,
        reference: np.ndarray,
        output: np.ndarray,
        dt: float,
        filter_time_constant: float = 0.01,
    ) -> float:
        """Perform one control step with derivative filtering.
        
        Args:
            reference: Reference signal [n_state x 1]
            output: Current output measurement [n_state x 1]
            dt: Time step
            filter_time_constant: Filter time constant for derivative term
        
        Returns:
            Control action u (scalar)
        """
        # Compute error
        error = self.summator.compute_error(reference, output)
        
        # Compute control action with filter
        u = self.controller.compute_with_derivative_filter(
            reference, output, dt, filter_time_constant
        )
        
        # Record history
        self.reference_history.append(reference.copy())
        self.output_history.append(output.copy())
        self.error_history.append(error.copy())
        self.control_history.append(u)
        
        return u
    
    def get_history(self) -> dict[str, list]:
        """Get the history of signals.
        
        Returns:
            Dictionary with keys: 'reference', 'output', 'error', 'control'
        """
        return {
            'reference': self.reference_history,
            'output': self.output_history,
            'error': self.error_history,
            'control': self.control_history,
        }
