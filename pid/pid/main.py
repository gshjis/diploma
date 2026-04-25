"""Main entry point for testing the PID-controlled cart+double-pendulum system."""

from __future__ import annotations

import numpy as np

from config import PIDConfig, AGGRESSIVE_PID_CONFIG
from physics_engine import SystemParameters
from system import CartDoublePendulum, LinearizedSystem
from pid_controller import PIDController
from feedback import Summator, UnityFeedback


def create_stabilization_system(
    params: SystemParameters | None = None,
    config: PIDConfig | None = None,
) -> tuple[LinearizedSystem, PIDController, UnityFeedback]:
    """Create a complete PID-controlled linearized system.
    
    Args:
        params: System parameters. If None, uses defaults.
        config: PID configuration. If None, uses AGGRESSIVE_PID_CONFIG.
    
    Returns:
        Tuple of (linear_system, controller, feedback_system)
    """
    # Create system
    system_params = params or SystemParameters()
    nonlinear_system = CartDoublePendulum(system_params)
    
    # Linearize around upright equilibrium
    equilibrium_state = np.array([
        0.0,  # x
        0.0,  # dx
        np.pi,  # θ1
        0.0,  # dθ1
        np.pi,  # θ2
        0.0,  # dθ2
    ], dtype=np.float64)
    
    linear_system = nonlinear_system.linearize(equilibrium_state)
    
    # Create controller
    controller = PIDController(config)
    
    # Create feedback system
    feedback = UnityFeedback(controller)
    
    return linear_system, controller, feedback


def run_simulation(
    dt: float = 0.01,
    duration: float = 10.0,
    initial_state: np.ndarray | None = None,
    reference: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Run a simulation of the PID-controlled system.
    
    Args:
        dt: Time step
        duration: Simulation duration
        initial_state: Initial state. If None, starts near equilibrium.
        reference: Reference signal. If None, uses equilibrium.
    
    Returns:
        Dictionary with simulation results:
            - 'time': Time array
            - 'state': State history
            - 'control': Control input history
            - 'error': Error history
    """
    # Create system
    linear_system, controller, feedback = create_stabilization_system()
    
    # Default reference: equilibrium point
    if reference is None:
        reference = np.array([
            0.0,  # x
            0.0,  # dx
            np.pi,  # θ1
            0.0,  # dθ1
            np.pi,  # θ2
            0.0,  # dθ2
        ], dtype=np.float64)
    
    # Default initial state: small perturbation from equilibrium
    if initial_state is None:
        initial_state = np.array([
            0.1,   # x (small displacement)
            0.0,   # dx
            np.pi + 0.1,  # θ1 (small angle)
            0.0,   # dθ1
            np.pi + 0.1,  # θ2 (small angle)
            0.0,   # dθ2
        ], dtype=np.float64)
    
    # Initialize
    state = initial_state.copy()
    feedback.reset()
    
    # Storage
    n_steps = int(duration / dt)
    time_history = np.zeros(n_steps, dtype=np.float64)
    state_history = np.zeros((n_steps, 6), dtype=np.float64)
    control_history = np.zeros(n_steps, dtype=np.float64)
    error_history = np.zeros((n_steps, 6), dtype=np.float64)
    
    # Run simulation
    for i in range(n_steps):
        time_history[i] = i * dt
        state_history[i] = state
        
        # Compute control action
        u = feedback.step(reference, state, dt)
        control_history[i] = u
        
        # Get error from feedback
        if feedback.error_history:
            error_history[i] = feedback.error_history[-1]
        
        # Step the system
        state = linear_system.step(state, u, dt)
    
    return {
        'time': time_history,
        'state': state_history,
        'control': control_history,
        'error': error_history,
    }


def main() -> None:
    """Run a demonstration simulation."""
    print("PID Controller for Cart+Double-Pendulum")
    print("=" * 50)
    
    # Run simulation
    results = run_simulation(dt=0.01, duration=10.0)
    
    # Print summary
    print(f"\nSimulation completed:")
    print(f"  Duration: {results['time'][-1]:.2f} s")
    print(f"  Initial state: {results['state'][0]}")
    print(f"  Final state: {results['state'][-1]}")
    print(f"  Max control: {np.max(np.abs(results['control'])):.2f} N")
    print(f"  Final error norm: {np.linalg.norm(results['error'][-1]):.4f}")
    
    # Check if stabilized
    final_error = np.linalg.norm(results['error'][-1])
    if final_error < 0.1:
        print("\n✓ System stabilized successfully!")
    else:
        print("\n✗ System did not fully stabilize.")
        print("  Consider tuning PID gains.")


if __name__ == "__main__":
    main()
