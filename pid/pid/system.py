"""System models for cart+double-pendulum (nonlinear and linearized)."""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm, solve

from physics_engine import SystemParameters


class CartDoublePendulum:
    """Nonlinear model of cart+double-pendulum system.
    
    State vector: [x, dx, θ1, dθ1, θ2, dθ2]
    Input: u (force on cart)
    Output: y = x (full state observation)
    """
    
    def __init__(self, params: SystemParameters | None = None) -> None:
        """Initialize the system.
        
        Args:
            params: System parameters. If None, default parameters are used.
        """
        self.params = params or SystemParameters()
        self.n_state = 6
        self.n_input = 1
        self.n_output = 6
    
    def equations_of_motion(self, state: np.ndarray, u: float) -> np.ndarray:
        """Compute the equations of motion ẋ = f(x, u) using Lagrangian dynamics.
        
        Uses the standard formulation for cart+double-pendulum:
        M(q) * q_ddot + C(q, q_dot) * q_dot + G(q) = B * u
        
        Args:
            state: Current state [x, dx, θ1, dθ1, θ2, dθ2]
            u: Control force on the cart
        
        Returns:
            State derivative [dx, ddx, dθ1, ddθ1, dθ2, ddθ2]
        """
        x, dx, θ1, dθ1, θ2, dθ2 = state
        p = self.params
        
        # Trigonometric terms
        c1, s1 = np.cos(θ1), np.sin(θ1)
        c2, s2 = np.cos(θ2), np.sin(θ2)
        c12 = np.cos(θ1 - θ2)
        s12 = np.sin(θ1 - θ2)
        
        # Mass matrix M(q) - 3x3 for generalized coordinates [x, θ1, θ2]
        M = np.array([
            [p.m0 + p.m1 + p.m2, p.m1 * p.L1 * c1, p.m2 * p.L2 * c2],
            [p.m1 * p.L1 * c1, (p.m1 + p.m2) * p.L1**2, p.m2 * p.L1 * p.L2 * c12],
            [p.m2 * p.L2 * c2, p.m2 * p.L1 * p.L2 * c12, p.m2 * p.L2**2],
        ], dtype=np.float64)
        
        # Coriolis and centrifugal terms
        # C(q, q_dot) * q_dot
        # For the 3x3 system [x, θ1, θ2]
        
        # Force vector from Coriolis/centrifugal effects
        # F_cor = [F_x_cor, F_θ1_cor, F_θ2_cor]
        h = p.m2 * p.L1 * p.L2
        
        F_cor = np.array([
            h * dθ1 * dθ2 * s12,  # x direction
            -h * dθ1 * dθ2 * s12,  # θ1 direction
            h * dθ1 * dθ2 * s12,  # θ2 direction
        ], dtype=np.float64)
        
        # Gravity forces G(q)
        G = np.array([
            0,  # No gravity in x direction
            (p.m1 + p.m2) * p.g * p.L1 * s1,  # θ1 direction
            p.m2 * p.g * p.L2 * s2,  # θ2 direction
        ], dtype=np.float64)
        
        # Friction forces B * q_dot
        friction = np.array([
            p.b0 * dx,
            p.b1 * dθ1,
            p.b2 * dθ2,
        ], dtype=np.float64)
        
        # Control input matrix B (only affects x)
        B = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        
        # Right-hand side: B*u - F_cor - G - friction
        rhs = B * u - F_cor - G - friction
        
        # Solve M * q_ddot = rhs for q_ddot
        try:
            q_ddot = solve(M, rhs)
        except np.linalg.LinAlgError:
            # Fallback if matrix is singular
            q_ddot = np.zeros(3, dtype=np.float64)
        
        # Extract accelerations
        x_ddot, θ1_ddot, θ2_ddot = q_ddot
        
        return np.array([dx, x_ddot, dθ1, θ1_ddot, dθ2, θ2_ddot], dtype=np.float64)
    
    def step(self, state: np.ndarray, u: float, dt: float) -> np.ndarray:
        """Perform one integration step using RK4.
        
        Args:
            state: Current state
            u: Control force
            dt: Time step
        
        Returns:
            Next state
        """
        k1 = self.equations_of_motion(state, u)
        
        state2 = state + k1 * dt / 2
        k2 = self.equations_of_motion(state2, u)
        
        state3 = state + k2 * dt / 2
        k3 = self.equations_of_motion(state3, u)
        
        state4 = state + k3 * dt
        k4 = self.equations_of_motion(state4, u)
        
        return state + (k1 + 2*k2 + 2*k3 + k4) * dt / 6
    
    def linearize(self, equilibrium_state: np.ndarray | None = None) -> LinearizedSystem:
        """Linearize the system around an equilibrium point.
        
        Args:
            equilibrium_state: Equilibrium state [x, dx, θ1, dθ1, θ2, dθ2].
                              If None, uses the upright equilibrium.
        
        Returns:
            Linearized system (A, B, C, D matrices)
        """
        if equilibrium_state is None:
            # Upright equilibrium: cart at origin, pendulums pointing up (θ = π)
            equilibrium_state = np.array([
                0.0,  # x
                0.0,  # dx
                np.pi,  # θ1
                0.0,  # dθ1
                np.pi,  # θ2
                0.0,  # dθ2
            ], dtype=np.float64)
        
        return linearize_system(self, equilibrium_state, self.params)


def linearize_system(
    nonlinear_system: CartDoublePendulum,
    equilibrium_state: np.ndarray,
    params: SystemParameters,
    eps: float = 1e-6,
) -> LinearizedSystem:
    """Linearize a nonlinear system around an equilibrium point using finite differences.
    
    Args:
        nonlinear_system: The nonlinear system to linearize
        equilibrium_state: Equilibrium state point
        params: System parameters
        eps: Finite difference step size
    
    Returns:
        LinearizedSystem with matrices A, B, C, D
    """
    n = len(equilibrium_state)
    m = 1  # Single input
    
    # Equilibrium input (should be 0 for natural equilibrium)
    u_eq = 0.0
    
    # Compute f(x_eq, u_eq)
    f_eq = nonlinear_system.equations_of_motion(equilibrium_state, u_eq)
    
    # Compute A matrix (∂f/∂x)
    A = np.zeros((n, n), dtype=np.float64)
    
    for i in range(n):
        state_perturbed = equilibrium_state.copy()
        state_perturbed[i] += eps
        f_perturbed = nonlinear_system.equations_of_motion(state_perturbed, u_eq)
        A[:, i] = (f_perturbed - f_eq) / eps
    
    # Compute B matrix (∂f/∂u)
    B = np.zeros((n, m), dtype=np.float64)
    u_perturbed = u_eq + eps
    f_perturbed = nonlinear_system.equations_of_motion(equilibrium_state, u_perturbed)
    B[:, 0] = (f_perturbed - f_eq) / eps
    
    # C and D matrices (identity for full state observation)
    C = np.eye(n, dtype=np.float64)
    D = np.zeros((n, m), dtype=np.float64)
    
    return LinearizedSystem(A, B, C, D)


class LinearizedSystem:
    """Linear time-invariant (LTI) system representation.
    
    State-space form:
        ẋ = Ax + Bu
        y = Cx + Du
    
    Attributes:
        A: State matrix (n x n)
        B: Input matrix (n x m)
        C: Output matrix (p x n)
        D: Feedthrough matrix (p x m)
    """
    
    def __init__(self, A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray) -> None:
        """Initialize the linear system.
        
        Args:
            A: State matrix
            B: Input matrix
            C: Output matrix
            D: Feedthrough matrix
        """
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.n_state = A.shape[0]
        self.n_input = B.shape[1]
        self.n_output = C.shape[0]
    
    def step(self, state: np.ndarray, u: float, dt: float) -> np.ndarray:
        """Perform one discrete-time step.
        
        Uses exact discretization: x[k+1] = Φ*x[k] + Γ*u[k]
        where Φ = exp(A*dt) and Γ = ∫₀ᵈᵗ exp(A*τ)dτ * B
        
        Args:
            state: Current state
            u: Input
            dt: Time step
        
        Returns:
            Next state
        """
        # Discretize using matrix exponential
        Phi = expm(self.A * dt)
        
        # Compute Gamma = ∫₀ᵈᵗ exp(A*τ)dτ * B
        if np.allclose(self.A, 0):
            Gamma = self.B * dt
        else:
            try:
                A_inv = np.linalg.inv(self.A)
                Gamma = A_inv @ (Phi - np.eye(self.n_state)) @ self.B
            except np.linalg.LinAlgError:
                Gamma = self.B * dt
        
        Gamma = Gamma.flatten()
        
        return Phi @ state + Gamma * u
    
    def continuous_step(self, state: np.ndarray, u: float, dt: float) -> np.ndarray:
        """Perform one continuous-time step using Euler integration.
        
        Args:
            state: Current state
            u: Input
            dt: Time step
        
        Returns:
            Next state
        """
        dx = self.A @ state + self.B * u
        return state + dx * dt
    
    def get_matrices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return the system matrices.
        
        Returns:
            Tuple (A, B, C, D)
        """
        return self.A, self.B, self.C, self.D
