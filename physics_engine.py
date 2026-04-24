from dataclasses import dataclass
from typing import Callable

import numpy as np

@dataclass
class Positions:
    x: np.float64 = np.float64(0.0)
    Teta1: np.float64 = np.float64(0.5)
    Teta2: np.float64 = np.float64(0.5)

@dataclass
class Velocities:
    dx: np.float64 = np.float64(0.0)
    dTeta1: np.float64 = np.float64(0.0)
    dTeta2: np.float64 = np.float64(0.0)

@dataclass
class SystemParameters:
    g: np.float64 = np.float64(-9.81)
    m0: np.float64 = np.float64(5.0)
    m1: np.float64 = np.float64(0.5)
    m2: np.float64 = np.float64(0.5)
    L1: np.float64 = np.float64(0.25)
    L2: np.float64 = np.float64(0.25)
    b0: np.float64 = np.float64(0)
    b1: np.float64 = np.float64(0.01)
    b2: np.float64 = np.float64(0.01)

@dataclass
class State:
    """Полное состояние системы"""
    pos: Positions
    vel: Velocities
    
    def to_array(self) -> np.ndarray:
        """Преобразовать в numpy массив [x, dx, θ1, dθ1, θ2, dθ2]"""
        # Важно для скорости: один np.array без лишних промежуточных списков.
        # (Список всё равно создаётся, но мы избегаем dtype-переопределений в вызовах ниже.)
        return np.array(
            (
                self.pos.x,
                self.vel.dx,
                self.pos.Teta1,
                self.vel.dTeta1,
                self.pos.Teta2,
                self.vel.dTeta2,
            ),
            dtype=np.float64,
        )
    
    @staticmethod
    def from_array(arr: np.ndarray) -> "State":
        """Восстановить состояние из массива"""
        return State(
            pos=Positions(x=arr[0], Teta1=arr[2], Teta2=arr[4]),
            vel=Velocities(dx=arr[1], dTeta1=arr[3], dTeta2=arr[5])
        )


def equations_of_motion(state: State, u: float, p: SystemParameters) -> np.ndarray:
    """
    Вычисляет производные состояния ẋ = [dx, ddx, dθ1, ddθ1, dθ2, ddθ2]
    
    Вход:
        state - текущее состояние (позиции и скорости)
        u - сила F, приложенная к тележке
        p - параметры системы
    
    Выход:
        массив производных [dx, ddx, dθ1, ddθ1, dθ2, ddθ2]
    """
    # Распаковываем координаты и скорости
    x = state.pos.x
    θ1 = state.pos.Teta1
    θ2 = state.pos.Teta2
    dx = state.vel.dx
    dθ1 = state.vel.dTeta1
    dθ2 = state.vel.dTeta2
    
    # Параметры
    g = p.g
    m0, m1, m2 = p.m0, p.m1, p.m2
    L1, L2 = p.L1, p.L2
    b0, b1, b2 = p.b0, p.b1, p.b2
    
    # Вспомогательные величины
    s1 = np.sin(θ1)
    s2 = np.sin(θ2)
    c1 = np.cos(θ1)
    c2 = np.cos(θ2)
    c12 = np.cos(θ1 - θ2)
    s12 = np.sin(θ1 - θ2)
    
    # Предвычисления для скорости (меньше pow и лишних операций)
    L1_2 = L1 * L1
    L2_2 = L2 * L2

    # Эффективные моменты инерции для точечных масс
    I1_eff = m1 * L1_2
    I2_eff = m2 * L2_2
    
    # Матрица инерции M(q) для q = [x, θ1, θ2]
    M11 = m0 + m1 + m2
    M12 = (m1 + m2) * L1 * c1
    M13 = m2 * L2 * c2
    M22 = I1_eff + (m1 + m2) * L1_2
    M23 = m2 * L1 * L2 * c12
    M33 = I2_eff + m2 * L2_2
    
    M = np.array([
        [M11, M12, M13],
        [M12, M22, M23],
        [M13, M23, M33]
    ], dtype=np.float64)
    
    # Вектор кориолисовых/центробежных сил C(q, dq)
    C1 = -(m1 + m2) * L1 * dθ1**2 * s1 - m2 * L2 * dθ2**2 * s2
    C2 = m2 * L1 * L2 * dθ2**2 * s12
    C3 = -m2 * L1 * L2 * dθ1**2 * s12
    
    C = np.array([C1, C2, C3], dtype=np.float64)
    
    # Вектор гравитационных сил G(q)
    G1 = 0.0
    # Инвертируем гравитацию так, чтобы θ=0 стало "верхом".
    # Иначе при θ=0 (sin(0)=0) модель соответствует низу.
    G2 = -(m1 + m2) * g * L1 * s1
    G3 = -m2 * g * L2 * s2
    
    G = np.array([G1, G2, G3], dtype=np.float64)
    
    # Обобщённые силы Q (трение + управление)
    Q1 = u - b0 * dx
    Q2 = -b1 * dθ1
    Q3 = -b2 * dθ2
    
    Q = np.array([Q1, Q2, Q3], dtype=np.float64)
    
    # Решаем M * d²q = Q - C - G
    rhs = Q - C - G
    # Быстрее, чем общий solve для 3x3: используем explicit inverse через inv
    # (для маленьких матриц часто быстрее, но сохраняем читаемость)
    ddq = np.linalg.solve(M, rhs)  # d²q = [ddx, ddθ1, ddθ2]
    
    # Собираем ẋ = [dx, ddx, dθ1, ddθ1, dθ2, ddθ2]
    x_dot = np.array(
        (dx, ddq[0], dθ1, ddq[1], dθ2, ddq[2]),
        dtype=np.float64,
    )
    
    return x_dot


def rk4_step(
    state: State,
    u: float,
    p: SystemParameters,
    dt: float,
    x_min: float | None = None,
    x_max: float | None = None,
) -> State:
    """
    Один шаг интегрирования методом Рунге-Кутты 4-го порядка
    """
    x0 = state.to_array()
    
    def f(x: np.ndarray) -> np.ndarray:
        s = State.from_array(x)
        return equations_of_motion(s, u, p)
    
    half_dt = 0.5 * dt
    sixth_dt = dt / 6.0

    k1 = f(x0)
    k2 = f(x0 + half_dt * k1)
    k3 = f(x0 + half_dt * k2)
    k4 = f(x0 + dt * k3)
    
    x_new = x0 + sixth_dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    # Жёсткое ограничение положения x (скорости dx не трогаем)
    if x_min is not None:
        x_new[0] = max(float(x_min), float(x_new[0]))
    if x_max is not None:
        x_new[0] = min(float(x_max), float(x_new[0]))

    return State.from_array(x_new)


def simulate(
    p: SystemParameters,
    state0: State,
    u_func: Callable[[float, State], float],
    dt: float,
    t_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Запуск симуляции
    
    Вход:
        p - параметры системы
        state0 - начальное состояние
        u_func - функция u = u_func(t, state) для вычисления управления
        dt - шаг по времени
        t_max - время симуляции
    
    Выход:
        массивы времени и истории состояний
    """
    n_steps = int(t_max / dt) + 1
    time = np.linspace(0, t_max, n_steps, dtype=np.float64)

    # Предвыделяем буфер под историю (экономия на аллокациях)
    history = np.empty((n_steps, 6), dtype=np.float64)

    state = state0
    for i, t in enumerate(time):
        history[i, :] = state.to_array()
        u = u_func(float(t), state)
        state = rk4_step(state, u, p, dt)

    return time, history


# ========== Пример использования ==========
if __name__ == "__main__":
    # Параметры
    p = SystemParameters()
    
    # Начальное состояние: маятники отклонены на 0.1 рад, тележка в нуле
    state0 = State(
        pos=Positions(
            x=np.float64(0.0),
            Teta1=np.float64(0.1),
            Teta2=np.float64(0.1),
        ),
        vel=Velocities(
            dx=np.float64(0.0),
            dTeta1=np.float64(0.0),
            dTeta2=np.float64(0.0),
        ),
    )
    
    # Управление: просто 0 (свободное падение)
    def u_zero(t: float, state: State) -> float:
        return 0.0
    
    # Запуск
    dt = 0.001
    t_max = 5.0
    time, history = simulate(p, state0, u_zero, dt, t_max)
    
    print(f"Симуляция завершена. Сохранено {len(time)} шагов.")
    print(f"Последнее состояние: {history[-1]}")
