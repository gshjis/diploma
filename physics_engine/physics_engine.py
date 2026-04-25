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
class OneLinkState:
    """Состояние cart+один маятник.

    Состояние упаковано как (x, dx, θ, dθ) для простоты интегратора.
    """

    x: np.float64 = np.float64(0.0)
    dx: np.float64 = np.float64(0.0)
    theta: np.float64 = np.float64(0.5)
    dtheta: np.float64 = np.float64(0.0)

@dataclass
class SystemParameters:
    g: np.float64 = np.float64(-1)
    m0: np.float64 = np.float64(5)
    m1: np.float64 = np.float64(1)
    m2: np.float64 = np.float64(1)
    L1: np.float64 = np.float64(0.25)
    L2: np.float64 = np.float64(0.25)
    b0: np.float64 = np.float64(0.01)
    b1: np.float64 = np.float64(0.1)
    b2: np.float64 = np.float64(0.1)

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


def equations_of_motion_fast(
    x: float,
    dx: float,
    th1: float,
    dth1: float,
    th2: float,
    dth2: float,
    u: float,
    p: SystemParameters,
) -> tuple[float, float, float, float, float, float]:
    """Быстрый вариант уравнений движения без аллокаций numpy.

    Возвращает производные: (dx, ddx, dth1, ddth1, dth2, ddth2).
    """

    g = p.g
    m0, m1, m2 = p.m0, p.m1, p.m2
    L1, L2 = p.L1, p.L2
    b0, b1, b2 = p.b0, p.b1, p.b2

    s1 = float(np.sin(th1))
    s2 = float(np.sin(th2))
    c1 = float(np.cos(th1))
    c2 = float(np.cos(th2))
    c12 = float(np.cos(th1 - th2))
    s12 = float(np.sin(th1 - th2))

    L1_2 = L1 * L1
    L2_2 = L2 * L2
    I1_eff = m1 * L1_2
    I2_eff = m2 * L2_2

    # Матрица M(q) для q=[x, th1, th2]
    M11 = m0 + m1 + m2
    M12 = (m1 + m2) * L1 * c1
    M13 = m2 * L2 * c2
    M22 = I1_eff + (m1 + m2) * L1_2
    M23 = m2 * L1 * L2 * c12
    M33 = I2_eff + m2 * L2_2

    # C(q,dq)
    C1 = -(m1 + m2) * L1 * dth1 * dth1 * s1 - m2 * L2 * dth2 * dth2 * s2
    C2 = m2 * L1 * L2 * dth2 * dth2 * s12
    C3 = -m2 * L1 * L2 * dth1 * dth1 * s12

    # G(q)
    G1 = 0.0
    G2 = -(m1 + m2) * g * L1 * s1
    G3 = -m2 * g * L2 * s2

    # Q (управление + трение)
    Q1 = u - b0 * dx
    Q2 = -b1 * dth1
    Q3 = -b2 * dth2

    # rhs = Q - C - G
    rhs1 = Q1 - C1 - G1
    rhs2 = Q2 - C2 - G2
    rhs3 = Q3 - C3 - G3

    # Решаем M * [ddx, ddth1, ddth2] = rhs для симметричной 3x3.
    # Gaussian elimination (inline), без сборки массивов.
    a11, a12, a13 = M11, M12, M13
    a21, a22, a23 = M12, M22, M23
    a31, a32, a33 = M13, M23, M33
    b1v, b2v, b3v = rhs1, rhs2, rhs3

    # Forward elimination
    # Row2 -= m21*Row1
    m21 = a21 / a11
    a22 -= m21 * a12
    a23 -= m21 * a13
    b2v -= m21 * b1v

    # Row3 -= m31*Row1
    m31 = a31 / a11
    a32 = a32 - m31 * a12
    a33 = a33 - m31 * a13
    b3v = b3v - m31 * b1v

    # Row3 -= m32*Row2
    m32 = a32 / a22
    a33 -= m32 * a23
    b3v -= m32 * b2v

    # Back substitution
    ddth2 = b3v / a33
    ddth1 = (b2v - a23 * ddth2) / a22
    ddx = (b1v - a12 * ddth1 - a13 * ddth2) / a11

    return dx, ddx, dth1, ddth1, dth2, ddth2


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


def rk4_step_fast(
    state: State,
    u: float,
    p: SystemParameters,
    dt: float,
    x_min: float | None = None,
    x_max: float | None = None,
) -> State:
    """RK4 без `State`/`ndarray` упаковок внутри подшагов."""

    x0 = state.pos.x
    dx0 = state.vel.dx
    th1_0 = state.pos.Teta1
    dth1_0 = state.vel.dTeta1
    th2_0 = state.pos.Teta2
    dth2_0 = state.vel.dTeta2

    half_dt = 0.5 * dt
    sixth_dt = dt / 6.0

    # k1
    k1 = equations_of_motion_fast(x0, dx0, th1_0, dth1_0, th2_0, dth2_0, u, p)

    # k2
    x1 = x0 + half_dt * k1[0]
    dx1 = dx0 + half_dt * k1[1]
    th1_1 = th1_0 + half_dt * k1[2]
    dth1_1 = dth1_0 + half_dt * k1[3]
    th2_1 = th2_0 + half_dt * k1[4]
    dth2_1 = dth2_0 + half_dt * k1[5]
    k2 = equations_of_motion_fast(x1, dx1, th1_1, dth1_1, th2_1, dth2_1, u, p)

    # k3
    x2 = x0 + half_dt * k2[0]
    dx2 = dx0 + half_dt * k2[1]
    th1_2 = th1_0 + half_dt * k2[2]
    dth1_2 = dth1_0 + half_dt * k2[3]
    th2_2 = th2_0 + half_dt * k2[4]
    dth2_2 = dth2_0 + half_dt * k2[5]
    k3 = equations_of_motion_fast(x2, dx2, th1_2, dth1_2, th2_2, dth2_2, u, p)

    # k4
    x3 = x0 + dt * k3[0]
    dx3 = dx0 + dt * k3[1]
    th1_3 = th1_0 + dt * k3[2]
    dth1_3 = dth1_0 + dt * k3[3]
    th2_3 = th2_0 + dt * k3[4]
    dth2_3 = dth2_0 + dt * k3[5]
    k4 = equations_of_motion_fast(x3, dx3, th1_3, dth1_3, th2_3, dth2_3, u, p)

    x_new = x0 + sixth_dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0])
    dx_new = dx0 + sixth_dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])
    th1_new = th1_0 + sixth_dt * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2])
    dth1_new = dth1_0 + sixth_dt * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3])
    th2_new = th2_0 + sixth_dt * (k1[4] + 2.0 * k2[4] + 2.0 * k3[4] + k4[4])
    dth2_new = dth2_0 + sixth_dt * (k1[5] + 2.0 * k2[5] + 2.0 * k3[5] + k4[5])

    if x_min is not None:
        x_new = max(float(x_min), float(x_new))
    if x_max is not None:
        x_new = min(float(x_max), float(x_new))

    return State(
        pos=Positions(x=np.float64(x_new), Teta1=np.float64(th1_new), Teta2=np.float64(th2_new)),
        vel=Velocities(dx=np.float64(dx_new), dTeta1=np.float64(dth1_new), dTeta2=np.float64(dth2_new)),
    )


def one_link_equations_of_motion(state: OneLinkState, u: float, p: SystemParameters) -> tuple[float, float, float, float]:
    """Динамика cart+один маятник.

    Возвращает производные (dx, ddx, dθ, ddθ).
    Здесь используем параметризацию из исходной двухмаятниковой модели,
    но редуцируем второе плечо (θ2 = 0, dθ2 = 0, m2=b2=L2=0).
    """

    x = float(state.x)
    dx = float(state.dx)
    th = float(state.theta)
    dth = float(state.dtheta)

    g = float(p.g)
    m0 = float(p.m0)
    m1 = float(p.m1)
    L1 = float(p.L1)
    b0 = float(p.b0)
    b1 = float(p.b1)

    # Для вывода используем те же структуры М, C, G, Q что в equations_of_motion,
    # но для q=[x, θ] (исключаем третий dof).
    s = float(np.sin(th))
    c = float(np.cos(th))

    I1_eff = m1 * (L1 * L1)
    M11 = m0 + m1
    M12 = m1 * L1 * c
    M22 = I1_eff

    # rhs = Q - C - G
    # C1, C2 для q=[x, θ]
    C1 = -m1 * L1 * dth * dth * s
    C2 = 0.0

    G1 = 0.0
    G2 = -m1 * g * L1 * s

    Q1 = u - b0 * dx
    Q2 = -b1 * dth

    rhs1 = Q1 - C1 - G1
    rhs2 = Q2 - C2 - G2

    # Solve 2x2:
    # [M11 M12; M12 M22] [ddx; ddth] = [rhs1; rhs2]
    det = M11 * M22 - M12 * M12
    if det == 0.0:
        det = 1e-12
    ddx = (rhs1 * M22 - M12 * rhs2) / det
    ddth = (M11 * rhs2 - M12 * rhs1) / det

    return dx, ddx, dth, ddth


def one_link_rk4_step(state: OneLinkState, u: float, p: SystemParameters, dt: float) -> OneLinkState:
    x0 = float(state.x)
    dx0 = float(state.dx)
    th0 = float(state.theta)
    dth0 = float(state.dtheta)

    half_dt = 0.5 * dt
    sixth_dt = dt / 6.0

    def f(x: float, dx: float, th: float, dth: float):
        s = OneLinkState(x=np.float64(x), dx=np.float64(dx), theta=np.float64(th), dtheta=np.float64(dth))
        return one_link_equations_of_motion(s, u=u, p=p)

    k1_x, k1_dx, k1_th, k1_dth = f(x0, dx0, th0, dth0)
    k2_x, k2_dx, k2_th, k2_dth = f(x0 + half_dt * k1_x, dx0 + half_dt * k1_dx, th0 + half_dt * k1_th, dth0 + half_dt * k1_dth)
    k3_x, k3_dx, k3_th, k3_dth = f(x0 + half_dt * k2_x, dx0 + half_dt * k2_dx, th0 + half_dt * k2_th, dth0 + half_dt * k2_dth)
    k4_x, k4_dx, k4_th, k4_dth = f(x0 + dt * k3_x, dx0 + dt * k3_dx, th0 + dt * k3_th, dth0 + dt * k3_dth)

    x_new = x0 + sixth_dt * (k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x)
    dx_new = dx0 + sixth_dt * (k1_dx + 2.0 * k2_dx + 2.0 * k3_dx + k4_dx)
    th_new = th0 + sixth_dt * (k1_th + 2.0 * k2_th + 2.0 * k3_th + k4_th)
    dth_new = dth0 + sixth_dt * (k1_dth + 2.0 * k2_dth + 2.0 * k3_dth + k4_dth)

    return OneLinkState(
        x=np.float64(x_new),
        dx=np.float64(dx_new),
        theta=np.float64(th_new),
        dtheta=np.float64(dth_new),
    )


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
