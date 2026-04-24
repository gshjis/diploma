"""Pygame-based controller for the analytical cart-pendulum model.

Left/Right arrows apply constant force u to the cart:
  - Left  : u = -F_MAX
  - Right : u = +F_MAX
  - No arrow pressed: u = 0
"""

from __future__ import annotations

import sys
import numpy as np
import pygame

from physics_engine import Positions, SystemParameters, Velocities, State, rk4_step


def _float(x: np.float64 | float) -> float:
    return float(x)


def main() -> None:
    pygame.init()

    # -------------------- Model params --------------------
    p = SystemParameters()

    # Force bounds (N)
    F_MAX = 70.0

    # Simulation step: держим рилтайм.
    # Интегрируем с фиксированным dt, но делаем столько шагов,
    # сколько накопилось по реальному времени (clock.tick()).
    dt = 1.0 / 240.0

    # Initial state
    state: State = State(
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

    # -------------------- Pygame setup --------------------
    width, height = 900, 600
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Analytical engine control (arrows -> force u)")
    clock = pygame.time.Clock()

    # -------------------- SAC agent (optional) --------------------
    model_path = "sac_cart_pendulum.zip"
    sac_model = None
    if model_path is not None:
        try:
            from stable_baselines3 import SAC

            sac_model = SAC.load(model_path)
        except Exception:
            sac_model = None

    # World-to-screen mapping
    origin_x = width // 2
    origin_y = height // 3 - 30  # чуть выше
    scale = 220.0  # meters -> pixels (arbitrary for visualization)

    # Rendering sizes
    cart_w, cart_h = 90, 30
    theta1_len = _float(p.L1) * scale
    theta2_len = _float(p.L2) * scale

    def draw_scene(s: State) -> None:
        screen.fill((245, 245, 245))

        # Ground line
        pygame.draw.line(screen, (120, 120, 120), (0, origin_y + 260), (width, origin_y + 260), 2)

        # Cart position
        x_px = origin_x + _float(s.pos.x) * scale
        cart_rect = pygame.Rect(int(x_px - cart_w / 2), int(origin_y + 260 - cart_h / 2), cart_w, cart_h)
        pygame.draw.rect(screen, (40, 40, 40), cart_rect, border_radius=6)

        # Pendulums (two serial links for illustration)
        # We assume both angles are measured from vertical.
        # Для отображения в радианах удобно ограничить их по диапазону [0, 2π).
        # Но печатаем также исходные значения — если модель уходит далеко, это
        # влияет на геометрию, а не на формат вывода.
        theta1 = float(s.pos.Teta1)
        theta2 = float(s.pos.Teta2)

        theta1_vis = theta1 % (2.0 * np.pi)
        theta2_vis = theta2 % (2.0 * np.pi)

        # Joint 1 at cart center
        joint1 = pygame.Vector2(x_px, origin_y + 260)
        p1 = pygame.Vector2(
            joint1.x + theta1_len * np.sin(theta1_vis),
            joint1.y + theta1_len * np.cos(theta1_vis),
        )

        # Second link (visualization only).
        # If your theta2 is defined relative to theta1 in the real model,
        # adjust this drawing accordingly.
        p2 = pygame.Vector2(
            p1.x + theta2_len * np.sin(theta2_vis),
            p1.y + theta2_len * np.cos(theta2_vis),
        )

        pygame.draw.line(screen, (30, 120, 220), joint1, p1, 4)
        pygame.draw.circle(screen, (30, 120, 220), (int(p1.x), int(p1.y)), 8)
        pygame.draw.line(screen, (220, 80, 30), p1, p2, 4)
        pygame.draw.circle(screen, (220, 80, 30), (int(p2.x), int(p2.y)), 8)

        # Text
        u_text = f"u={u_current:+.2f} N"
        # Печатаем только значения в скобках: (theta1_vis), (theta2_vis)
        state_text = (
            f"x={_float(s.pos.x):+.3f}, "
            f"th1=  {theta1_vis:+.3f}, "
            f"th2={theta2_vis:+.3f}"
        )
        font = pygame.font.Font(None, 28)
        screen.blit(font.render(u_text, True, (0, 0, 0)), (15, 15))
        screen.blit(font.render(state_text, True, (0, 0, 0)), (15, 45))

        pygame.display.flip()

    u_current: float = 0.0

    running = True
    sim_time_acc = 0.0
    last_t = pygame.time.get_ticks()
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

        keys = pygame.key.get_pressed()
        if sac_model is not None:
            # agent controls u
            obs = np.array(
                [
                    _float(state.pos.x),
                    _float(state.vel.dx),
                    _float(state.pos.Teta1),
                    _float(state.vel.dTeta1),
                    _float(state.pos.Teta2),
                    _float(state.vel.dTeta2),
                ],
                dtype=np.float32,
            )
            action, _ = sac_model.predict(obs, deterministic=True)
            # action shape=(1,)
            u_current = float(action[0])
        else:
            # manual keyboard fallback
            if keys[pygame.K_LEFT]:
                u_current = -F_MAX
            elif keys[pygame.K_RIGHT]:
                u_current = F_MAX
            else:
                u_current = 0.0

        # Integrate in (quasi) real-time with fixed dt
        now_t = pygame.time.get_ticks()
        frame_dt = (now_t - last_t) / 1000.0
        last_t = now_t

        # защитимся от больших пауз (например при сворачивании окна)
        frame_dt = min(frame_dt, 0.05)

        sim_time_acc += frame_dt
        while sim_time_acc >= dt:
            state = rk4_step(state, float(u_current), p, dt)
            sim_time_acc -= dt

        draw_scene(state)
        clock.tick(60)

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
