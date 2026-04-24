"""Baseline controller and environment wrapper for `phisics_engine.py`.

Полностью SAC здесь не включаем (требует torch + корректная типизация/нейросети).
Функционально: среда + PD-контроллер, использующий динамику из `phisics_engine.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

import physics_engine as phys


@dataclass
class HoldConfig:
    u_limit: float = 20.0
    x_limit: float = 5.0
    max_steps: int = 5000
    dt: float = 1.0 / 240.0

    # failure threshold (wide enough to let the system evolve)
    fall_threshold: float = 1.0

    # PD gains (stabilization baseline)
    kx: float = 0.0
    vx: float = 0.1
    k1: float = 15.0
    kw1: float = 3.0
    k2: float = 10.0
    kw2: float = 2.5


def _reward(s: phys.State) -> float:
    th1 = float(s.pos.Teta1)
    th2 = float(s.pos.Teta2)
    dth1 = float(s.vel.dTeta1)
    dth2 = float(s.vel.dTeta2)
    x = float(s.pos.x)
    dx = float(s.vel.dx)
    
    target1 = np.pi
    target2 = np.pi
    
    cost = (
        2.0 * ((th1 - target1)**2 + (th2 - target2)**2)
        + 0.5 * (dth1 * dth1 + dth2 * dth2)
        + 0.05 * (x * x)
        + 0.01 * (dx * dx)
    )
    return float(-cost)


class PhysicsEnv:
    def __init__(self, p: phys.SystemParameters, cfg: HoldConfig) -> None:
        self.p = p
        self.cfg = cfg
        self._state: phys.State | None = None
        self._steps = 0

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)

        th1 = np.random.uniform(-0.2, 0.2)
        th2 = np.random.uniform(-0.2, 0.2)
        self._state = phys.State(
            pos=phys.Positions(x=np.float64(0.0), Teta1=np.float64(th1), Teta2=np.float64(th2)),
            vel=phys.Velocities(dx=np.float64(0.0), dTeta1=np.float64(0.0), dTeta2=np.float64(0.0)),
        )
        self._steps = 0
        return self._obs(self._state)

    def _obs(self, s: phys.State) -> np.ndarray:
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
        assert self._state is not None
        self._steps += 1

        u = float(np.clip(u, -self.cfg.u_limit, self.cfg.u_limit))
        self._state = phys.rk4_step(self._state, u=u, p=self.p, dt=self.cfg.dt)
        s = self._state

        out_of_bounds = abs(float(s.pos.x)) > self.cfg.x_limit
        fallen = abs(float(s.pos.Teta1)) > self.cfg.fall_threshold or abs(float(s.pos.Teta2)) > self.cfg.fall_threshold

        state_arr = s.to_array()
        nan_state = bool(np.isnan(state_arr).any())

        done = bool(out_of_bounds or fallen or nan_state or self._steps >= self.cfg.max_steps)

        control_penalty = 1e-2 * (u * u)
        reward = _reward(s) - control_penalty

        obs = self._obs(s)
        info = {"fallen": fallen, "out_of_bounds": out_of_bounds, "nan_state": nan_state}
        return obs, reward, done, info


def train_sac_stable_baselines3(
    total_timesteps: int = 100_000,
    seed: int = 0,
) -> None:
    """Train SAC using stable-baselines3.

    Требует gymnasium (или gym) окружение; мы делаем минимальный адаптер.
    """

    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import SAC

    p = phys.SystemParameters()
    cfg = HoldConfig()
    env_impl = PhysicsEnv(p=p, cfg=cfg)

    obs_dim = 6
    act_dim = 1

    class GymAdapter(gym.Env):
        metadata: dict[str, list[str]] = {"render_modes": []}

        def __init__(self) -> None:
            super().__init__()
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(obs_dim,),
                dtype=np.float32,
            )
            self.action_space = spaces.Box(
                low=-cfg.u_limit,
                high=cfg.u_limit,
                shape=(act_dim,),
                dtype=np.float32,
            )

        def reset(self, *, seed: int | None = None, options=None):
            obs = env_impl.reset(seed=seed)
            return obs, {}

        def step(self, action: np.ndarray):
            u = float(action[0])
            obs, reward, done, info = env_impl.step(u)
            terminated = bool(done)
            truncated = False
            return obs, float(reward), terminated, truncated, info

    env = GymAdapter()
    np.random.seed(seed)

    model = SAC(
        policy="MlpPolicy",
        env=env,
        seed=seed,
        verbose=1,
        learning_rate=3e-4,
        gamma=0.99,
        batch_size=256,
        buffer_size=200_000,
        train_freq=1,
        gradient_steps=1,
    )

    # tqdm только как прогресс-бар на общий прогресс обучения.
    from tqdm import tqdm

    remaining = total_timesteps
    with tqdm(total=total_timesteps, desc="SAC train", unit="ts") as pbar:
        while remaining > 0:
            step_now = min(10_000, remaining)
            model.learn(total_timesteps=step_now, reset_num_timesteps=False)
            remaining -= step_now
            pbar.update(step_now)
    model.save("sac_cart_pendulum")


class PDHoldAgent:
    """PD feedback baseline for keeping θ1 and θ2 near 0."""

    def __init__(self, cfg: HoldConfig) -> None:
        self.cfg = cfg

    def act(self, obs: np.ndarray) -> float:
        x, dx, th1, dth1, th2, dth2 = [float(v) for v in obs.tolist()]
        u = (
            self.cfg.kx * x
            -self.cfg.vx * dx
            -self.cfg.k1 * th1
            -self.cfg.kw1 * dth1
            -self.cfg.k2 * th2
            -self.cfg.kw2 * dth2
        )
        return float(u)


def train_mlp(
    episodes: int = 2000,
    steps_per_episode: int = 2000,
    lr: float = 1e-3,
    batch_size: int = 256,
    seed: int = 0,
) -> None:
    """Train a small MLP policy with supervised imitation from PD.

    Это "обучение" без SAC: мы собираем (obs -> u_PD) пары и
    подбираем нейросеть, которая аппроксимирует стабилизирующий контроллер.
    """

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(seed)
    np.random.seed(seed)

    p = phys.SystemParameters()
    cfg = HoldConfig()
    env = PhysicsEnv(p=p, cfg=cfg)
    teacher = PDHoldAgent(cfg)

    obs_dim = 6
    act_dim = 1

    class Policy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(obs_dim, 128)
            self.fc2 = nn.Linear(128, 128)
            self.out = nn.Linear(128, act_dim)

        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            x = F.relu(self.fc1(obs))
            x = F.relu(self.fc2(x))
            return self.out(x)

    policy = Policy()
    opt = torch.optim.Adam(policy.parameters(), lr=lr)

    # replay buffers for imitation
    obs_buf: list[np.ndarray] = []
    act_buf: list[np.ndarray] = []

    def push_transition(obs: np.ndarray, u: float) -> None:
        obs_buf.append(obs.astype(np.float32, copy=False))
        act_buf.append(np.array([u], dtype=np.float32))

    policy.train()
    for ep in range(episodes):
        obs = env.reset(seed=seed + ep)
        for _ in range(steps_per_episode):
            with torch.no_grad():
                u_teacher = teacher.act(obs)
            next_obs, _r, done, _info = env.step(u_teacher)
            push_transition(obs, u_teacher)
            obs = next_obs
            if done:
                break

            if len(obs_buf) >= batch_size:
                idx = np.random.choice(len(obs_buf), size=batch_size, replace=False)
                obs_b = torch.tensor(np.stack([obs_buf[i] for i in idx]), dtype=torch.float32)
                act_b = torch.tensor(np.stack([act_buf[i] for i in idx]), dtype=torch.float32)

                pred = policy(obs_b)
                loss = F.mse_loss(pred, act_b)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

        if (ep + 1) % 100 == 0:
            print(f"[imit] ep={ep+1}/{episodes}")

    torch.save(policy.state_dict(), "policy_pd_imitation.pt")
    print("Saved policy to policy_pd_imitation.pt")


def run_episode(seed: int = 0) -> float:
    p = phys.SystemParameters()
    cfg = HoldConfig()
    env = PhysicsEnv(p=p, cfg=cfg)
    agent = PDHoldAgent(cfg)

    obs = env.reset(seed=seed)
    total_reward = 0.0
    done = False
    while not done:
        u = agent.act(obs)
        obs, r, done, _info = env.step(u)
        total_reward += r
    return total_reward


if __name__ == "__main__":

    # quick training (уменьшено, чтобы не убивало процесс по времени/памяти)
    train_sac_stable_baselines3()
