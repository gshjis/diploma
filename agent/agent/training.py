"""Training functions for the cart+double-pendulum system."""

from __future__ import annotations

import numpy as np

import physics_engine as phys
from .config import HoldConfig
from .env import PhysicsEnv
from .agents import PDHoldAgent


def train_sac_stable_baselines3(
    total_timesteps: int = 100_000,
    seed: int = 0,
    n_envs: int = 8,
) -> None:
    """Train SAC using stable-baselines3.
    
    Requires gymnasium and stable-baselines3 packages.
    
    Args:
        total_timesteps: Total number of training timesteps.
        seed: Random seed for reproducibility.
        n_envs: Number of parallel environments.
    """
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import SubprocVecEnv

    p = phys.SystemParameters()
    cfg = HoldConfig()
    
    obs_dim = 6
    act_dim = 1

    def make_env(rank: int) -> gym.Env:
        env_impl = PhysicsEnv(p=p, cfg=cfg)

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

        return GymAdapter()

    np.random.seed(seed)

    # Parallel rollout environments.
    # SubprocVecEnv requires "pickleable" factory functions.
    env_fns = []
    for i in range(n_envs):
        env_fns.append(lambda i=i: make_env(i))

    env = SubprocVecEnv(env_fns, start_method="fork")

    model = SAC(
        policy="MlpPolicy",
        env=env,
        seed=seed,
        verbose=1,
        learning_rate=1e-4,
        tau=0.005,
        gamma=0.99,
        batch_size=256,
        buffer_size=200_000,
        target_entropy=-2.0,
        train_freq=1,
        gradient_steps=1,
    )

    # Progress bar
    from tqdm import tqdm

    # Callback for logging episode rewards
    from stable_baselines3.common.callbacks import BaseCallback

    class EpisodeRewardLogger(BaseCallback):
        """Callback to log episode rewards during training."""
        
        def __init__(self) -> None:
            super().__init__()
            self.cur_ep_rewards: list[float] | None = None
            self.ep_rewards: list[float] = []
            self.last_log_step = 0

        def _on_step(self) -> bool:
            infos = self.locals.get("infos")
            if not infos:
                return True

            # Initialize counters on first step
            if self.cur_ep_rewards is None:
                self.cur_ep_rewards = [0.0 for _ in range(len(infos))]

            rewards = self.locals.get("rewards")
            rewards_arr = rewards if rewards is not None else None

            # Accumulate reward per step
            if rewards_arr is not None:
                for i in range(len(infos)):
                    self.cur_ep_rewards[i] += float(rewards_arr[i])

            # Check for episode info from stable-baselines3
            for info in infos:
                ep_info = info.get("episode") if isinstance(info, dict) else None
                if ep_info is not None and "r" in ep_info:
                    self.ep_rewards.append(float(ep_info["r"]))

            # Check for done signals
            dones = self.locals.get("dones")
            if dones is not None and rewards_arr is not None and self.cur_ep_rewards is not None:
                for i, d in enumerate(dones):
                    if bool(d):
                        self.ep_rewards.append(float(self.cur_ep_rewards[i]))
                        self.cur_ep_rewards[i] = 0.0

            # Log every ~10k timesteps
            if self.num_timesteps - self.last_log_step >= 10_000 and self.ep_rewards:
                mean_r = sum(self.ep_rewards) / len(self.ep_rewards)
                print(f"[reward-log] mean_episode_reward={mean_r:.3f} over {len(self.ep_rewards)} episodes")
                self.ep_rewards.clear()
                self.last_log_step = self.num_timesteps

            # Force output if buffer has rewards
            if len(self.ep_rewards) > 0 and (self.num_timesteps % 5000 == 0):
                mean_r = sum(self.ep_rewards) / len(self.ep_rewards)
                print(f"[reward-log] mean_reward={mean_r:.2f} (buffer={len(self.ep_rewards)})")
            return True

    remaining = total_timesteps
    with tqdm(total=total_timesteps, desc="SAC train", unit="ts") as pbar:
        while remaining > 0:
            step_now = min(10_000, remaining)
            model.learn(
                total_timesteps=step_now,
                reset_num_timesteps=False,
                callback=EpisodeRewardLogger(),
            )
            remaining -= step_now
            pbar.update(step_now)
    
    model.save("sac_cart_pendulum")


def train_mlp(
    episodes: int = 2000,
    steps_per_episode: int = 2000,
    lr: float = 1e-4,
    batch_size: int = 512,
    seed: int = 0,
) -> None:
    """Train a small MLP policy with supervised imitation from PD controller.
    
    This trains a neural network to approximate the PD controller's actions
    through supervised learning.
    
    Args:
        episodes: Number of training episodes.
        steps_per_episode: Maximum steps per episode.
        lr: Learning rate.
        batch_size: Batch size for training.
        seed: Random seed.
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
        """Simple MLP policy network."""
        
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

    # Replay buffers for imitation
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
    """Run a single episode with PD controller and return total reward.
    
    Args:
        seed: Random seed for environment reset.
    
    Returns:
        Total reward accumulated during the episode.
    """
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
