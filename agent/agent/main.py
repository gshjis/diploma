"""Main entry point for the cart+double-pendulum training."""

from .training import train_sac_stable_baselines3, train_mlp, run_episode


def main() -> None:
    """Run SAC training."""
    train_sac_stable_baselines3()


if __name__ == "__main__":
    # Quick training (reduced to avoid killing process by time/memory)
    main()
