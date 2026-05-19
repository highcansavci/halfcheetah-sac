import pybullet_data

pybullet_data.getDataPath = lambda: r"C:\pybullet_data"

import os
import sys
import argparse
import numpy as np
import torch
import gymnasium as gym
import pybullet_envs_gymnasium  # noqa: F401
import yaml

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from sac_reward_wrapper import UprightWrapper

from train import Actor
from gymnasium.wrappers import RecordVideo
from datetime import datetime

with open("config/sac.yaml", "r") as f:
    config = yaml.safe_load(f)
cfg = config["agent"]


def load_actor(model_dir: str, obs_dim: int, action_dim: int) -> Actor:
    actor = Actor(obs_dim, action_dim, hidden=cfg["hidden_size"])
    path  = os.path.join(model_dir, "actor.pth")
    actor.load_state_dict(torch.load(path, map_location="cpu"))
    actor.eval()
    return actor


@torch.no_grad()
def select_action(actor: Actor, state: np.ndarray, deterministic: bool = True) -> np.ndarray:
    s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    if deterministic:
        mean, _ = actor(s)
        return torch.tanh(mean).squeeze(0).numpy()
    action, _, _ = actor.sample(s)
    return action.squeeze(0).numpy()


def evaluate(model_dir: str, num_episodes: int = 10, render: bool = True, deterministic: bool = True, record: bool = False) -> list:
    render_mode = "rgb_array" if render else None
    env = UprightWrapper(
        gym.make(config["env"]["name"], render_mode=render_mode),
        orientation_coef  = 1.0,
        height_coef       = 4.0,
        min_height        = 0.5,
        terminate_on_fall = False,
    )

    if record:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_dir = os.path.join("videos", timestamp)

        env = RecordVideo(
            env,
            video_folder=video_dir,
            episode_trigger=lambda ep: True,
            name_prefix="sac_eval"
        )

        print(f"\nRecording videos to: {video_dir}\n")


    obs_dim    = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    actor      = load_actor(model_dir, obs_dim, action_dim)

    rewards = []

    print(f"\n{'─'*55}")
    print(f"  Model   : {model_dir}")
    print(f"  Episodes: {num_episodes}")
    print(f"  Mode    : {'deterministic' if deterministic else 'stochastic'}")
    print(f"{'─'*55}\n")

    for ep in range(num_episodes):
        state, _       = env.reset()
        episode_reward = 0.0
        steps          = 0
        done           = False

        while not done:
            action                              = select_action(actor, state, deterministic)
            state, reward, term, trunc, _       = env.step(action)
            done                                = term or trunc
            episode_reward                     += reward
            steps                              += 1

        rewards.append(episode_reward)
        print(f"  Ep {ep+1:>3}/{num_episodes}  R {episode_reward:>8.1f}  Steps {steps:>5}")

    env.close()

    print(f"\n{'─'*55}")
    print(f"  Mean   : {np.mean(rewards):>8.1f}")
    print(f"  Std    : {np.std(rewards):>8.1f}")
    print(f"  Min    : {np.min(rewards):>8.1f}")
    print(f"  Max    : {np.max(rewards):>8.1f}")
    print(f"{'─'*55}\n")

    return rewards


def find_latest_model(base_dir: str = "models") -> str:
    """Return the most recently saved model directory."""
    dirs = [
        os.path.join(base_dir, d)
        for d in os.listdir(base_dir)
        if d.startswith("sac_ep") and os.path.isdir(os.path.join(base_dir, d))
    ]
    if not dirs:
        raise FileNotFoundError(f"No SAC model directories found in '{base_dir}'")
    # sort by episode number
    dirs.sort(key=lambda p: int(os.path.basename(p).replace("sac_ep", "")))
    return dirs[-1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a saved SAC policy on HalfCheetah")
    parser.add_argument("--model",       type=str,  default=None,
                        help="Path to model directory (default: latest in models/)")
    parser.add_argument("--episodes",    type=int,  default=10,
                        help="Number of evaluation episodes (default: 10)")
    parser.add_argument("--no-render",   action="store_true",
                        help="Disable GUI rendering")
    parser.add_argument("--stochastic",  action="store_true",
                        help="Use stochastic policy instead of deterministic")
    parser.add_argument("--record",      action="store_true",
                        help="Record evaluation episodes")
    args = parser.parse_args()

    model_dir = args.model or find_latest_model()

    evaluate(
        model_dir    = model_dir,
        num_episodes = args.episodes,
        render       = not args.no_render,
        deterministic= not args.stochastic,
        record       = args.record,
    )