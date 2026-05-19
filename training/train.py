import pybullet_data

pybullet_data.getDataPath = lambda: r"C:\pybullet_data"

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import pybullet_envs_gymnasium  # noqa: F401
import yaml
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from sac_reward_wrapper import UprightWrapper
with open("config/sac.yaml", "r") as f:
    config = yaml.safe_load(f)
cfg = config["agent"]


LOG_STD_MIN, LOG_STD_MAX = -5, 2


class Actor(nn.Module):
    """Squashed Gaussian policy with reparameterization."""
    def __init__(self, obs_dim, action_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
        )
        self.mean_layer    = nn.Linear(hidden, action_dim)
        self.log_std_layer = nn.Linear(hidden, action_dim)

    def forward(self, x):
        h = self.net(x)
        mean    = self.mean_layer(h)
        log_std = self.log_std_layer(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, x):
        mean, log_std = self(x)
        std    = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        u      = normal.rsample()                            # reparameterization
        action = torch.tanh(u)

        # log π(a|s) with tanh Jacobian correction
        log_prob = normal.log_prob(u) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob, torch.tanh(mean)           # action, log_π, det_action

    def save(self, path): torch.save(self.state_dict(), path)
    def load(self, path): self.load_state_dict(torch.load(path))


class TwinCritic(nn.Module):
    """Twin Q-networks — prevents value overestimation (SAC key ingredient)."""
    def __init__(self, obs_dim, action_dim, hidden=256):
        super().__init__()

        def _mlp():
            return nn.Sequential(
                nn.Linear(obs_dim + action_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden),                nn.ReLU(),
                nn.Linear(hidden, 1),
            )

        self.q1 = _mlp()
        self.q2 = _mlp()

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)

    def save(self, path): torch.save(self.state_dict(), path)
    def load(self, path): self.load_state_dict(torch.load(path))


class ReplayBuffer:
    def __init__(self, capacity, obs_dim, action_dim):
        self.cap, self.ptr, self.size = capacity, 0, 0
        self.s  = np.zeros((capacity, obs_dim),    dtype=np.float32)
        self.a  = np.zeros((capacity, action_dim), dtype=np.float32)
        self.r  = np.zeros((capacity, 1),          dtype=np.float32)
        self.ns = np.zeros((capacity, obs_dim),    dtype=np.float32)
        self.d  = np.zeros((capacity, 1),          dtype=np.float32)

    def push(self, s, a, r, ns, d):
        self.s[self.ptr]  = s
        self.a[self.ptr]  = a
        self.r[self.ptr]  = r
        self.ns[self.ptr] = ns
        self.d[self.ptr]  = d
        self.ptr  = (self.ptr + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, n):
        idx = np.random.randint(0, self.size, n)
        return (torch.tensor(self.s[idx]),
                torch.tensor(self.a[idx]),
                torch.tensor(self.r[idx]),
                torch.tensor(self.ns[idx]),
                torch.tensor(self.d[idx]))

    def __len__(self): return self.size


class SAC:
    def __init__(self, obs_dim, action_dim):
        h = cfg["hidden_size"]

        self.actor          = Actor(obs_dim, action_dim, h)
        self.critic         = TwinCritic(obs_dim, action_dim, h)
        self.critic_target  = TwinCritic(obs_dim, action_dim, h)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad = False                         # target never trained directly

        self.actor_opt  = torch.optim.Adam(self.actor.parameters(),  lr=cfg["actor_lr"])
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg["critic_lr"])

        self.target_entropy = -float(action_dim)
        self.log_alpha      = torch.tensor([np.log(1)], requires_grad=True, dtype=torch.float32)  # α floor ~0.01
        self.alpha_opt      = torch.optim.Adam([self.log_alpha], lr=cfg["alpha_lr"])

        self.gamma  = cfg["gamma"]
        self.tau    = cfg["tau"]
        self.batch  = cfg["batch_size"]
        self.buffer = ReplayBuffer(cfg["buffer_size"], obs_dim, action_dim)

    @property
    def alpha(self): return self.log_alpha.exp()

    def select_action(self, state, deterministic=False):
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            if deterministic:
                mean, _ = self.actor(s)
                return torch.tanh(mean).squeeze(0).numpy()
            action, _, _ = self.actor.sample(s)
            return action.squeeze(0).numpy()

    def update(self):
        if len(self.buffer) < self.batch:
            return None, None, None

        s, a, r, ns, d = self.buffer.sample(self.batch)

        with torch.no_grad():
            na, log_pi_na, _ = self.actor.sample(ns)
            q1_t, q2_t  = self.critic_target(ns, na)
            q_target     = r + self.gamma * (1 - d) * (
                               torch.min(q1_t, q2_t) - self.alpha.detach() * log_pi_na)

        q1, q2      = self.critic(s, a)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_opt.step()

        new_a, log_pi, _ = self.actor.sample(s)
        q1_new, q2_new   = self.critic(s, new_a)

        actor_loss = (self.alpha.detach() * log_pi - torch.min(q1_new, q2_new)).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (log_pi.detach() + self.target_entropy)).mean()

        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.mul_(1 - self.tau).add_(self.tau * p.data)

        return critic_loss.item(), actor_loss.item(), self.alpha.item()

    # ── persistence ──────────────────────────────────────────────
    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        self.actor.save(f"{directory}/actor.pth")
        self.critic.save(f"{directory}/critic.pth")
        torch.save(self.log_alpha, f"{directory}/log_alpha.pth")

    def load(self, directory):
        self.actor.load(f"{directory}/actor.pth")
        self.critic.load(f"{directory}/critic.pth")
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.log_alpha = torch.load(f"{directory}/log_alpha.pth")


def main():
    env = UprightWrapper(
        gym.make(config["env"]["name"]),
        orientation_coef  = 1.0,
        height_coef       = 4.0,
        min_height        = 0.2,
        terminate_on_fall = False,
    )
    obs_dim    = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = SAC(obs_dim, action_dim)

    warmup_steps = cfg["warmup_steps"]   # fill buffer with random actions first
    update_every = cfg["update_every"]   # gradient steps per env step
    num_episodes = cfg["num_episodes"]

    total_steps    = 0
    reward_history = []

    for episode in range(num_episodes):
        state, _       = env.reset()
        episode_reward = 0.0
        done           = False
        c_loss = a_loss = alpha_val = None

        while not done:
            # ── warmup: purely random actions ──
            if total_steps < warmup_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state)
                action += np.random.normal(0, 0.1, size=action.shape)
                action = np.clip(action, -1, 1)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.buffer.push(state, action, reward / cfg["reward_scale"], next_state, float(done))
            episode_reward += reward
            state          = next_state
            total_steps   += 1

            # ── gradient updates ──
            if total_steps >= warmup_steps and total_steps % update_every == 0:
                c_loss, a_loss, alpha_val = agent.update()

        reward_history.append(episode_reward)
        avg100 = np.mean(reward_history[-100:])

        if c_loss is not None:
            print(f"Ep {episode+1:>6}/{num_episodes}  "
                  f"R {episode_reward:>8.1f}  Avg100 {avg100:>8.1f}  "
                  f"Steps {total_steps:>8}  "
                  f"CriticL {c_loss:>8.3f}  ActorL {a_loss:>7.3f}  "
                  f"α {alpha_val:.4f}")
        else:
            print(f"Ep {episode+1:>6}/{num_episodes}  "
                  f"R {episode_reward:>8.1f}  "
                  f"[warmup — {total_steps}/{warmup_steps} steps]")

        if (episode + 1) % 200 == 0:
            agent.save(f"models/sac_ep{episode+1}")
            print(f"  └─ saved to models/sac_ep{episode+1}/")

    env.close()


if __name__ == "__main__":
    main()