# HalfCheetah-SAC

A clean, from-scratch PyTorch implementation of **Soft Actor-Critic (SAC)** for the `HalfCheetahBulletEnv-v0` environment (pybullet).

The agent learns a robust running policy with a custom reward wrapper that discourages "falling" or low-body hacking strategies common in standard HalfCheetah.

## Features

- **Squashed Gaussian policy** with proper tanh reparameterization and Jacobian correction
- **Twin Q-networks** (Clipped Double Q-learning)
- **Automatic temperature tuning** (learnable α)
- **Custom UprightWrapper**:
  - Height-based reward scaling
  - Orientation bonus
  - Height penalty
  - (Optional) termination on fall
- Simple ReplayBuffer with NumPy storage
- YAML configuration
- Training + evaluation + video recording scripts
- Gradient clipping for stability

## Demo
https://github.com/user-attachments/assets/2ce7be22-1c3c-4d59-8a4f-21555e50916a

## Repository Structure
halfcheetah-sac/  
├── config/  
│   └── sac.yaml                 # Hyperparameters  
├── training/  
│   ├── train.py                 # Main training script  
│   ├── eval.py                  # Evaluation + video recording  
│   ├── sac_reward_wrapper.py    # UprightWrapper  
│   ├── measure_height.py        # Utility to inspect environment  
│   └── init.py  
├── models/                      # (generated) saved checkpoints  
├── videos/                      # (generated) evaluation recordings  
└── .gitignore  


## Installation

1. Clone the repo:
   ```bash
   git clone https://github.com/highcansavci/halfcheetah-sac.git
   cd halfcheetah-sac

Create a virtual environment (recommended) and install dependencies:Bashpip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu  # or cuda if you have GPU
pip install gymnasium pybullet pybullet-envs-gymnasium numpy pyyaml

Note: The code hardcodes a Windows path for pybullet_data. You may need to adjust or remove the override in train.py and eval.py for your system.
## Training
cd training
python train.py

Models are saved every 200 episodes to models/sac_epXXXX/
Training progress (episode reward, Avg100, losses, α) is printed to console.
Default config runs for 100,000 episodes (very long — adjust in config/sac.yaml).

## Evaluation
cd training
python eval.py --help
Examples:

## Evaluate latest model (deterministic, with GUI)
python eval.py

## Record videos
python eval.py --record

## Stochastic policy, no render, 20 episodes
python eval.py --stochastic --no-render --episodes 20

## Specific model
python eval.py --model models/sac_ep2000
Configuration (config/sac.yaml)
env:
  name: HalfCheetahBulletEnv-v0

agent:
  hidden_size: 256
  actor_lr:    0.0003
  critic_lr:   0.0003
  alpha_lr:    0.0003
  gamma:       0.99
  tau:         0.005
  batch_size:  256
  buffer_size: 1000000
  warmup_steps: 10000
  update_every: 1
  reward_scale: 5.0
  num_episodes: 100000
Feel free to tweak these values.

## Notes

The UprightWrapper significantly changes the reward landscape to promote stable, upright running.
Early training is mostly random (warmup). Learning starts after ~10k steps.
Videos are saved under videos/YYYYMMDD_HHMMSS/.

## References

Original SAC paper: Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor


Happy running! 🐆
Pull requests, issues, and improvements are welcome.
