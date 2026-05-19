# halfcheetah-sac
A clean, from-scratch PyTorch implementation of Soft Actor-Critic (SAC) for the HalfCheetahBulletEnv-v0 environment (pybullet).
The project includes a custom reward wrapper that encourages the cheetah to stay upright and at a reasonable height, making the task more stable and less prone to degenerate "crawling" policies.
Features

SAC with twin critics, entropy regularization (learnable α), and squashed Gaussian policy
Reparameterization trick + proper tanh Jacobian correction
Target network with soft updates (Polyak averaging)
Simple numpy-based replay buffer
Custom UprightWrapper for better reward shaping
Training + evaluation scripts
Model saving/loading and video recording support

Project Structure
texthalfcheetah-sac/
├── config/
│   └── sac.yaml          # Hyperparameters
├── training/
│   ├── train.py          # Main training loop
│   ├── eval.py           # Evaluation + video recording
│   ├── sac_reward_wrapper.py
│   └── measure_height.py # Utility to inspect environment
├── videos/               # Generated evaluation videos
├── models/               # Saved model checkpoints (created during training)
└── .gitignore
Installation
Bash# 1. Clone the repo
git clone https://github.com/highcansavci/halfcheetah-sac.git
cd halfcheetah-sac

# 2. Create environment (recommended)
conda create -n halfcheetah-sac python=3.10
conda activate halfcheetah-sac

# 3. Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu  # or +cu121 for CUDA
pip install gymnasium pybullet pybullet_envs_gymnasium numpy pyyaml
Note: The code contains a hardcoded path for pybullet_data. You may need to adjust the line in train.py and eval.py if you encounter data loading issues.
Configuration (config/sac.yaml)
YAMLenv:
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
Training
Bashcd training
python train.py

Models are automatically saved every 200 episodes to models/sac_epXXXX/
Progress (episode reward, Avg100, losses, α) is printed to console

Evaluation
Bash# Evaluate latest model
python eval.py

# Specific options
python eval.py --model models/sac_ep2000 --episodes 50 --record --no-render
Arguments:

--model: path to model folder (default = latest)
--episodes: number of eval episodes
--record: save videos to videos/
--stochastic: use stochastic instead of deterministic policy
--no-render: disable GUI

Custom Reward Shaping (UprightWrapper)
The wrapper modifies the original reward to discourage falling/low posture:

Scales forward reward by torso height
Adds orientation bonus (cos(pitch))
Applies quadratic height penalty
Optional early termination on fall

This helps the agent learn a more natural running gait.
Requirements

Python 3.9+
PyTorch
Gymnasium + pybullet_envs_gymnasium
numpy, pyyaml

References / Inspiration

Original SAC paper: Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor
Clean implementations from the RL community (denisyarats/pytorch_sac, etc.)


Happy running! 🐆
Feel free to open issues or PRs for improvements, hyperparameter sweeps, or MuJoCo/Gymnasium v1 support.
