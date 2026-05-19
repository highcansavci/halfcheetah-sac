import gymnasium as gym
import numpy as np
import pybullet

class UprightWrapper(gym.Wrapper):
    def __init__(self, env, min_height=0.5, orientation_coef=1.0,
                 height_coef=10.0, terminate_on_fall=False):
        super().__init__(env)
        self.min_height       = min_height
        self.orientation_coef = orientation_coef
        self.height_coef      = height_coef
        self.terminate_on_fall = terminate_on_fall

    def _get_robot_state(self):
        robot  = self.env.unwrapped.robot
        height = robot.body_xyz[2]        # reliable torso z
        pitch  = robot.body_rpy[1]        # forward tilt
        return height, pitch

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        try:
            height, pitch = self._get_robot_state()
        except Exception:
            return obs, reward, terminated, truncated, info

        # Orientation: 0 when upright, -2 when inverted
        orient_bonus = self.orientation_coef * (np.cos(pitch) - 1.0)

        # Quadratic height penalty — grows fast as cheetah dips
        height_diff    = max(0.0, self.min_height - height)
        height_penalty = self.height_coef * (height_diff ** 2)

        # Reward scale: forward reward shrinks toward 0 as cheetah lowers body
        # This makes hacking unprofitable regardless of forward velocity
        height_scale   = float(np.clip(height / self.min_height, 0.0, 1.0))
        scaled_reward  = reward * height_scale

        modified_reward = scaled_reward + orient_bonus - height_penalty

        if self.terminate_on_fall and height < self.min_height * 0.5:
            terminated = True

        return obs, modified_reward, terminated, truncated, info