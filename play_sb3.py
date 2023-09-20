"""Uses Stable-Baselines3 to train agents in the Connect Four environment using invalid action masking.

For information about invalid action masking in PettingZoo, see https://pettingzoo.farama.org/api/aec/#action-masking
For more information about invalid action masking in SB3, see https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html

"""
import glob
import os
import time
import numpy as np

from epsim.envs.myenv import MyEnv
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

import hydra


@hydra.main(config_path="./config", config_name="args", version_base="1.3")
def main(cfg: "DictConfig"):  # noqa: F821


    # Parallel environments
    # vec_env = make_vec_env(lambda :MyEnv(render_mode=None,args=cfg), n_envs=8)

    # model = PPO("MlpPolicy", vec_env, verbose=1)
    # model.learn(total_timesteps=1000000)
    # model.save("ppo")

    # del model # remove to demonstrate saving and loading

    model = PPO.load("ppo")
    env=MyEnv(render_mode='human',args=cfg)

    
    obs ,info= env.reset(seed=123)
    for _ in range(10000):
        action, _states = model.predict(obs)
        obs, reward, done,truc, info = env.step(action)
        if done or truc:
            obs ,info= env.reset()


if __name__ == "__main__":
    main()
    