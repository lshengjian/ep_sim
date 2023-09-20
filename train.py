import numpy as np

from epsim.envs.myenv import MyEnv
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

import hydra


@hydra.main(config_path="./config", config_name="args", version_base="1.3")
def main(cfg: "DictConfig"):  # noqa: F821


    # Parallel environments
    vec_env = make_vec_env(lambda :MyEnv(render_mode=None,args=cfg), n_envs=8)

    model = PPO("MlpPolicy", vec_env, verbose=1,policy_kwargs=dict(net_arch=[256, 256, 256]),)
    model.learn(total_timesteps=100000)
    model.save("models/ppo")

    del model # remove to demonstrate saving and loading




if __name__ == "__main__":
    main()
    