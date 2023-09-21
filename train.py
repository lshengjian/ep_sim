import gymnasium as gym
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from epsim.core import SHARE
from epsim.envs.myenv import MyEnv
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO

import hydra
def mask_fn(env: gym.Env) -> np.ndarray:
    # Do whatever you'd like in this function to return the action mask
    # for the current env. In this example, we assume the env has a
    # helpful method we can rely on.
    return env.world._masks[SHARE.DISPATCH_CODE]

@hydra.main(config_path="./config", config_name="args", version_base="1.3")
def main(cfg: "DictConfig"):  # noqa: F821
    vec_env = make_vec_env(lambda :ActionMasker(MyEnv(render_mode=None,args=cfg),mask_fn), n_envs=8)
    model = MaskablePPO("MlpPolicy", vec_env, verbose=1,policy_kwargs=dict(net_arch=[256, 256, 256]))
    model.learn(total_timesteps=100000)
    model.save("models/ppo_mask")



# from stable_baselines3 import PPO

# @hydra.main(config_path="./config", config_name="args", version_base="1.3")
# def main(cfg: "DictConfig"):  # noqa: F821


#     # Parallel environments
#     vec_env = make_vec_env(lambda :MyEnv(render_mode=None,args=cfg), n_envs=8)

#     model = PPO("MlpPolicy", vec_env, verbose=1,policy_kwargs=dict(net_arch=[256, 256, 256]),)
#     model.learn(total_timesteps=100000)
#     model.save("models/ppo")

#     del model # remove to demonstrate saving and loading




if __name__ == "__main__":
    main()
    