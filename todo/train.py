from gymnasium.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

import hydra
import  sys
from os import path

dir=path.abspath(path.dirname(__file__))
sys.path.append(dir)

@hydra.main(config_path="./config", config_name="args", version_base="1.3")
def main(cfg: "DictConfig"):  # noqa: F821
    register(
        id="epsim-v1",
        entry_point="epsim.envs:MyEnv",
        kwargs={"args":cfg},
    )


    # Parallel environments
    vec_env = make_vec_env("epsim-v1", n_envs=4)

    model = PPO("MlpPolicy", vec_env, verbose=1)
    model.learn(total_timesteps=50000)
    model.save("ppo_epsim")

    del model # remove to demonstrate saving and loading

    model = PPO.load("ppo_epsim")

    obs = vec_env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        vec_env.render("human")


if __name__ == "__main__":
   main()