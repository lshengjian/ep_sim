from sb3_contrib import MaskablePPO
from sb3_contrib.common.envs import InvalidActionEnvDiscrete
from epsim.envs.myenv import MyEnv
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.utils import get_action_masks
import hydra

@hydra.main(config_path="./config", config_name="args", version_base="1.3")
def main(cfg: "DictConfig"):  # noqa: F821
    env = MyEnv(render_mode='ansi',args=cfg)
    model = MaskablePPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=5000)
    evaluate_policy(model, env, n_eval_episodes=20, reward_threshold=90, warn=False)

    model.save("ppo_mask")
    del model # remove to demonstrate saving and loading

    model = MaskablePPO.load("ppo_mask")

    obs, _ = env.reset()
    while True:
        # Retrieve current action mask
        action_masks = get_action_masks(env)
        action, _states = model.predict(obs, action_masks=action_masks)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()


if __name__ == "__main__":
   main()