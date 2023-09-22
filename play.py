from epsim.envs.myenv import MyEnv
import gymnasium as gym
from sb3_contrib.ppo_mask import MaskablePPO
from epsim.core import SHARE
#from sb3_contrib.common.maskable.utils import get_action_masks
import hydra
@hydra.main(config_path="./config", config_name="args", version_base="1.3")
def main(cfg: "DictConfig"):  # noqa: F821


    model = MaskablePPO.load("models/ppo_mask")
    env=MyEnv(render_mode='human',args=cfg)

    
    obs ,info= env.reset(seed=123)
    for _ in range(10000):
        action_masks = env.world._masks[SHARE.DISPATCH_CODE]
        #print(action_masks)
        action, _states = model.predict(obs,action_masks=1-action_masks)
        obs, reward, done,truc, info = env.step(action)
        if done or truc:
            obs ,info= env.reset()


if __name__ == "__main__":
    main()
    