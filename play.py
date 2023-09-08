from epsim.envs import parallel_env
from polices import RandomSelect,MaskSelect
import numpy as np
import hydra
@hydra.main(config_path="./config", config_name="args", version_base="1.3")
def main(cfg: "DictConfig"):  # noqa: F821
   env = parallel_env(render_mode="human",args=cfg)
   policy=MaskSelect(env) 
   observations, infos = env.reset(seed=40)
   for _ in range(1000):
      actions=policy.decision(observations,infos)
      observations, rewards, terminateds, truncateds, infos = env.step(actions)
      
      if np.any(terminateds):
         observations, infos = env.reset()
   # img = Image.fromarray(observation)
   # img.save("state.jpg")
   env.close()
   
# 
#git tag V1.0
#git push origin --tags
if __name__ == "__main__":
   main()
  