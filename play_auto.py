from epsim.core import SHARE
from epsim.envs.myenv import MyEnv
from polices import MaskSelect

import numpy as np
import hydra
@hydra.main(config_path="./config", config_name="args", version_base="1.3")
def main(cfg: "DictConfig"):  # noqa: F821
   env = MyEnv(render_mode="human",args=cfg)
   policy=MaskSelect(env) 
   observation, info = env.reset(seed=1234)
   for k in range(10000):
      #print(info)
      actions=policy.decision(info)
      observation, reward, terminated, truncated, info = env.step(actions[SHARE.DISPATCH_CODE])
      done=terminated or truncated
      if done:
         observation, info = env.reset()

   env.close()
   
 


if __name__ == "__main__":
   main()
  