from epsim.envs.electroplating_v1 import parallel_env
from polices import RandomSelect,MaskSelect
from epsim.utils import save_img
import numpy as np
import hydra
@hydra.main(config_path="./config", config_name="args", version_base="1.3")
def main(cfg: "DictConfig"):  # noqa: F821
   env = parallel_env(render_mode="human",args=cfg)
   policy=MaskSelect(env)  
   #policy=RandomSelect(env) 
   observations, infos = env.reset(seed=123)
   for k in range(3000):
      actions=policy.decision(observations,infos)
      observations, rewards, terminateds, truncateds, infos = env.step(actions)
      if k==10:
         save_img(env.render(),'outputs/state.jpg')
      dones=list(terminateds.values())
      if any(dones):
         #print('RESET')
         observations, infos = env.reset()
      env.world.next_crane()
   # img = Image.fromarray(observation)
   # img.save("state.jpg")
   env.close()
   
 


if __name__ == "__main__":
   main()
  