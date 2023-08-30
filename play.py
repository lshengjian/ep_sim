from epsim.envs.env import MyGrid
from epsim.polices import RandomSelect,MaskSelect

import hydra
@hydra.main(config_path="./config", config_name="args", version_base="1.3")
def main(cfg: "DictConfig"):  # noqa: F821
   env = MyGrid(render_mode="human",args=cfg)
   #policy=RandomSelect(env) 
   policy=MaskSelect(env) 
   observation, info = env.reset(seed=40)
   for _ in range(500):
      action=policy.decision(info)
      observation, reward, terminated, truncated, info = env.step(action)
      
      if terminated or truncated:
         
         observation, info = env.reset()
   env.close()
   
if __name__ == "__main__":
   main()
  