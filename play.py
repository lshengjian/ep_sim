from epsim.envs.myenv import MyEnv
from epsim.polices import RandomSelect,MaskSelect
from PIL import Image
import hydra
@hydra.main(config_path="./config", config_name="args", version_base="1.3")
def main(cfg: "DictConfig"):  # noqa: F821
   env = MyEnv(render_mode="rgb_array",args=cfg)
   #policy=RandomSelect(env) 
   policy=MaskSelect(env) 
   observation, info = env.reset(seed=40)
   for _ in range(10):
      action=policy.decision(info)
      observation, reward, terminated, truncated, info = env.step(action)
      
      if terminated or truncated:
         observation, info = env.reset()
   img = Image.fromarray(observation)
   img.save("state.jpg")
   env.close()
   
# sprite_image
#git tag V1.0
#git push origin --tags
if __name__ == "__main__":
   main()
  