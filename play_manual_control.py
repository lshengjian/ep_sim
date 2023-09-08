from __future__ import annotations
from epsim.envs.myenv import MyEnv
from epsim.core import *
from epsim.utils import save_img
import hydra
import pygame

class ManualControl:
    def __init__(
        self,
        env
    ) -> None:
        self.env = env
        self.running = True
        self.info={'action_mask':[1]*5}

    def start(self):
        # pygame.init()
        # pygame.display.init()
        # pygame.display.set_caption('test')
        #         #self.font  =  pygame.font.Font(None,26)
        # window=pygame.display.set_mode((400,200))# ,pygame.DOUBLEBUF, 32)
        # clock = pygame.time.Clock()


        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    key=pygame.key.name(event.key)
                    self.key_handler(key)
                    self.env.render()

                    

            # window.fill((0, 0, 0))
            # pygame.draw.circle(window, (255,0,0), (100, 100), 60)
            # pygame.display.flip()
            # clock.tick(60)
        pygame.quit()


    def step(self, action: Actions):
        obs, reward, terminated, truncated,self.info = self.env.step(action)
        #print(obs)
        #save_img(img,'outputs/'+self.env.world.cur_crane.cfg.name+'.jpg')
        if terminated:
            print("game over!")
            self.env.reset()
        elif truncated:
            print("truncated!")
            self.env.reset()


        

    def key_handler(self,key):
        #print(key)
        if key == "escape":
            self.env.close()
            self.running=False
            return
        if key == "backspace":
            self.env.reset()
            return
        if key == "q":
            self.env.world._next_product()
            return
        if key == "tab":
            self.env.next_crane()
            return
        # if key == "space":
        #     self.env.world.put_product()
        #     self.env.render()
        #     return

        key_to_action = {
            "left": Actions.left,
            "right": Actions.right,
            "down": Actions.bottom , 
            "up": Actions.top,
        }
        if key in key_to_action.keys():
            action = key_to_action[key]
            if self.info['action_masks'][action]:
                self.step(action)
        else:
            self.step(Actions.stay)

@hydra.main(config_path="./config", config_name="args", version_base="1.3")
def main(args: "DictConfig"):  # noqa: F821
    env=MyEnv("human",args)
    env.reset(seed=123)

    manual_control = ManualControl(env)
    manual_control.start()

if __name__ == "__main__":
    main()

