from __future__ import annotations
from epsim.env.env import MyGrid
from epsim.core import *
import hydra
import pygame
class ManualControl:
    def __init__(
        self,
        env,
        seed=None,
    ) -> None:
        self.env = env
        self.seed = seed
        self.closed = False
        self.info={'action_mask':[1]*5}

    def start(self):
        """Start the window display with blocking event loop"""
        self.reset(self.seed)

        while not self.closed:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.env.close()
                    self.closed=True
                    break
                if event.type == pygame.KEYDOWN:
                    event.key = pygame.key.name(int(event.key))
                    self.key_handler(event)

    def step(self, action: Actions):
        _, reward, terminated, truncated,self.info = self.env.step(action)
        
        if terminated:
            print("terminated!")
            self.reset(self.seed)
        elif truncated:
            print("truncated!")
            self.reset(self.seed)


    def reset(self, seed=None):
        self.env.reset(seed=seed)

    def key_handler(self, event):
        key: str = event.key
        if key == "escape":
            self.env.close()
            self.closed=True
            return
        if key == "backspace":
            self.reset()
            return
        if key == "tab":
            self.env.world.next_product()
            self.env.render()
            return
        if key == "z":
            self.env.next_crane()
            self.env.render()
            return
        if key == "space":
            self.env.world.put_product()
            self.env.render()
            return

        key_to_action = {
            "left": Actions.left,
            "right": Actions.right,
            "down": Actions.bottom , #y轴向下
            "up": Actions.top,
        }
        if key in key_to_action.keys():
            action = key_to_action[key]
            if self.info['action_mask'][action]:
                self.step(action)
        else:
            self.step(Actions.stay)

@hydra.main(config_path="./config", config_name="args", version_base="1.3")
def main(args: "DictConfig"):  # noqa: F821
    env=MyGrid("human",args)
    env.reset(seed=123)

    manual_control = ManualControl(env, seed=1234)
    manual_control.start()

if __name__ == "__main__":
    main()


