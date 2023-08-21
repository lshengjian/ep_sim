#!/usr/bin/env python3

from __future__ import annotations

import gymnasium as gym
import pygame
from gymnasium import Env

from epsim.core import Actions
from epsim.envs.playground import PlaygroundEnv
# from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper


class ManualControl:
    def __init__(
        self,
        env: Env,
        seed=None,
    ) -> None:
        self.env = env
        self.seed = seed
        self.closed = False

    def start(self):
        """Start the window display with blocking event loop"""
        self.reset(self.seed)

        while not self.closed:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.env.close()
                    break
                if event.type == pygame.KEYDOWN:
                    event.key = pygame.key.name(int(event.key))
                    self.key_handler(event)

    def step(self, action: Actions):
        _, reward, terminated, truncated, _ = self.env.step(action)
        print(f"step={self.env.step_count}, reward={reward:.2f}")

        if terminated:
            print("terminated!")
            self.reset(self.seed)
        elif truncated:
            print("truncated!")
            self.reset(self.seed)
        else:
            self.env.render()

    def reset(self, seed=None):
        self.env.reset(seed=seed)
        #self.env.render()

    def key_handler(self, event):
        key: str = event.key
        #print("pressed", key)

        if key == "escape":
            self.env.close()
            return
        if key == "backspace":
            self.reset()
            return

        key_to_action = {
            "left": Actions.left,
            "right": Actions.right,
            "up": Actions.forward
        }
        if key in key_to_action.keys():
            action = key_to_action[key]
            self.step(action)
        else:
            self.step(Actions.idle)


if __name__ == "__main__":

    env=PlaygroundEnv(render_mode="human")
    env.reset(seed=42)

    manual_control = ManualControl(env, seed=1234)
    manual_control.start()
