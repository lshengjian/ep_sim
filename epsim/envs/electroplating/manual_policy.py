from __future__ import annotations
#from .parallel_epsim import parallel_env
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
        #self.cur_crane_idx=0
        self.info={'action_mask':[1]*5}

    def start(self):
        self.reset()
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type==pygame.K_q:
                        self.running = False
                        break
                elif event.type == pygame.KEYDOWN:
                    

                    #print(event.key)
                    key=pygame.key.name(event.key)
                    self.key_handler(key)
                    self.env.render()
        pygame.quit()

    def reset(self):
        obs, self.infos = self.env.reset(seed=123)
        self.env.world.shift_product()
        self.env.render()
        #print(obs['H11'].shape)
        

    def step(self, actions):
        obs, rewards, terminateds, truncateds,self.infos = self.env.step(actions)
        key=self.env.world.cur_crane.cfg.name
        #print(obs[key].shape)
        save_img(obs[key],'outputs/'+key+'.jpg')
        dones=list(terminateds.values())
        if dones[0]:
            #print("game over!")
            self.reset()



        

    def key_handler(self,key):
        #print(key)
        if key == "escape":
            self.env.close()
            self.running=False
            return
        if key == "backspace":
            self.reset()
            return
        if key == "left shift":
            self.env.world.shift_product()
            return
        if key == "left ctrl":
            #print('next_crane')
            self.env.world.next_crane()
            return
        if key == "space":
            self.env.world.put_product()
            self.env.render()
            return

        key_to_action = {
            "left": Actions.left,
            "right": Actions.right,
            "down": Actions.bottom , 
            "up": Actions.top,
        }
        actions={}#np.zeros(len(self.env.world.all_cranes),dtype=np.uint8)
        for carne in self.env.world.all_cranes:
            actions[carne.cfg.name]=Actions.stay
        if key in key_to_action.keys():
            action = key_to_action[key]
            carne=self.env.world.cur_crane
            #print('action_masks',self.env.infos[carne.cfg.name]['action_masks'])
            if  self.env.infos[carne.cfg.name]['action_masks'][action]:
                actions[carne.cfg.name]=action
        self.step(actions)
        
        