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
        obs, self.infos = self.env.reset()
        self.env.render()
        #print(obs['H11'].shape)
        

    def step(self, actions):
        obs, rewards, terminateds, truncateds,self.infos = self.env.step(actions)
        #key=self.env.world.cur_crane.cfg.name
        #print(obs[key].shape)
        #save_img(obs[key],'outputs/'+key+'.jpg')
        dones=list(terminateds.values())+list(truncateds.values())
        if any(dones):
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
        if key == "left ctrl":
            self.env.world.next_crane()


        key_to_action1 = {
            "left shift": DispatchAction.NEXT_PRODUCT_TYPE,
            "space": DispatchAction.SELECT_CUR_PRODUCT,

        }
        key_to_action2 = {
            "left": CraneAction.left,
            "right": CraneAction.right,
            "down": CraneAction.bottom , 
            "up": CraneAction.top,
        }
        actions={SHARE.DISPATCH_CODE:DispatchAction.NOOP}
        
        if key in key_to_action1 :
            action = key_to_action1[key]
            if self.env.world._masks[SHARE.DISPATCH_CODE][action]>0 :
                actions[SHARE.DISPATCH_CODE]=action


        for carne in self.env.world.all_cranes:
            actions[carne.cfg.name]=CraneAction.stay
        if key in key_to_action2:
            action = key_to_action2[key]
            carne=self.env.world.cur_crane
            #print('action_masks',self.env.infos[carne.cfg.name]['action_masks'])
            if  self.env.world._masks[carne.cfg.name][action]:
                actions[carne.cfg.name]=action
        self.step(actions)
        
        