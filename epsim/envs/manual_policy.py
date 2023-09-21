# from __future__ import annotations
# #from .parallel_epsim import parallel_env
# from epsim.core import *

# import pygame

# class ManualControl:
#     def __init__(
#         self,
#         env
#     ) -> None:
#         self.env = env
#         self.running = True


#     def start(self):
#         self.reset()
#         while self.running:
#             for event in pygame.event.get():
#                 if event.type == pygame.QUIT:
#                     self.running = False
#                 elif event.type==pygame.K_q:
#                         self.running = False
#                         break
#                 elif event.type == pygame.KEYDOWN:
#                     key=pygame.key.name(event.key)
#                     self.key_handler(key)
#                     self.env.render()
#         pygame.quit()

#     def reset(self):
#         obs, info = self.env.reset()
#         self.env.render()

        

#     def step(self, action):
#         obs, reward, terminated, truncated,info = self.env.step(action)
#         dones=terminated or truncated
#         if dones:
#             self.reset()



        

#     def key_handler(self,key):
#         #print(key)
#         if key == "escape":
#             self.env.close()
#             self.running=False
#             return
#         if key == "backspace":
#             self.reset()
#             return


#         key_to_action = {
#             "left shift": DispatchAction.NEXT_PRODUCT_TYPE,
#             "space": DispatchAction.SELECT_CUR_PRODUCT
#         }
        
#         if key in key_to_action:
#             self.step(key_to_action[key])
#         else:
#             self.step(DispatchAction.NOOP)
        
        