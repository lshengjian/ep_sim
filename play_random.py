from epsim.envs.env import MyGrid



   
if __name__ == "__main__":
   env = MyGrid(render_mode="human",fps=4)
   observation, info = env.reset(seed=40)
   for _ in range(100):
      action = env.action_space.sample()  # this is where you would insert your policy
      observation, reward, terminated, truncated, info = env.step(action)

      if terminated or truncated:
         observation, info = env.reset()

   env.close()