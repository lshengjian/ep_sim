from epsim.envs.playground import PlaygroundEnv



   
if __name__ == "__main__":
   env = PlaygroundEnv(render_mode="human")
   observation, info = env.reset(seed=40)
   for _ in range(1000):
      action = env.action_space.sample()  # this is where you would insert your policy
      observation, reward, terminated, truncated, info = env.step(action)

      if terminated or truncated:
         observation, info = env.reset()

   env.close()