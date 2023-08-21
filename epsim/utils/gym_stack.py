import gym
from gym.wrappers import FrameStack,GrayScaleObservation,ResizeObservation

def stack_imgs(env,nb_frames=4,size=64):
    env=GrayScaleObservation(env,keep_dim=False)
    env=ResizeObservation(env,size)
    env=FrameStack(env,nb_frames)
    return env

# def make_env(env_id,**kws):
#     def get_env():
#         env = gym.make(env_id,kws)
#         return stack_imgs(env)

    return get_env
if __name__ == "__main__":
    import gym,time
    
    env = gym.make('CarRacing-v2',continuous=False)
    env=stack_imgs(env)
    print(env.observation_space.shape)

    env_fns = [lambda: stack_imgs(gym.make('CarRacing-v2',continuous=False))] * 8
    envs = gym.vector.AsyncVectorEnv(env_fns, shared_memory=True) #SyncVectorEnv
    objs,infos=envs.reset()
            
    print(envs.observation_space.shape)
    print(envs.action_space)
    
    print(objs[0].shape)
    time.sleep(2.0)
    envs.close()
    