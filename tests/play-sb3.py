"""Uses Stable-Baselines3 to train agents in the Connect Four environment using invalid action masking.

For information about invalid action masking in PettingZoo, see https://pettingzoo.farama.org/api/aec/#action-masking
For more information about invalid action masking in SB3, see https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html

"""
import glob
import os
import time
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker

import pettingzoo.utils
#from pettingzoo.classic import connect_four_v3
from epsim.envs import electroplating_v1

class SB3ActionMaskWrapper(pettingzoo.utils.BaseWrapper):
    """Wrapper to allow PettingZoo environments to be used with SB3 illegal action masking."""

    def reset(self, seed=None, options=None):
        """Gymnasium-like reset function which assigns obs/action spaces to be the same for each agent.

        This is required as SB3 is designed for single-agent RL and doesn't expect obs/action spaces to be functions
        """
        super().reset(seed, options)

        # Strip the action mask out from the observation space
        self.observation_space = super().observation_space(self.possible_agents[0])["observation"]
        self.action_space = super().action_space(self.possible_agents[0])

        # Return initial observation, info (PettingZoo AEC envs do not by default)
        return self.observe(self.agent_selection), {}

    def step(self, action):
        """Gymnasium-like step function, returning observation, reward, termination, truncation, info."""
        super().step(action)
        return super().last()

    def observe(self, agent):
        """Return only raw observation, removing action mask."""
        #print(agent,super().observe(agent))
        return self.unwrapped.observations[agent]

    def action_mask(self):
        #print(self.unwrapped.infos)
        """Separate function used in order to access the action mask."""
        return self.unwrapped.infos[self.agent_selection]["action_masks"]


def mask_fn(env):
    # Do whatever you'd like in this function to return the action mask
    # for the current env. In this example, we assume the env has a
    # helpful method we can rely on.
    return env.action_mask() #np.ones(5,dtype=np.uint8) 


def train_action_mask(env_fn, steps=10_000, seed=0, **env_kwargs):
    """Train a single model to play as each agent in a zero-sum game environment using invalid action masking."""
    env = env_fn.env(**env_kwargs)#render_mode='human',

    print(f"Starting training on {str(env.metadata['name'])}.")

    # Custom wrapper to convert PettingZoo envs to work with SB3 action masking
    env = SB3ActionMaskWrapper(env)

    env.reset(seed=seed)  # Must call reset() in order to re-define the spaces

    env = ActionMasker(env, mask_fn)  # Wrap to enable masking (SB3 function)
    # MaskablePPO behaves the same as SB3's PPO unless the env is wrapped
    # with ActionMasker. If the wrapper is detected, the masks are automatically
    # retrieved and used when learning. Note that MaskablePPO does not accept
    # a new action_mask_fn kwarg, as it did in an earlier draft.

    model = MaskablePPO(MaskableActorCriticPolicy, env, verbose=1)

    model.set_random_seed(seed)
    model.learn(total_timesteps=steps)

    model.save(f"outputs/{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}")

    print("Model has been saved.")

    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.\n")

    env.close()


def eval_action_mask(env_fn, num_games=100, render_mode=None, **env_kwargs):
    # Evaluate a trained agent vs a random agent
    env = env_fn.env(render_mode=render_mode,**env_kwargs) #render_mode=None,

    model = load(env)

    scores = {agent: 0 for agent in env.possible_agents}
    total_rewards = {agent: 0 for agent in env.possible_agents}
    round_rewards = []

    for i in range(num_games):
        env.reset(seed=i)
        env.action_space(env.possible_agents[0]).seed(i)

        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            ms=info['action_masks']
            # if sum(ms)<2:
            #     print(ms)
            observation=obs

            if termination or truncation:
                
                for a in env.possible_agents:
                    total_rewards[a] += env.rewards[a]

                break
            else:
                # if agent == env.possible_agents[-1]:
                #     act = env.action_space(agent).sample(env.infos[agent]['action_masks'])
                # else:
                    # Note: PettingZoo expects integer actions # TODO: change chess to cast actions to type int?
                act = int(
                    model.predict(
                        observation,action_masks=ms,  deterministic=True #
                    )[0]
                )
                #print(act)
            env.step(act)
    env.close()


    print("Total rewards (incl. negative rewards): ", total_rewards)

    return  total_rewards

def load(env):
    model =None
    try:
        latest_policy = max(
            glob.glob(f"outputs/{env.metadata['name']}*.zip"), key=os.path.getctime
        )
        model= MaskablePPO.load(latest_policy)
    except ValueError:
        print("Policy not found.")
    
    return model
import hydra
@hydra.main(config_path="./config", config_name="args", version_base="1.3")
def main(cfg: "DictConfig"):  # noqa: F821
    env_fn = electroplating_v1

    env_kwargs = {'args':cfg}
    train_action_mask(env_fn, steps=20000, seed=123, **env_kwargs)
    eval_action_mask(env_fn, num_games=100,render_mode='human', **env_kwargs)



if __name__ == "__main__":
    main()
    