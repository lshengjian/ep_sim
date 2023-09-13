import os
from typing import Optional, Tuple

import gymnasium
import numpy as np
import torch
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import BasePolicy, DQNPolicy, MultiAgentPolicyManager, RandomPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils.net.common import Net

from epsim.envs.electroplating_v1 import env
import hydra
def _get_agents(
    agent_learn: Optional[BasePolicy] = None,
    optim: Optional[torch.optim.Optimizer] = None,
) -> Tuple[BasePolicy, torch.optim.Optimizer, list]:
    e = _get_env()
    # observation_space = (
    #     env.observation_space["observation"]
    #     if isinstance(env.observation_space, gymnasium.spaces.Dict)
    #     else env.observation_space
    # )
    if agent_learn is None:
        # model
        net = Net(
            state_shape=15*19,#observation_space.shape or observation_space.n,
            action_shape=5,#env.action_space.shape or env.action_space.n,
            hidden_sizes=[128, 128, 128, 128],
            device="cuda" if torch.cuda.is_available() else "cpu",
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        if optim is None:
            optim = torch.optim.Adam(net.parameters(), lr=1e-4)
        agent_learn = DQNPolicy(
            model=net,
            optim=optim,
            discount_factor=0.9,
            estimation_step=3,
            target_update_freq=320,
        )

    # if agent_opponent is None:
    #     agent_opponent = RandomPolicy()

    agents = [agent_learn, agent_learn]
    policy = MultiAgentPolicyManager(agents, e)
    return policy, optim, e.agents

CFG=None

def _get_env():
    """This function is needed to provide callables for DummyVectorEnv."""
    return PettingZooEnv(env(render_mode=None,args=CFG))


@hydra.main(config_path="./config", config_name="args", version_base="1.3")
def main(cfg: "DictConfig"):  # noqa: F821
    global CFG
    CFG=cfg
    # e1=env(render_mode="human",args=CFG)
    # e2=env(render_mode="human",args=CFG)
    # ======== Step 1: Environment setup =========
    train_envs = DummyVectorEnv([_get_env for _ in range(10)])
    test_envs = DummyVectorEnv([_get_env for _ in range(10)])

    # seed
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_envs.seed(seed)
    test_envs.seed(seed)

    # ======== Step 2: Agent setup =========
    policy, optim, agents = _get_agents()

    # ======== Step 3: Collector setup =========
    train_collector = Collector(
        policy,
        train_envs,
        VectorReplayBuffer(20_000, len(train_envs)),
        exploration_noise=True,
    )
    test_collector = Collector(policy, test_envs, exploration_noise=True)
    # policy.set_eps(1)
    train_collector.collect(n_step=64 * 10)  # batch size * training_num

    # ======== Step 4: Callback functions setup =========
    def save_best_fn(policy):
        model_save_path = os.path.join("log", "ttt", "dqn", "policy2.pth")
        os.makedirs(os.path.join("log", "ttt", "dqn"), exist_ok=True)
        torch.save(policy.policies[agents[1]].state_dict(), model_save_path)

    def stop_fn(mean_rewards):
        return mean_rewards >= 4

    def train_fn(epoch, env_step):
        policy.policies[agents[0]].set_eps(0.1)
        policy.policies[agents[1]].set_eps(0.1)

    def test_fn(epoch, env_step):
        policy.policies[agents[0]].set_eps(0.05)
        policy.policies[agents[1]].set_eps(0.05)

    def reward_metric(rews):
        return rews[:, 1]

    # ======== Step 5: Run the trainer =========
    result = offpolicy_trainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=50,
        step_per_epoch=1000,
        step_per_collect=50,
        episode_per_test=10,
        batch_size=64,
        train_fn=train_fn,
        test_fn=test_fn,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        update_per_step=0.1,
        test_in_train=False,
        reward_metric=reward_metric,
    )

    # return result, policy.policies[agents[1]]
    print(f"\n==========Result==========\n{result}")
    print("\n(the trained policy can be accessed via policy.policies[agents[1]])")




if __name__ == "__main__":
    main()
