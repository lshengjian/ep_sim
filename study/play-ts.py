from tianshou.data import Collector
from tianshou.env import DummyVectorEnv, PettingZooEnv
from tianshou.policy import MultiAgentPolicyManager, RandomPolicy
from epsim.envs.electroplating_v1 import env
import hydra
@hydra.main(config_path="./config", config_name="args", version_base="1.3")
def main(cfg: "DictConfig"):  # noqa: F821
   e = env(render_mode="human",args=cfg)
   # Step 2: Wrap the environment for Tianshou interfacing
   e = PettingZooEnv(e)

   # Step 3: Define policies for each agent
   policies = MultiAgentPolicyManager([RandomPolicy(), RandomPolicy()], e)

   # Step 4: Convert the env to vector format
   e = DummyVectorEnv([lambda: e])

   # Step 5: Construct the Collector, which interfaces the policies with the vectorised environment
   collector = Collector(policies, e)

   # Step 6: Execute the environment with the agents playing for 1 episode, and render a frame every 0.1 seconds
   result = collector.collect(n_episode=1, render=0.016)

if __name__ == "__main__":
    main()

    