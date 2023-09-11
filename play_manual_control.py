
from epsim.envs.electroplating_v1 import parallel_env,ManualControl

import hydra
@hydra.main(config_path="./config", config_name="args", version_base="1.3")
def main(args: "DictConfig"):  # noqa: F821
    args.auto_dispatch=False
    env=parallel_env("human",args)
    manual_control = ManualControl(env)
    manual_control.start()

if __name__ == "__main__":
    main()

