
from epsim.envs.myenv import MyEnv
from epsim.envs.manual_policy import ManualControl

import hydra
@hydra.main(config_path="./config", config_name="args", version_base="1.3")
def main(args: "DictConfig"):  # noqa: F821
    args.auto_put_starts=False
    env=MyEnv("human",args)
    manual_control = ManualControl(env)
    manual_control.start()

if __name__ == "__main__":
    main()

