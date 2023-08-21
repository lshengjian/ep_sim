import hydra

@hydra.main(config_path="./epsim/config", config_name="demo", version_base="1.3")
def main(cfg: "DictConfig"):  # noqa: F821
    print(cfg.SCREEN)
    print(cfg.FPS)
    print(cfg.slots[0])
    print(cfg.cranes[0])

if __name__ == "__main__":
    main()