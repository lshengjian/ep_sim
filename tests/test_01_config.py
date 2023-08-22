import hydra

@hydra.main(config_path="../config", config_name="demo", version_base="1.3")
def main(cfg: "DictConfig"):  # noqa: F821
    assert cfg.tile_size==[96,32]
    assert cfg.FPS==24
    assert cfg.slots[0].code=='S'
    assert cfg.cranes[0].code=='H1'

if __name__ == "__main__":
    main()