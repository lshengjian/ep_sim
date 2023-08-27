import  sys
from os import path
dir=path.abspath(path.dirname(__file__) + './..')
sys.path.append(dir)

from epsim.utils import split_field
from epsim.core.config import get_files,get_file_info,build_config,prodct_operates

def test_split_field():
    assert split_field('1.3')==[1.3]
    assert split_field('1|3|5')==[1,3,5]
    assert split_field('1~5')==[1,2,3,4,5]

def test_list_files():
    fs=get_files()
    assert len(fs)==4

def test_file_info():
    fs=get_files()
    name,field_names,data=get_file_info(fs[0])
    assert name=='1-operates'
    assert field_names==['key', 'name', 'color']
    assert len(data)==8

def test_build():
    cfg=build_config()
    assert str(cfg['1-operates'][0])=='[1]上料 (5, 25, 255)'
    assert str(cfg['2-slots'][0])=='(1,) 上料 (1, 2)'
    assert str(cfg['3-cranes'][0])=='(1)H1 1 (1.0,1.0)'
    assert str(cfg['4-procedures'][0])=='[A]上料 0->0'

def test_product_proc():
    cfg=build_config()
    ps1=prodct_operates(cfg['4-procedures'],'A')
    ps2=prodct_operates(cfg['4-procedures'],'B')
    assert len(ps2)==len(ps1)+2


import hydra
@hydra.main(config_path="../config", config_name="args", version_base="1.3")
def main(cfg: "DictConfig"):  # noqa: F821
    assert cfg.tile_size==[96,32]
    assert cfg.FPS==24
    assert cfg.products[0].code=='A' and cfg.products[0].num==3

if __name__ == "__main__":
    main()

    
