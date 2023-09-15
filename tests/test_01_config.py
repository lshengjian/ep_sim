import  sys
from os import path
dir=path.abspath(path.dirname(__file__) + './..')
sys.path.append(dir)

from epsim.core import split_field,get_files,get_file_info,build_config

def test_split_field():
    assert split_field('1.3')==[1]
    assert split_field('1|3|5')==[1,3,5]
    assert split_field('1~5')==[1,2,3,4,5]

def test_list_files():
    fs=get_files('test01')
    assert len(fs)==4
    name,field_names,data=get_file_info(fs[0])
    assert name=='operates'
    assert field_names==['key', 'name', 'color']
    assert len(data)==8



def test_build():
    ops_map,slots,cranes,procs=build_config('test01')
    assert ops_map[101].name=='上料'
    assert len(slots)==3
    assert str(slots[0]) =='[1] 上料 1|2'
    assert len(cranes)==1
    assert str(cranes[0])=='[1] H11 3 (1.0,1.0)'
    assert len(procs)==3
    assert str(procs[0]) =='[A] 上料 0->0'

    





import hydra
@hydra.main(config_path="../config", config_name="args", version_base="1.3")
def main(cfg: "DictConfig"):  # noqa: F821
    assert cfg.tile_size==48
    assert cfg.products[0].code=='A' and cfg.products[0].num==3

if __name__ == "__main__":
    #main()
    test_list_files()

    
