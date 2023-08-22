import  sys
from os import path

dir=path.abspath(path.dirname(__file__) + './..')
sys.path.append(dir)
from epsim.core import World
def test_ok():
    wd=World(10)
    wd._build()
    wd.pprint()

    assert cfg.tile_size==[96,32]
    assert cfg.FPS==24
    assert cfg.slots[0].code=='S'
    assert cfg.cranes[0].code=='H1'

if __name__ == "__main__":
    main()