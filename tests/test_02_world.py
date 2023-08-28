import  sys
from os import path

dir=path.abspath(path.dirname(__file__) + './..')
sys.path.append(dir)
from epsim.core import build_config,World,Workpiece,Actions

def test_world():
    wd=World()
    assert len(wd.group_cranes)==2
    assert wd.group_slots[1][-1]==wd.group_slots[2][0] # swap

def test_cranes():
    world=World()
    wp=Workpiece(0,'A')
    world.plan_next(wp)
    start=world.get_free_slot(1,wp)
    world.attach(wp,start)
    assert wp.y==1 and  wp.x==1
    #assert wp.target_op.op_key==start.cfg.op_key
    start2=world.get_free_slot(1,wp)
    assert start!=start2

    crane=world.group_cranes[1][0]
    assert crane.y==2 and  crane.x==1
    crane.set_command(Actions.up)
    world.update()
    assert crane.y==1 and  crane.x==1 
    assert  wp.attached==crane and crane.carrying==wp
    assert wp.y==1 and  wp.x==1 
    #world.plan_next(wp)
    assert wp.target_op.op_name=='镀银'
    world.update()
    assert wp.y==0 and  wp.x==1 
    crane.set_command(Actions.forward)
    world.update()
    world.update()
    assert wp.y==0 and  wp.x==3
    crane.set_command(Actions.down)
    world.update()
    world.update()

        




    #     crane.set_command(Actions.down)
    #     self.update()
    #     assert wp.y==1 and  wp.x==3
    #     # self.pprint()
    #     # print('='*18) 
    #     self.update()
    #     self.pprint()
       
    #     assert wp.y==1 and  wp.x==3
    #     assert crane.y==2 and  crane.x==3

    #     crane.set_command(Actions.stay)
    #     self.update()
    #     assert crane.y==2 and  crane.x==3
    #     print('='*18)
    #     self.pprint()
        
     
   

    # assert cfg.tile_size==[96,32]
    # assert cfg.FPS==24
    # assert cfg.slots[0].code=='S'
    # assert cfg.cranes[0].code=='H1'

if __name__ == "__main__":
    world=World()
    wp=Workpiece(0,'A')
    world.plan_next(wp)
    start=world.get_free_slot(1,wp)
    world.attach(wp,start)
    assert wp.y==1 and  wp.x==1
    #assert wp.target_op.op_key==start.cfg.op_key
    start2=world.get_free_slot(1,wp)
    assert start!=start2