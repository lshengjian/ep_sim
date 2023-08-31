import  sys
from os import path

dir=path.abspath(path.dirname(__file__) + './..')
sys.path.append(dir)
from epsim.core import build_config,World,Workpiece,Actions

def test_cranes():
    world=World()
    world.reset(['A']*3)
    
    assert world.starts[1].carrying!=None
    start=world.starts[0]
    wp:Workpiece=start.carrying
    assert wp.y==1 and  wp.x==1

    crane=world.group_cranes[1][0]
    assert crane.y==2 and  crane.x==1
    crane.set_command(Actions.top)
    world.update()
    assert crane.y==1 and  crane.x==1 
    assert  wp.attached==crane and crane.carrying==wp
    assert wp.y==1 and  wp.x==1 
    assert wp.target_op_limit.op_name=='镀银'
    crane.set_command(Actions.top)
    world.update()
    assert wp.y==0 and  wp.x==1 
    for i in range(8):
        crane.set_command(Actions.right)
        world.update()
    assert wp.y==0 and  wp.x==9
    crane.set_command(Actions.bottom)
    world.update()
    assert wp.y==1
    crane.set_command(Actions.bottom)
    world.update()
    assert crane.carrying==None
    

 


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