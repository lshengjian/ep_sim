import  sys
from os import path

dir=path.abspath(path.dirname(__file__) + './..')
sys.path.append(dir)
from epsim.core import build_config,World,Workpiece,Actions

def test_cranes():
    world=World()
    world.reset()
    world.add_jobs(['A']*3)
    start=world.starts[1]
    wp:Workpiece=start.carrying
    assert wp.y==1 and  wp.x==2
    assert  wp.attached==start and start.carrying==wp

    crane=world.group_cranes[1][0]
    assert   crane.x==3 and crane.y==2
    crane.set_command(Actions.left)
    world.update()
    assert crane.x==2 and crane.y==2
    crane.set_command(Actions.top)
    world.update()
    assert crane.x==2 and crane.y==1
    assert  wp.attached==crane and crane.carrying==wp
    assert wp.x==2 and wp.y==1
    assert wp.target_op_limit.op_name=='镀银'
    crane.set_command(Actions.top)
    world.update()
    for i in range(7):
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
    world.reset()
    world.add_jobs(['A']*3)
    start=world.starts[1]
    wp:Workpiece=start.carrying
    assert wp.y==1 and  wp.x==2
    assert  wp.attached==start and start.carrying==wp

    crane=world.group_cranes[1][0]
    assert   crane.x==3 and crane.y==2
    crane.set_command(Actions.left)
    world.update()
    assert crane.x==2 and crane.y==2
    crane.set_command(Actions.top)
    world.update()
    assert crane.x==2 and crane.y==1
    assert  wp.attached==crane and crane.carrying==wp
    assert wp.x==2 and wp.y==1
    assert wp.target_op_limit.op_name=='镀银'
    crane.set_command(Actions.top)
    world.update()
