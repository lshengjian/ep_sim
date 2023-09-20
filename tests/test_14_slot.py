import  sys
from os import path

dir=path.abspath(path.dirname(__file__) + './..')
sys.path.append(dir)
from epsim.core import build_config,World,Workpiece,CraneAction,SHARE

def test_slot_no_product_state():
    wd=World()
    wd.reset()
    wd.add_jobs(['B']*3)
    op_slot=wd.pos_slots[3]
    #wp:Workpiece=stat_slot.carrying
    data=op_slot.state2data()
    size=SHARE.OBJ_TYPE_SIZE+SHARE.OP_TYPE1_SIZE+SHARE.OP_TYPE2_SIZE+SHARE.PRODUCT_TYPE_SIZE+4

    start=0
    d=data[start:SHARE.OBJ_TYPE_SIZE]
    assert d.tolist()==[0.,1.,0.]  #可用加工槽位
    start+=SHARE.OBJ_TYPE_SIZE
    d=data[start:start+SHARE.OP_TYPE1_SIZE]
    assert d.tolist()==[0.,1.,0.]
    start+=SHARE.OP_TYPE1_SIZE
    d=data[start:start+SHARE.OP_TYPE2_SIZE]
    assert d.tolist()==[1.,0.,0.,0.,0.,0.]
    start+=SHARE.OP_TYPE2_SIZE
    d=data[start:start+SHARE.PRODUCT_TYPE_SIZE]
    assert d.tolist()==[0.,0.,0.]

    start+=SHARE.PRODUCT_TYPE_SIZE
    d=data[start:start+2]
    assert d.tolist()==[0.03,0.5]

    start+=2
    d=data[start:start+2]
    assert d.tolist()==[0.0,0.]

def test_slot_with_product_state():
    wd=World()
    wd.reset()
    wd.add_jobs(['B']*3)
    stat_slot=wd.pos_slots[1]
    #wp:Workpiece=stat_slot.carrying
    data=stat_slot.state2data()
    size=SHARE.OBJ_TYPE_SIZE+SHARE.OP_TYPE1_SIZE+SHARE.OP_TYPE2_SIZE+SHARE.PRODUCT_TYPE_SIZE+4
    assert size==19
    assert size==len(data)

    start=0
    d=data[start:SHARE.OBJ_TYPE_SIZE]
    assert d.tolist()==[0.,0.,1.]  #有加工物料，登记为产品，否则算可用加工槽位
    start+=SHARE.OBJ_TYPE_SIZE
    d=data[start:start+SHARE.OP_TYPE1_SIZE]
    assert d.tolist()==[1.,0.,0.]
    start+=SHARE.OP_TYPE1_SIZE
    d=data[start:start+SHARE.OP_TYPE2_SIZE]
    assert d.tolist()==[1.,0.,0.,0.,0.,0.]
    start+=SHARE.OP_TYPE2_SIZE
    d=data[start:start+SHARE.PRODUCT_TYPE_SIZE]
    assert d.tolist()==[0.,1.,0.]

    start+=SHARE.PRODUCT_TYPE_SIZE
    d=data[start:start+2]
    assert d.tolist()==[0.01,0.5]

    start+=2
    d=data[start:start+2]
    assert d.tolist()==[0.,0.]




if __name__ == "__main__":
    test_slot_state()    
    

