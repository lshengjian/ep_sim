import numpy as np
def type2onehot(type_key:int,size:int):
    type_idx:int=type_key-1
    rt=np.zeros(size,dtype=np.float32)
    if type_idx>=0:
        rt=np.eye(size,dtype=np.float32)[type_idx]
    return rt

def op_ket2onehots(op_key:int,size1=3,size2=5):
    y1=type2onehot(int(op_key/100),size1)
    y2=type2onehot(int(op_key%100),size2)
    return np.concatenate((y1,y2))
if __name__ == "__main__":
   assert list(type2onehot(0,3))==[0.0,0.0,0.0]
   assert list(type2onehot(1,3))==[1.0,0.0,0.0]
   assert list(type2onehot(2,3))==[0.0,1.0,0.0]
   assert list(type2onehot(3,3))==[0.0,0.0,1.0]
   assert list(op_ket2onehots(0))   == [0. ,0. ,0. ,0. ,0. ,0. ,0. ,0.]
   assert list(op_ket2onehots(102)) == [1. ,0. ,0. ,0. ,1. ,0. ,0. ,0.]
   assert list(op_ket2onehots(205)) == [0. ,1. ,0. ,0. ,0. ,0. ,0. ,1.]
#    print(op_ket2onehots(205))