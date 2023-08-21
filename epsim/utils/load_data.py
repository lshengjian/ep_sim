import yaml
import os
#from io import StringIO
from os import path
import numpy as np

def load_config(fname):
    fpath=os.path.abspath(os.path.dirname(__file__) + '/../config/'+fname)
    with  open(fpath) as f:
        data=yaml.load(f,Loader=yaml.Loader) #encoding='utf-8'
    return data

# s = StringIO("1.618, 2.296\n3.141, 4.669\n")
# conv = {
#     0: lambda x: np.floor(float(x)),  # conversion fn for column 0
#     1: lambda x: np.ceil(float(x)),  # conversion fn for column 1
# }
# data=np.loadtxt(s, delimiter=",", converters=conv)
# print(data)

def load_data(fpath):
    with  open(fpath) as f:
        data=np.loadtxt(f, delimiter=",")
    return data

def get_stat(data):
    steps=len(data[0])
    rt=np.zeros((steps,2))
    for i in range(steps):
        cols=data[:,i]
        rt[i,:]=[np.mean(cols),np.std(cols)]
    return rt

if __name__ == "__main__":
    fpath=path.abspath(path.dirname(__file__) + '/mock_result.txt')
    data=load_data(fpath)
    print(data)
    data=get_stat(data)
    print(data)