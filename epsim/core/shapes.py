#from ..core.componets import OperateData
from .world_object import WorldObj
from .rendering import *
from  functools  import lru_cache
import re


CHS=3

def is_uper(string):
    pattern = r'^[0-9]+$'  # 正则表达式模式，表示只包含数字的字符串
    if re.match(pattern, string):
        return True
    else:
        return False
def is_A2Z(string):
    pattern = r'^[A-Z]$'
    if re.match(pattern, string):
        return True
    else:
        return False
    
def is_number(string):
    pattern = r'^[0-9]+$'  # 正则表达式模式，表示只包含数字的字符串
    if re.match(pattern, string):
        return True
    else:
        return False

@lru_cache(4)
def get_crane_shape(dir:int):  
    #print('make HeadCrane')
    img=np.zeros((WorldObj.TILE_SIZE,WorldObj.TILE_SIZE,3),dtype=np.uint8)
    if dir==0:
        return fill_coords(img,point_in_rect(0.3,0.7,0.4,0.6))

    #fill_coords(img,point_in_circle(0.5,0.5,0.3))
    tri_fn = point_in_triangle(
                (0.2, 0.29),
                (0.8, 0.50),
                (0.2, 0.71),
            )

    d=dir-1
    tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=-0.5 * math.pi*d )
    fill_coords(img, tri_fn)
    return img
    

@lru_cache(26)
def get_workpiece_shape(prd_code:str='A'):  # facage method
    '''
    A~Z : 产品1~26
    '''
    return make_workpiece(ord(prd_code[0])-61)
def make_workpiece(num_side=3): 
    #print('make workpiece')
    img=np.zeros((WorldObj.TILE_SIZE,WorldObj.TILE_SIZE,CHS),dtype=np.uint8)
    fill_coords(img,point_in_rect(0,1,0,0.1))
    fill_coords(img,point_in_rect(0.48,0.52,0.1,0.6))
    fill_coords(img,point_in_polygon(0.5,0.62,0.3,num_side))
    return img

def make_buff():
    #print('make buff')
    img=np.zeros((WorldObj.TILE_SIZE*2,WorldObj.TILE_SIZE,CHS),dtype=np.uint8)
    fill_coords(img,point_in_rect(0.1,0.2,0.36,1))
    fill_coords(img,point_in_rect(0.8,0.9,0.36,1))

    fill_coords(img,point_in_rect(0,0.3,0.26,0.38))
    fill_coords(img,point_in_rect(0.7,1.0,0.26,0.38))
   
    return img

def make_tank():
    #print('make tank')
    img=make_buff()
    fill_coords(img,point_in_rect(0,1.0,0.95,1))
    for i in range(3):
        ylo = 0.6 + 0.1 * i
        yhi = 0.65 + 0.1 * i
        fill_coords(img, point_in_line(0.1, ylo, 0.3, yhi, r=0.03))
        fill_coords(img, point_in_line(0.3, yhi, 0.5, ylo, r=0.03))
        fill_coords(img, point_in_line(0.5, ylo, 0.7, yhi, r=0.03))
        fill_coords(img, point_in_line(0.7, yhi, 0.9, ylo, r=0.03))
    return img


def get_progress_bar(p:int=100):
    p=np.clip(p,1,100)
    n=int(WorldObj.TILE_SIZE*(100-p)/100)
    img=np.ones((WorldObj.TILE_SIZE//4,n,CHS),dtype=np.uint8)*255
    return img


@lru_cache(128)
def get_slot_shape(op_key=1):  # facage method
    '''
    1,上料
    2,交换
    3,下料
    11,除油
    12,水洗
    13,烘干
    21,镀铜
    22,镀银
    ......
    '''
    if op_key<10:
        return make_buff()
    return make_tank()
        


