#from ..core.componets import OperateData
from .rendering import *
from  functools  import lru_cache

@lru_cache(128)
def get_shape(op_key:int):  # facage method
    if op_key<10:
        return make_buff()
    return make_tank()


def make_buff():
    print('make buff')
    img=np.zeros((64,64,3),dtype=np.uint8)
    fill_coords(img,point_in_polygon(0.5,0.5,0.4,3))
    return img

def make_tank():
    print('make tank')
    img=np.zeros((64,64,3),dtype=np.uint8)
    #fill_coords(img,point_in_circle(0.5,0.5,0.3))
    tri_fn = point_in_triangle(
                (0.12, 0.19),
                (0.87, 0.50),
                (0.12, 0.81),
            )

            # Rotate the agent based on its direction
    tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi*2 )
    fill_coords(img, tri_fn)
    return img


    
    