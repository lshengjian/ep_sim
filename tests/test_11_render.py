import  sys
from os import path
import numpy as np
from PIL import Image
dir=path.abspath(path.dirname(__file__) + './..')
sys.path.append(dir)
from epsim.utils.rendering import point_in_circle,fill_coords
def set_color(img,r,g,b):
    img2=img.astype(np.float32)
    img2/=255.0
    img2[:,:]*=np.array([r,g,b],dtype=np.float32)
    img2*=255.0
    return img2.astype(np.uint8)

def test_image():
    img=np.zeros((64,64,3),dtype=np.uint8)
    fill_coords(img, point_in_circle(0.5,0.5,0.45),(255,255,255))
    img=set_color(img,1,1,0)
    image = Image.fromarray(img)
    
    image.save('output.jpg')

    