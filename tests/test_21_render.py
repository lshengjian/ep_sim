import  sys
from os import path
import numpy as np
from PIL import Image
dir=path.abspath(path.dirname(__file__) + './..')
sys.path.append(dir)
from epsim.core.rendering import point_in_circle,fill_coords,set_color,point_in_polygon


def test_image():
    img=np.zeros((64,64,3),dtype=np.uint8)
    fill_coords(img,point_in_polygon(0.5,0.8,0.2,6))
    #fill_coords(img, point_in_circle(0.5,0.5,0.45),(255,255,255))
    img=set_color(img,255,0,0)
    image = Image.fromarray(img)
    img2=np.zeros((64,64,4),dtype=np.uint8)
    for i in range(64):
        for j in range(64):
            img2[i,j,:3]=img[i,j]
            if (img[i,j]==0).all() :
                img2[i,j,3]=0
            else:
                img2[i,j,3]=255
    image2 = Image.fromarray(img2)
    image.save(dir+'/tests/demo.jpg')
    image2.save(dir+'/tests/demo.png')

if __name__ == "__main__":
   test_image()     