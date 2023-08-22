import torch,torchvision
import numpy as np
import os,sys
from einops import rearrange

dir=os.path.abspath(os.path.dirname(__file__) + './..')


sys.path.append(dir)
from epsim.utils import load_img,img2cnn,imgs2cnn,save_img,load_img_data

# def test_load_image():
#     img=load_img(dir+'/images/piston_4.jpg')
#     img = img.convert('L')
#     fname=dir+'/tests/demo1.jpg'
#     img.save(fname)
#     data=load_img_data(dir+'/images/piston_7.jpg')
#     data=img2cnn(data,84,False)
#     assert data.shape==(3,84,84) 
#     data=torch.transpose(data, 0, 2)
#     assert data.shape==(84,84,3) 
#     data=np.array(data*255,dtype=np.uint8)
#     save_img(data,dir+'/tests/demo2.jpg','RGB') 


def test_random_images():
    imgs=np.random.randint(0,255,size=(5,86,86,3),dtype=np.uint8)
    
    data=imgs2cnn(imgs,43)#.numpy()
    assert data.shape==(5,3,43,43) 
    data=torch.transpose(data, 1, -1)
    img=np.array(data[0]*255,dtype=np.uint8) 
    assert img.shape==(43,43,3) 
    img[:,4:8,:]=255 #rows
    img[14:18,:,:]=0 #cols
    save_img(img,dir+'/tests/demo3.jpg','RGB') 

    # torchvision.utils.save_image(data,dir+'/tests/demo4.jpg')


    