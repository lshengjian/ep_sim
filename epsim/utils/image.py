from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

import torch 
from PIL import Image
from einops import rearrange
from os import path
transform=transforms.Compose([
            transforms.ToPILImage(),
            #transforms.Grayscale(),
            transforms.Resize((84, 84)),
            # Converts to tensor and from [0,255] to [0,1]
            transforms.ToTensor(),
            # For a tensor in range (0, 1), this will convert to range (-1, 1)
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

def imgs2cnn(imgs,width=84):
    rt=[]
    for img in imgs:
        rt.append(img2cnn(img,width,True).numpy())
    return torch.from_numpy(np.array(rt,dtype=np.float32))


def img2cnn(img,width=84,keep_dim=False):
    t= transform if width==84 else \
    transforms.Compose([
            transforms.ToPILImage(),
            #transforms.Grayscale(),
            transforms.Resize((width, width)),
            # Converts to tensor and from [0,255] to [0,1]
            transforms.ToTensor(),
            # For a tensor in range (0, 1), this will convert to range (-1, 1)
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    rt=t(img)
    if not keep_dim:
        rt=rt.squeeze() #去掉只有一个数据的维度
    return rt

def load_img(fname):
    return Image.open(fname)
    
def to_gray(data):
    img=Image.fromarray(data,'RGB')
    return img.convert('L')

def load_img_data(fname):
    img = Image.open(fname)
    return np.asarray(img)

def save_img(data,fname,mode="RGB"):
    img = Image.fromarray(data,mode) #mode="L","RGB"
    img.save(fname)

def data2img(data,fname='demo.jpg',mode="RGB"):
    data=np.transpose(data*255, axes=(1, 2, 0))
    save_img(np.array(data,dtype=np.uint8),fname,mode)


#分割图片
# def cut_image(image,p_width,p_height):
#     width, height = image.size
#     h = height//p_height
#     w = width//p_width
#     boxs = []
#     for r in range(h):
#         for c in range(w):    
#             # (left, upper, right, lower)
#             box = (c*p_width,r*p_height,(c+1)*p_width,(r+1)*p_height)
#             boxs.append(box)
#     return [image.crop(box) for box in boxs]
def cut_image(image,p_width,p_height):
    data=np.asarray(image)
    return rearrange(data, '(h p1) (w p2) c -> h w p1 p2 c', p1 = p_height, p2 = p_width) # 转置，对角线对称

def sort_images(images,reverse_even_rows=False):
    rows,cols,*_=images.shape
    rt=[]
    for r in range(rows):
        tp=[]
        tp.extend(images[r])
        if r>0 and r%2==0 and reverse_even_rows:
            tp.reverse()
        rt.extend(tp)
    return rt



#保存分割后的图片
def save_images(data,file_name):
    index = 1
    for d in data:
        img=Image.fromarray(d,'RGB')
        img.save( file_name+'_' + str(index) + '.jpg')
        index+=1



if __name__ == "__main__":
    fname=path.abspath(path.dirname(__file__) + '/../../images/demo.jpg')
    img= Image.open(fname).resize((600,800),Image.Resampling.BILINEAR)
    print(img.size)
    images = cut_image(np.asarray(img), 100, 100)
    print(images.shape)
    # plt.imshow(images[0][0])
    # plt.show()
    data=sort_images(images,True)
    save_images(data,'demo')

    
    # 
    # imgs=cut_image(img,300,400)
    # save_images(imgs,'demo')
    # data=np.asarray(img)
    # print(data.shape)
    # data = rearrange(torch.Tensor(data), '(h p1) (w p2) c -> h w p1 p2 c', p1 = 100, p2 = 100)
    # print(data.shape)
    # img=Image.fromarray(data[0][0].numpy(),'RGB')
