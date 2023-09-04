import numpy as np

from PIL import Image
from einops import rearrange
from os import path




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
def cut_image(data,p_height,p_width):
    return rearrange(data, '(h p1) (w p2) c -> h w p1 p2 c', p1 = p_height, p2 = p_width) # 转置，对角线对称

def sort_images(images):
    rows,cols,*_=images.shape
    rt=[]
    for r in range(rows):
        tp=[]
        tp.extend(images[r])
        # if r>0 and r%2==0 and reverse_even_rows:
        #     tp.reverse()
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
    fname=path.abspath(path.dirname(__file__) + '/../../state.jpg')
    img= Image.open(fname)#.resize((600,800),Image.Resampling.BILINEAR)
    data=np.asarray(img)
    
    images = cut_image(data,  48*3,48)
    print(images.shape)
    data=sort_images(images)
    save_images(data,'demo')

    
    # 
    # imgs=cut_image(img,300,400)
    # save_images(imgs,'demo')
    # data=np.asarray(img)
    # print(data.shape)
    # data = rearrange(torch.Tensor(data), '(h p1) (w p2) c -> h w p1 p2 c', p1 = 100, p2 = 100)
    # print(data.shape)
    # img=Image.fromarray(data[0][0].numpy(),'RGB')
