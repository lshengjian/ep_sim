import numpy as np

from PIL import Image
from einops import rearrange
from os import path

__all__=['get_state','save_img','merge_images','merge_two_images','get_observation']


def load_img(fname):
    return Image.open(fname)

def save_img(data,fname,mode="RGB"):
    img = Image.fromarray(data,mode) #mode="L","RGB"
    img.save(fname)  

def cut_image(data,p_height,p_width):
    return rearrange(data, '(h p1) (w p2) c -> (h w) p1 p2 c', p1 = p_height, p2 = p_width) # 转置，对角线对称

def merge_images(machine_imgs:np.ndarray,cell_size=48):
    return rearrange(machine_imgs, 'n h w c -> h (n w) c',h=cell_size*3,w=cell_size)

def merge_two_images(machine_imgs:np.ndarray,products_imgs:np.ndarray,cell_size=48):
    imgs= np.concatenate((machine_imgs,products_imgs),axis=0)
    #print(imgs.shape)
    #return imgs
    return rearrange(imgs, 'n h w c -> h (n w) c',h=cell_size*3,w=cell_size)
def get_state(img:np.ndarray,num_pructs:int=5,ncols:int=17,max_x:int=32,tile_size:int=32):
    imgs=cut_image(img,tile_size*3,tile_size)
    #save_images(imgs,'cell')
    prds,slots= imgs[0:num_pructs],imgs[ncols:max_x+ncols]
    return slots[1:],prds

 
def get_observation(big_img:np.ndarray,x:int,view_distance=7,cell_size=48,max_x=32):
    assert x>0
    h,w,c=big_img.shape
    rt=np.zeros((h,cell_size*view_distance,c),dtype=np.uint8)
    half=view_distance//2
    x1,x2=int(x-half-1),int(x+half-1)
    if x1<0:x1=0
    if x2>=max_x:x2=max_x-1
    dis=x2-x1
    img=big_img[:,x1*cell_size:(x2+1)*cell_size]
    start=half-dis//2
    rt[:,start*cell_size:(start+dis+1)*cell_size,:]=img
    return rt




def merge_demo():
    fname=path.abspath(path.dirname(__file__) + '/../../outputs/state.jpg')
    img= Image.open(fname)#.resize((600,800),Image.Resampling.BILINEAR)
    data=np.asarray(img)
    slots,prds = get_state(data,6,20,32,48)
    img=merge_two_images(slots,prds)
    img=Image.fromarray(img,'RGB')
    img.save('outputs/state2.jpg')

def split_demo():
    fname=path.abspath(path.dirname(__file__) + '/../../outputs/state2.jpg')
    img= Image.open(fname)#.resize((600,800),Image.Resampling.BILINEAR)
    data=np.asarray(img)
    img=get_observation(data,3,7)
    img=Image.fromarray(img,'RGB')
    img.save('outputs/observation_7_3.jpg')
    img=get_observation(data,3,9)
    img=Image.fromarray(img,'RGB')
    img.save('outputs/observation_9_3.jpg')
    img=get_observation(data,3,41)
    img=Image.fromarray(img,'RGB')
    img.save('outputs/observation_41_3.jpg')

if __name__ == "__main__":
    split_demo()

# def to_gray(data):
#     img=Image.fromarray(data,'RGB')
#     return img.convert('L')

# def load_img_data(fname):
#     img = Image.open(fname)
#     return np.asarray(img)




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


#保存分割后的图片
# def save_images(data,file_name):
#     index = 1
#     for d in data:
#         img=Image.fromarray(d,'RGB')
#         img.save('outputs/'+file_name+'_' + str(index) + '.jpg')
#         index+=1