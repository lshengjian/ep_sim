import numpy as np

from PIL import Image
#from einops import rearrange
from os import path

__all__=['get_state','save_img','get_observation']


def load_img(fname):
    return Image.open(fname)

def save_img(data,fname,mode="RGB"):
    img = Image.fromarray(data,mode) #mode="L","RGB"
    img.save(fname)  
'''
def cut_image(data,p_height,p_width):
    return rearrange(data, '(h p1) (w p2) c -> (h w) p1 p2 c', p1 = p_height, p2 = p_width) # 转置，对角线对称

def merge_images(machine_imgs:np.ndarray,cell_size=48):
    return rearrange(machine_imgs, 'n h w c -> h (n w) c',h=cell_size*3,w=cell_size)

def merge_two_images(machine_imgs:np.ndarray,products_imgs:np.ndarray,cell_size=48):
    imgs= np.concatenate((machine_imgs,products_imgs),axis=0)
    return rearrange(imgs, 'n h w c -> h (n w) c',h=cell_size*3,w=cell_size)
def get_state(img:np.ndarray,num_pructs:int=5,ncols:int=17,max_x:int=32,tile_size:int=32):
    imgs=cut_image(img,tile_size*3,tile_size)
    prds,slots= imgs[0:num_pructs],imgs[ncols:max_x+ncols]
    return slots[1:],prds
'''




def get_state(screen_img:np.ndarray,nrows=3,ncols=20,cell_size=48):
    h,w,c=screen_img.shape
    row_imgs=[]
    step=cell_size*3
    for r in range(nrows):
        row_img=np.zeros((cell_size*3,ncols*cell_size,c),dtype=np.uint8)
        start=r*step
        row_img[:,:,:]=screen_img[r*step:(r+1)*step,:,]
        row_imgs.append(row_img)
    head=row_imgs[0]
    row_imgs.remove(head)
    row_imgs.append(head)
    return np.concatenate(row_imgs,axis=1)

def get_observation(state_img:np.ndarray,x:int,view_distance=7,cell_size=48,max_x=32):
    assert x>0
    h,w,c=state_img.shape
    total=1+2*view_distance
    rt=np.zeros((h,cell_size*total,c),dtype=np.uint8)
    x1,x2=int(x-view_distance),int(x+view_distance)
    if x1<0:x1=0
    if x2>=max_x:x2=max_x
    dis=x2-x1
    img=state_img[:,x1*cell_size:(x2+1)*cell_size]
    start=view_distance-dis//2
    rt[:,start*cell_size:(start+dis+1)*cell_size,:]=img
    return rt

def merge_demo():
    fname=path.abspath(path.dirname(__file__) + '/../../outputs/state.jpg')
    img= Image.open(fname)#.resize((600,800),Image.Resampling.BILINEAR)
    data=np.asarray(img)
    img = get_state(data)
    img=Image.fromarray(img,'RGB')
    img.save('outputs/state2.jpg')

def split_demo():
    fname=path.abspath(path.dirname(__file__) + '/../../outputs/state2.jpg')
    img= Image.open(fname)#.resize((600,800),Image.Resampling.BILINEAR)
    data=np.asarray(img)
    img=get_observation(data,1,1)
    img=Image.fromarray(img,'RGB')
    img.save('outputs/observation_1_1.jpg')
    img=get_observation(data,1,2)
    img=Image.fromarray(img,'RGB')
    img.save('outputs/observation_1_2.jpg')
    img=get_observation(data,3,2)
    img=Image.fromarray(img,'RGB')
    img.save('outputs/observation_3_2.jpg')

if __name__ == "__main__":
    split_demo()
    #merge_demo()

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