from __future__ import annotations

import math
import numpy as np


def set_color(img,r,g,b):
    img2=img.astype(np.float32)
    img2/=255.0
    img2[:,:]*=np.array([r/255.0,g/255.0,b/255.0],dtype=np.float32)
    img2*=255.0
    return img2.astype(np.uint8)


def blend_imgs(img_src:np.ndarray,img_dest:np.ndarray, start=(0,0)):
    """
    合并两张图片
    """
    left,top=start
    left=int(left)
    top=int(top)
    h1,w1,_=img_src.shape
    h2,w2,_=img_dest.shape

    rt=np.zeros_like(img_dest)
    rt[:,:,:]=img_dest

    for i in range(h2):
        for j in range(w2):
            if i>=top and j>=left and i<h1+top and j<w1+left:
                r,c=i-top,j-left
                #print(img_src.shape)
                if (img_src[r, c]).any()>0:
                    rt[i, j, :]=img_src[r, c]
    return rt
            
def highlight_img(img, color=(255, 255, 255), alpha=0.30):
    """
    Add highlighting to an image
    """

    blend_img = img + alpha * (np.array(color, dtype=np.uint8) - img)
    blend_img = blend_img.clip(0, 255).astype(np.uint8)
    img[:, :, :] = blend_img


def fill_coords(img, fn):
    """
    Fill pixels of an image with coordinates matching a filter function
    """

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            yf = (y + 0.5) / img.shape[0]
            xf = (x + 0.5) / img.shape[1]
            if fn(xf, yf):
                img[y, x] = (255,255,255)

    return img


def rotate_fn(fin, cx, cy, theta):
    def fout(x, y):
        x = x - cx
        y = y - cy

        x2 = cx + x * math.cos(-theta) - y * math.sin(-theta)
        y2 = cy + y * math.cos(-theta) + x * math.sin(-theta)

        return fin(x2, y2)

    return fout


def point_in_line(x0, y0, x1, y1, r):
    p0 = np.array([x0, y0], dtype=np.float32)
    p1 = np.array([x1, y1], dtype=np.float32)
    dir = p1 - p0
    dist = np.linalg.norm(dir)
    dir = dir / dist

    xmin = min(x0, x1) - r
    xmax = max(x0, x1) + r
    ymin = min(y0, y1) - r
    ymax = max(y0, y1) + r

    def fn(x, y):
        # Fast, early escape test
        if x < xmin or x > xmax or y < ymin or y > ymax:
            return False

        q = np.array([x, y])
        pq = q - p0

        # Closest point on line
        a = np.dot(pq, dir)
        a = np.clip(a, 0, dist)
        p = p0 + a * dir

        dist_to_line = np.linalg.norm(q - p)
        return dist_to_line <= r

    return fn


def point_in_circle(cx, cy, r):
    def fn(x, y):
        return (x - cx) * (x - cx) + (y - cy) * (y - cy) <= r * r

    return fn

def point_in_polygon(cx, cy, radius, num_sides):
    # 计算正多边形的半径
    #radius = side_length / (2 * math.sin(math.pi / num_sides))
    
    # 计算正多边形的顶点坐标
    K=2 * math.pi / num_sides
    vertices = []
    for i in range(num_sides):
        angle =  i * K #+math.pi / num_sides 
        vx = cx + radius * math.cos(angle)
        vy = cy + radius * math.sin(angle)
        vertices.append((vx, vy))
    
    # 判断点是否在正多边形内
    def fn(x, y):
        inside = False
        for i in range(num_sides):
            j = (i + 1) % num_sides
            if (vertices[i][1] > y) != (vertices[j][1] > y) and \
            x < (vertices[j][0] - vertices[i][0]) * (y - vertices[i][1]) / (vertices[j][1] - vertices[i][1]) + vertices[i][0]:
                inside = not inside
        return inside
    return fn

def point_in_rect(xmin, xmax, ymin, ymax):
    def fn(x, y):
        return x >= xmin and x <= xmax and y >= ymin and y <= ymax

    return fn


def point_in_triangle(a, b, c):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    c = np.array(c, dtype=np.float32)

    def fn(x, y):
        v0 = c - a
        v1 = b - a
        v2 = np.array((x, y)) - a

        # Compute dot products
        dot00 = np.dot(v0, v0)
        dot01 = np.dot(v0, v1)
        dot02 = np.dot(v0, v2)
        dot11 = np.dot(v1, v1)
        dot12 = np.dot(v1, v2)

        # Compute barycentric coordinates
        inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
        u = (dot11 * dot02 - dot01 * dot12) * inv_denom
        v = (dot00 * dot12 - dot01 * dot02) * inv_denom

        # Check if point is in triangle
        return (u >= 0) and (v >= 0) and (u + v) < 1

    return fn

'''
def downsample(img, factor):
    """
    Downsample an image along both dimensions by some factor
    """

    assert img.shape[0] % factor == 0
    assert img.shape[1] % factor == 0

    img = img.reshape(
        [img.shape[0] // factor, factor, img.shape[1] // factor, factor, 3]
    )
    img = img.mean(axis=3)
    img = img.mean(axis=1)

    return img
'''