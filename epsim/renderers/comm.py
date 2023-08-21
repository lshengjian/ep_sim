#import gymnasium
import pygame,os
from pygame import gfxdraw,Rect
import numpy as np
from epsim.core.componets import *
from epsim.core.consts import *
RENDER_DIR=os.path.abspath(os.path.dirname(__file__) )

def get_font(font_size=16,shadow=False):
    if shadow:
        return pygame.font.Font(RENDER_DIR+'/fonts/WHARMBY.TTF', font_size)
    return pygame.font.SysFont('arial', font_size)
def get_row_clomun(offset:float):
    '''
    0 | 1| 2| 3
    7 | 6| 5| 4
    8 | 9|10|11
    ......
    '''
    r=round(offset)//COLS
    c=round(offset)%COLS
    if r%2==1:
        c=COLS-1-c
    return r,c

def get_pos(offset:float):
    r,c=get_row_clomun(offset)
    #return (c+0.2)*CELL_SIZE,(r+1)*CELL_SIZE
    return c*CELL_SIZE,(r+1)*CELL_SIZE #留一行显示待处理物料

def show_text(screen,msg,x,y,fsize=16,shadow=False,color=(255, 221, 85)):
    font=get_font(fsize,shadow)
    textSerface = font.render(msg, True,color )#,(0,0,0)
    #pygame.image.save(text_serface, "text.png")
    # 绘制前先获取文本的宽高
    width, height = font.size(msg)
    # 绘制到显示器的surface上
    screen.blit(textSerface, (x, y))
    return width, height

def get_job_op_color(game,job):
    info=game.job_mgr.get_op_info(job.proc_code,job.op_index)
    return SLOT_COLORS[info.code]