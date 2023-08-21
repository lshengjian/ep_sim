from pygame import gfxdraw,Rect
from .comm import *
from ..core.componets import Job,WithJob
from ..core.consts import *

def draw_cranes(game,screen,secect_id):
    world=game.world
    for crane_id, crane in world.get_component(Crane):
        x,y=get_pos(crane.offset)
        y+=round((1.02-crane.height)*CELL_SIZE)
        rect=Rect(round(x),round(y),CELL_SIZE,CELL_SIZE*0.25)
        if secect_id==crane_id:
            r2=rect.copy()
            r2.x-=2
            r2.y-=2
            r2.w+=4
            r2.h+=4
            gfxdraw.box(screen,r2, (255, 0, 0))

        gfxdraw.box(screen,rect, (255, 255, 255))
        show_text(screen,crane.code,x+12,y,CELL_SIZE//5,False,(0,255,0))
        if wj:=world.try_component(crane_id,WithJob):
            job=world.component_for_entity(wj.job_id,Job)
            gfxdraw.filled_circle(screen,int(x+5),int(y+5),5,PROC_COLORS[job.proc_code])
            color=get_job_op_color(game,job)
            show_text(screen,str(job),x+35,y,CELL_SIZE//5,False,color)
            

            