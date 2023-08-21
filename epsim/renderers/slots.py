from pygame import gfxdraw,Rect
from .comm import *
from ..core.componets import Job,Slot,WithJob,Wait
from ..core.consts import *

kScale=0.618
def draw_slots(game,screen):
    row=0
    world=game.world
    for ent, s in world.get_component(Slot): 
        #self.debug(ent, row)
        row+=1
        left,top=get_pos(s.offset)
        rect=Rect(left,top,CELL_SIZE,CELL_SIZE)
        gfxdraw.rectangle(screen,
                rect,
                SLOT_COLORS[s.op_code],
        )

        if len(s.op_code)>1:
            r2=rect.copy()
            r2.y+=CELL_SIZE*kScale
            r2.h=CELL_SIZE*(1-kScale)
            if wt:=world.try_component(ent,Wait):
                k=(wt.duration-wt.timer)/wt.duration
                r2.w*=k
                if r2.w<1:r2.w=1
            gfxdraw.box( screen,
                    r2,
                    SLOT_COLORS[s.op_code],
            )
        elif s.op_code in {'T'}:
            r2=rect.copy()
            r2.x+=CELL_SIZE*0.46
            r2.w=10
            gfxdraw.box( screen,
                    r2,
                    SLOT_COLORS[s.op_code],
            )
        if 'E'==s.op_code:
            r3=rect.copy()
            r3.y+=CELL_SIZE*0.46
            r3.h=10
            gfxdraw.box( screen,
                    r3,
                    SLOT_COLORS[s.op_code],
            )

        if wj:=world.try_component(ent,WithJob):
            job=world.component_for_entity(wj.job_id,Job)
            job.op_index+=1#提示下一加工是什么
            gfxdraw.filled_circle(screen,int(left+5),int(top+CELL_SIZE//3+5),5,PROC_COLORS[job.proc_code])
            color=get_job_op_color(game,job)
            job.op_index-=1
            show_text(screen,str(job),left+20,top+CELL_SIZE//3,CELL_SIZE//5,False,color) 

