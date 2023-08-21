from pygame import gfxdraw
from ..comm import BaseRanderer,font2
from ...core.componets import Job,Idle
from ...core.consts import *

class HUD(BaseRanderer):
    def render(self,**kwargs):
        i=0
        self.show_text(f'TIME:{self.game.nb_steps}',8,8,font=font2)
        self.show_text('JOB:',160,8,font=font2)
        self.show_text(f'SCORE:{self.game.total_reward:.2f}',8,48,font=font2)
        
        for _, (job,_) in self.game.world.get_components(Job,Idle):
            #print(job)
            x,y=240+i*40,24
            gfxdraw.filled_circle(self.surf,x,y,4,PROC_COLORS[job.proc_code])
            i+=1
        
        
        jobs=self.game.jobs
        data=kwargs["job_data"]
        left=(GRID_SIZE[1]+1)*CELL_W
        for i in range(len(data)):
            for j in range(len(data[i])):
                info=self.game.get_op_info(jobs[i].proc_code,j)
                self.show_text(f'{info.code}:{data[i][j][1]}/{data[i][j][0]}',left+j*70,20+i*20,(255,255,255))