from .comm import *
from .slots import *
from .cranes import *
from epsim.core import EVENT_CLICKED
from pygame import gfxdraw
from epsim.core.consts import SHOW_JOB_LIST
import esper

class Renderer(object):
    def __init__(self,game):
        self.world=game.world
        self.game=game
        
        self.screen_width, self.screen_height=(COLS+INFO_COLS)*CELL_SIZE,CELL_SIZE*ROWS
        self.info_left=(COLS+0.5)*CELL_SIZE
        self.screen=None
        self.closed=False
        self.clock=None
        self.render_on=False

        self.selected_proc_index=0
        self.selected_crane_id=0
        #self.recorder=recorder
        
        esper.set_handler(EVENT_CLICKED,self.on_click)


    def on_click(self,pos):
        for c_id, crane in self.world.get_component(CraneData):
            x,y=get_pos(crane.offset)
            y+=(1-crane.height)*CELL_SIZE
            rect=Rect(round(x),round(y),CELL_SIZE,CELL_SIZE*0.2)
            if rect.left<=pos[0]<=rect.right and rect.top<=pos[1]<=rect.bottom:
                if self.selected_crane_id==c_id:
                    self.selected_crane_id=0
                else:
                    self.selected_crane_id=c_id

                #print(crane)
                return
        for _, slot in self.world.get_component(SlotData): 
            left,top=get_pos(slot.offset)
            rect=Rect(left,top,CELL_SIZE,CELL_SIZE)
            if rect.left<=pos[0]<=rect.right and rect.top<=pos[1]<=rect.bottom:
                if self.selected_crane_id:
                    crane=self.world.component_for_entity(self.selected_crane_id,CraneData)
                    df=slot.offset-crane.offset
                    esper.dispatch_event('crane_move',(crane.code,df))
                    #print(f'{crane.code}-->{df}')
                return
        # PROCS=self.game.PROCS
        # info=self.game.job_mgr.stat
        # for i,proc_code in enumerate(PROCS.keys()):
        #     #job=jobMgr.get_job(j_id)
        #     nb_jobs=0 if proc_code not in info else info[proc_code]
        #     x,y=round(CELL_SIZE*(1.5+i)),CELL_SIZE//2
        #     dx,dy=abs(x-pos[0]),abs(y-pos[1])
        #     if nb_jobs>0 and dx*dx+dy*dy<=100:
        #         # if self.selected_proc_index==i:
        #         #     self.selected_proc_index=-1
        #         # else:
        #         self.selected_proc_index=i
        #         esper.dispatch_event('select_proc_index',self.selected_proc_index)
        #         return


    def draw(self):
        self.screen.fill((0, 0, 0))
        PROCS=self.game.PROCS
        jm =self.game.job_mgr
        info=jm.stat
        for i,proc_code in enumerate(PROCS.keys()):
            nb_jobs=0 if proc_code not in info else info[proc_code]
            x,y=round(CELL_SIZE*(0.5+i)),CELL_SIZE//2
            if jm.cur_proc_key==proc_code:
                gfxdraw.circle(self.screen,x,y,16,(255,255,255))
            if nb_jobs>0:
                gfxdraw.filled_circle(self.screen,x,y,12,PROC_COLORS[proc_code])
            else:
                gfxdraw.circle(self.screen,x,y,12,PROC_COLORS[proc_code])

        draw_slots(self.game,self.screen)
        draw_cranes(self.game,self.screen,self.selected_crane_id)

        

    def draw_info(self):
        steps=self.game.slot_mgr.nb_steps
        score=self.game.total_reward
        show_text(self.screen,f'SCORE:{score:>04.1f} TIME:{steps}',self.info_left,20,36,True,(155,34,237))
        self.debug_cranes()
        if SHOW_JOB_LIST:
            self.debug_jobs()
        else:
            self.debug_slots()
        
    def debug_cranes(self):
        for crane_id, crane in self.world.get_component(CraneData):
            cs=self.game.world.components_for_entity(crane_id)
            info=''
            for c in cs:
                info+=str(c)+'|'
            left=self.info_left
            show_text(self.screen,info,left,50+int(crane.code[1:])*20,16,False,color=(123,234,214))

    def debug_slots(self):
        top=50+len(self.game.crane_mgr.CRANES)*20+30
        
        ws=[]
        for s_id, s in self.world.get_component(SlotData):
            ws.append((s.offset,s_id))
        ws=sorted(ws,key=lambda x: x[0])
        k=0
        for _,s_id in ws:
            cs=self.game.world.components_for_entity(s_id)
            info=''
            for c in cs:
                info+=str(c)+'|'
            left=self.info_left
            show_text(self.screen,info,left,top+k*20,16,False,color=(223,134,114))
            k+=1

    def debug_jobs(self):
        data=self.game.job_mgr.data
        left=self.info_left#(GRID_SIZE[1]+1)*CELL_W
        top=70+len(self.game.crane_mgr.CRANES)*20
        for i in range(len(data)):
            msg=''
            for j in range(len(data[i])):
                job=self.game.job_mgr.get_job(self.game.job_mgr.ids[i])
                info=self.game.job_mgr.get_op_info(job.proc_code,j)
                msg+=f'{info.code}:{data[i][j][1]}/{data[i][j][0]} '
            
            show_text(self.screen,msg,left,top +i*20,16,False,(255,255,255))

    def render(self,render_mode):
        if render_mode is None:
            print("You are calling render method without specifying any render mode.")
            return

        if not self.render_on:
            # sets self.renderOn to true and initializes display
            self.enable_render(render_mode)

        self.draw()
        self.draw_info()
        

        observation = np.array(pygame.surfarray.pixels3d(self.screen))
        if render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
        return (
            np.transpose(observation, axes=(1, 0, 2))
            if render_mode != "ansi"
            else None
        )


    
    def close(self):
        if not self.closed:
            self.closed = True
            if self.render_on:
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
                self.render_on = False
                pygame.event.pump()
                pygame.display.quit()



    def enable_render(self,render_mode):
        pygame.init()
        pygame.display.init()
        pygame.font.init()
        #pygame.key.set_repeat(1, 1)
        if self.clock is None:
            self.clock = pygame.time.Clock()
        if render_mode == "human":
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        else:
            self.screen = pygame.Surface((self.screen_width, self.screen_height))

        pygame.display.set_caption("Multi Agent Demo")   
        self.render_on = True

        



    
    

