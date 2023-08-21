from epsim.utils import load_config

cfg=load_config('comm.yaml')
SHOW_JOB_LIST=cfg['SHOW_JOB_LIST']

FPS=cfg['FPS']
EPS=cfg['EPS']
MAX_JOBS=cfg['MAX_JOBS']
FIRST_GROUP_MAX_JOBS=cfg['FIRST_GROUP_MAX_JOBS']
MIN_WAIT_TIME=cfg['MIN_WAIT_TIME']
MAX_LOCK_TIME=cfg['MAX_LOCK_TIME']
CELL_SIZE=cfg['CELL_SIZE'] #每个格子的边长，单位为像素
ROWS=cfg['ROWS']
COLS=cfg['COLS']
INFO_COLS=cfg['INFO_COLS']
NOTIFY_BEFORE_FINISH=cfg['NOTIFY_BEFORE_FINISH'] #物料加工处理结束前提前几秒请求天车调度
SAFE_CRANE_DISTANCE=cfg['SAFE_CRANE_DISTANCE'] #安全距离:n格子
H_TOP=cfg['H_TOP']  #天车上升最高1单元格
H_LOW=cfg['H_LOW']
SLOT_COLORS=cfg['SLOT_COLORS']
PROC_COLORS=cfg['PROC_COLORS']

