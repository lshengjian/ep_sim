from collections import namedtuple

job_state=namedtuple('JobState',  \
    ['job_num','proc_num','op_num','plan_op_time','op_time','offset'])


op_info=namedtuple('OpInfo',  ['code', 'op_time', 'out_wait'])
                             # 工艺代码，处理时间，滴液等待时间

crane_info=namedtuple('CraneInfo',  ['group','code','speed','speed_up_down','min_offset','max_offset','offset','stop_wait'])
slot_info=namedtuple('SlotInfo',  ['group','code', 'offset'])

EVENT_GAME_OVER='game_over'
EVENT_TIME_OUT='time_out'

EVENT_OP_START='op_start'
EVENT_OP_DOING='op_doing'
EVENT_OP_FINISHED='op_finished'
EVENT_OP_STOPED='op_stoped'
EVENT_JOB_FINISHED='job_finished'

EVENT_CLICKED='click_screen'



