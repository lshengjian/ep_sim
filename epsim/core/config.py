from __future__ import annotations
from pathlib import Path

from typing import List,Dict
from epsim.core.componets import *

__all__=['split_field','build_config','get_files','get_file_info']
import logging
logging.basicConfig(filename='epsim.log',format='%(name)s - %(levelname)s - %(message)s', encoding='utf-8', level=logging.ERROR)
#,format='%(asctime)s - %(levelname)s - %(message)s', encoding='utf-8', level=logging.DEBUG)
logger = logging.getLogger(__name__)
'''
import colorlog
handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
	'%(log_color)s%(levelname)s:%(name)s:%(message)s'))
logger = colorlog.getLogger(__name__)
logger.addHandler(handler)
'''
def split_field(data:str):
    sep=r'~'
    if data.find(sep)>0:
        ds=data.split(sep)
        return list(range( int(ds[0]),int(ds[-1])+1 ))
    sep=r'|'
    if data.find(sep)>0:
        ds=data.split(sep)
        return list(map(int,ds))
    return [int(float(data)+0.5)]

def get_files(config_directory:str='demo')->List[str]:
    fs:List[str]=[]
    dir=Path(__file__).parent.joinpath(f'../../config/{config_directory}')
    for file in dir.rglob('*.csv'): #[x for x in p.iterdir() if x.is_dir()]
        fs.append(str(file))
        # data = pd.read_csv(file)
        # ds[file.stem]=data
    return fs

def get_file_info(fn:str)->List[str]:
    fp=Path(fn)
    clean_func = lambda x : x.replace('\n','')
    #lines=f.read_text(encoding='utf8')
    with fp.open(encoding='utf8') as f: 
        lines=f.readlines()
        lines=list(map(clean_func,lines))
        lines=list(filter(lambda x :x[0]!='#',lines)) #去掉注释
        field_names=lines[0].split(',')
        data=lines[1:]
        
    return fp.stem,field_names,data

def _make_ops(data):
    rt:List[OperateData]=[]
    for i,d in enumerate(data):
        ds=d.split(',')
        cs=split_field(ds[2])
        color=Color(*cs)
        rt.append(OperateData(id=i,key=int(ds[0]),name=ds[1],color=color))
    return rt
    
def _make_slots(data):
    rt:List[SlotData]=[]
    for i,d in enumerate(data):
        ds=d.split(',')
        xs=tuple(split_field(ds[2]))
        rt.append(SlotData(id=i,group=int(ds[0]),op_key=int(ds[1]),offsets=xs))
    return rt

def _make_cranes(data):
    rt:List[CraneData]=[]
    for i,d in enumerate(data):
        ds=d.split(',')
        rt.append(CraneData(id=i,group=int(ds[0]),name=ds[1], \
                            offset=int(ds[2]), \
                            speed_x=float(ds[3]), \
                            speed_y=float(ds[4]) ) )
    return rt

# def prodct_operates(ps:List[ProcessData],product_code:str):
#     rt=list(filter(lambda x:x.product_code==product_code,ps))
#     return rt


def _make_procedures(data):
    
    rt:List[OpLimitData]=[]
    for i,d in enumerate(data):
        ds=d.split(',')
        rt.append(OpLimitData(id=i,product_code=ds[0],op_key=int(ds[1]), \
                            min_time=int(ds[2]), \
                            max_time=int(ds[3]) ) )
    return rt

def _make_one(fn:str):
    name,field_names,data=get_file_info(fn)
    rt=None

    logger.info(f'{name}-->{field_names}')
    if name.find('operates')>=0:
        rt = _make_ops(data)
    elif name.find('slots')>=0:
        rt =  _make_slots(data)
    elif name.find('cranes')>=0:
        rt =  _make_cranes(data)
    elif name.find('procedures')>=0:
        rt =  _make_procedures(data)
    return name,rt

def build_config(config_directory:str='demo')->Tuple:#->Tuple[|,Dict[str,List[Index]]
    ds:Dict[str,List[Index]]={}
    fs=get_files(config_directory)
    for f in fs:
        name,data=_make_one(f)
        ds[name]=data
    ops=ds['1-operates']
    op_dict:Dict[int,OperateData]={}
    for d in ops:
        op:OperateData=d
        op_dict[op.key]=op
    for d in ds['2-slots']:
        slot:SlotData=d
        slot.op_name=op_dict[slot.op_key].name
    for d in ds['4-procedures']:
        slot:SlotData=d
        slot.op_name=op_dict[slot.op_key].name
    return op_dict,ds['2-slots'],ds['3-cranes'] ,ds['4-procedures'] 



if __name__ == '__main__':
    assert split_field('1.3')==[1]
    assert split_field('1|3|5')==[1,3,5]
    assert split_field('1~5')==[1,2,3,4,5]

    # df=ds['3-cranes']
    # assert(list(df[['name','offset']].iloc[0]))==['H1',1] 
    