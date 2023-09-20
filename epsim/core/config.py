from __future__ import annotations
from pathlib import Path

from typing import List,Dict
from epsim.core.constants import SHARE
from epsim.core.componets import *

__all__=['split_field','build_config','get_files','get_file_info']
import logging
import logging.config
import yaml

logger=None
#logging.config.fileConfig('config/logging.yaml')

'''
# LOG_DICT={
#     'debug':logging.DEBUG,
#     'info':logging.INFO,
#     'error':logging.ERROR,
# }
# logging.basicConfig(filename='outputs/epsim.log',format='%(name)s - %(levelname)s - %(message)s', encoding='utf-8', level=LOG_DICT[SHARE.LOG_LEVEL])

#,format='%(asctime)s - %(levelname)s - %(message)s', encoding='utf-8', level=logging.DEBUG)

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

def get_files(data_directory:str='test01')->List[str]:
    fs:List[str]=[]
    dir=Path(__file__).parent.joinpath(f'../../config')
    fs.append(str(dir)+'/operates.csv')
    dir=Path(f'{dir}/{data_directory}')
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



def _make_slots(data,op_name2key:Dict[str,int]):
    rt:List[SlotData]=[]
    for i,d in enumerate(data):
        ds=d.split(',')
        xs=tuple(split_field(ds[2]))
        rt.append(SlotData(id=i,group=int(ds[0]),op_key=op_name2key[ds[1]],offsets=xs))
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


def _make_procedures(data,op_name2key:[str,int]):
    
    rt:List[OpLimitData]=[]
    for i,d in enumerate(data):
        ds=d.split(',')
        rt.append(OpLimitData(id=i,product_code=ds[0],op_key=op_name2key[ds[1]], \
                            min_time=int(ds[2]), \
                            max_time=int(ds[3]) ) )
    return rt

def _make_one(fn:str,op_name2key=None):
    name,field_names,data=get_file_info(fn)
    rt=None

    logger.info(f'{name}-->{field_names}')
    if name.find('operates')>=0:
        rt = _make_ops(data)
    elif name.find('cranes')>=0:
        rt =  _make_cranes(data)
    elif name.find('slots')>=0:
        rt =  _make_slots(data,op_name2key)
    elif name.find('procedures')>=0:
        rt =  _make_procedures(data,op_name2key)
    return name,rt

def build_config(data_directory:str='test01')->Tuple:#->Tuple[|,Dict[str,List[Index]]
    global logger
    with open(file="config/logging.yaml", mode='r', encoding="utf-8") as file:
        logging_yaml = yaml.load(stream=file, Loader=yaml.FullLoader)
        logging.config.dictConfig(config=logging_yaml)
        logger = logging.getLogger(__name__.split('.')[-1])
    ds:Dict[str,List[Index]]={}
    fs=get_files(data_directory)
    op_name2key={}
    op_dict:Dict[int,OperateData]={}
    for f in fs:
        name,data=_make_one(f,op_name2key)
        ds[name]=data
        if name=='operates':
            for dop in data:
                d:OperateData=dop
                op_name2key[d.name]=d.key
                op_dict[d.key]=d
        
    for d in ds['slots']:
        slot:SlotData=d
        slot.op_name=op_dict[slot.op_key].name
    for d in ds['procedures']:
        slot:SlotData=d
        slot.op_name=op_dict[slot.op_key].name
    return op_dict,ds['slots'],ds['cranes'] ,ds['procedures'] 



# if __name__ == '__main__':
#     assert split_field('1.3')==[1]
#     assert split_field('1|3|5')==[1,3,5]
#     assert split_field('1~5')==[1,2,3,4,5]

#     op_dict,slots,cranes ,procedures=build_config()

#     print(len(op_dict))


    # df=ds['3-cranes']
    # assert(list(df[['name','offset']].iloc[0]))==['H1',1] 
    