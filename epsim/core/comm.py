from ast import literal_eval
from .componets import *
from .consts import EPS,SAFE_CRANE_DISTANCE
import numpy as np
import torch,random

# -----------------------------------------------------------------------------
DEBUG=False

__ALL__=['debug','set_seed','CfgNode','arrived_to','is_safe']

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)






def debug(msg):
    if DEBUG:
        print(msg)

def arrived_to(target:float,pos:float,next_pos:float)->bool:
    return abs(target-next_pos)<EPS or (target-pos)*(target-next_pos)<=0

def is_safe(crane:Crane,crane_to:float,agv:Crane,agv_to:float,safe_dis=SAFE_CRANE_DISTANCE):
    d1=crane_to-crane.offset
    crane_dir=0 if abs(d1)<1e-10 else d1/abs(d1)
    d2=agv_to-agv.offset
    agv_dir=0 if abs(d2)<1e-10 else d2/abs(d2)
    cnt=0
    crane_pos=crane.offset
    agv_pos=agv.offset
    while cnt<100:
        pos1=crane_pos
        pos2=agv_pos
        crane_pos+=crane_dir*crane.speed
        agv_pos+=agv_dir*agv.speed
        if crane_pos<crane.min_offset: crane_pos=crane.min_offset
        elif crane_pos>crane.max_offset: crane_pos=crane.max_offset
        if agv_pos<agv.min_offset: agv_pos=agv.min_offset
        elif agv_pos>agv.max_offset: agv_pos=agv.max_offset
        if arrived_to(crane_to,pos1,crane_pos):
            crane_pos=crane_to
        if arrived_to(agv_to,pos2,agv_pos):
            agv_pos=agv_to

        if abs(crane_pos-agv_pos)<safe_dis:
            return False
        cnt+=1
    return True

class CfgNode:
    """ a lightweight configuration class inspired by yacs """
    # TODO: convert to subclass from a dict like in yacs?
    # TODO: implement freezing to prevent shooting of own foot
    # TODO: additional existence/override checks when reading/writing params?

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __str__(self):
        return self._str_helper(0)

    def _str_helper(self, indent):
        """ need to have a helper to support nested indentation for pretty printing """
        parts = []
        for k, v in self.__dict__.items():
            if isinstance(v, CfgNode):
                parts.append("%s:\n" % k)
                parts.append(v._str_helper(indent + 1))
            else:
                parts.append("%s: %s\n" % (k, v))
        parts = [' ' * (indent * 4) + p for p in parts]
        return "".join(parts)

    def to_dict(self):
        """ return a dict representation of the config """
        return { k: v.to_dict() if isinstance(v, CfgNode) else v for k, v in self.__dict__.items() }

    def merge_from_dict(self, d):
        self.__dict__.update(d)

    def merge_from_args(self, args):
        """
        update the configuration from a list of strings that is expected
        to come from the command line, i.e. sys.argv[1:].

        The arguments are expected to be in the form of `--arg=value`, and
        the arg can use . to denote nested sub-attributes. Example:

        --model.n_layer=10 --trainer.batch_size=32
        """
        for arg in args:

            keyval = arg.split('=')
            assert len(keyval) == 2, "expecting each override arg to be of form --arg=value, got %s" % arg
            key, val = keyval # unpack

            # first translate val into a python object
            try:
                val = literal_eval(val)
                """
                need some explanation here.
                - if val is simply a string, literal_eval will throw a ValueError
                - if val represents a thing (like an 3, 3.14, [1,2,3], False, None, etc.) it will get created
                """
            except ValueError:
                pass

            # find the appropriate object to insert the attribute into
            assert key[:2] == '--'
            key = key[2:] # strip the '--'
            keys = key.split('.')
            obj = self
            for k in keys[:-1]:
                obj = getattr(obj, k)
            leaf_key = keys[-1]

            # ensure that this attribute exists
            assert hasattr(obj, leaf_key), f"{key} is not an attribute that exists in the config"

            # overwrite the attribute
            print("command line overwriting config attribute %s with %s" % (key, val))
            setattr(obj, leaf_key, val)
