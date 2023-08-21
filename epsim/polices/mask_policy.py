import random
class MaskPolicy(object):
    '''
    根据状态进行决策
    '''
    def __init__(self,env):
        self.evn=env

    def __call__(self,obs,info):
        mask=info['action_mask']
        acts=[0]
        ps=[200] if info['step']>0 else [1]
        for i,m in enumerate(mask):
            if  m>0 and i>0:
                acts.append(i)
                if info['step']==0 and i==2:
                    ps.append(100)
                else:
                    ps.append(5 if i==2 else 1)
      
        #print(acts,ps)
        return random.choices(acts,weights=ps,k=1)[0]
