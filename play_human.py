from epsim.envs.eplate.eplate import MyEnv
from epsim.envs.eplate.manual_select_job import ManualJobPolicy
from epsim.utils import load_config
from datasets import Dataset 
DEMO_FILE='demo-3'

def one_demo(env,policy):
   print('demo.....')
   obss,acts,rs,ds=[],[],[],[]
   obs ,info= env.reset()
   
   cnt=0
   while cnt<900:
      info['step']=cnt
      act=policy(obs,info)
      obs, reward, terminated, truncated, info = env.step(act)
      obss.append(obs)
      acts.append(act)
      rs.append(reward)
      ds.append(terminated or truncated)
      if terminated or truncated:
         break
      cnt+=1
   return obss,acts,rs,ds
def demo(env,policy,cnt=1):
   obss,acts,rs,ds=[],[],[],[]
   for _ in range(cnt):
      o,a,r,d=one_demo(env,policy)
      obss.append(o)
      acts.append(a)
      rs.append(r)
      ds.append(d)
   my_dict = {"observations": obss,"actions":acts,"rewards":rs,"dones":ds} 
   dataset = Dataset.from_dict(my_dict)
   dataset.save_to_disk(f'data/{DEMO_FILE}')
   print(len(dataset))
   #print(dataset[0]["rewards"])
   
if __name__ == "__main__":
   args=load_config(f'{DEMO_FILE}.yaml')
   env = MyEnv(render_mode="human",args=args)
   policy = ManualJobPolicy(env)
   demo(env,policy)
   
   env.close()