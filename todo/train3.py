from models.ppo import PPO
from epsim.envs.myenv import MyEnv
from torch.distributions.categorical import Categorical
import hydra,torch
import numpy as np
@hydra.main(config_path="./config", config_name="args", version_base="1.3")
def main(cfg: "DictConfig"):  # noqa: F821
    env = MyEnv(render_mode='ansi',args=cfg)
    model = PPO()
    score = 0.0
    print_interval = 3
    T_horizon=20

    for n_epi in range(1000):
        print(n_epi)
        s,_ = env.reset()
        done = False
        cnt=0
        while cnt<10:
            for t in range(T_horizon):
                
                prob = model.pi(torch.from_numpy(np.array(s).ravel()).float())
                m = Categorical(prob)
                a = m.sample().item()
                s_prime, r, done,end, info = env.step(a)
                #print(f'{cnt}-{t}:{r}')
                done=done or end

                model.put_data((np.array(s).ravel(), a, r, np.array(s_prime).ravel(), prob[a].item(), done))
                s = s_prime

                score += r
                if done:
                    break
                cnt+=1

            
            model.train_net()

        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
            score = 0.0

    env.close()

if __name__ == '__main__':
    main()