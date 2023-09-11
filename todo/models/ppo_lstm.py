import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from epsim.envs.eplate.eplate import MyEnv
from epsim.utils import load_config
from epsim.core.consts import *
from epsim.utils import img2cnn
from linformer import Linformer

#Hyperparameters
learning_rate = 0.0005
gamma         = 0.98
lmbda         = 0.95
eps_clip      = 0.1
K_epoch       = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def make_data(data):
    img= data[:ROWS*CELL_SIZE,:CELL_SIZE*COLS,:]
    return img2cnn(img,224)
#device = "cpu"
class PPO(nn.Module):
    def __init__(self,action_n=3):
        super().__init__()
        self.data = []
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(inplace=True),
            nn.Flatten()
        )
        with torch.no_grad():
            output_dim = np.prod(self.net(torch.zeros(1, 3, 224, 224)).shape[1:])
        # self.net = Linformer(dim=7, seq_len=20, depth=16, heads=7, k=64)
        # self.net = nn.Sequential(
        #     self.net, nn.Flatten(),
        #     nn.Linear(7*20, 128),
        #     nn.ReLU(inplace=True))
      
        self.fc1   = nn.Linear(output_dim,64)
        self.lstm  = nn.LSTM(64,64)
        self.fc_pi = nn.Linear(64,action_n)
        self.fc_v  = nn.Linear(64,1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x, hidden,mask=None):
        x=x.to(device)
        x= self.net(x)
        x = F.relu(self.fc1(x))
        x = x.view(-1, 1, 64)
        x, lstm_hidden = self.lstm(x, hidden)
        x = self.fc_pi(x)
        
       
        # if mask is not None:
        #     flag=torch.tensor(np.array([[mask]],dtype=np.int))==0
        #     #x[flag]=-float('inf')
        #     x.masked_fill(flag.to(device),torch.tensor(-float('inf')).to(device)) 
        prob = F.softmax(x, dim=-1)
        return prob, lstm_hidden
    
    def v(self, x, hidden):
        x=x.to(device)
        x= self.net(x)
        x = F.relu(self.fc1(x))
        x = x.view(-1, 1, 64)
        x, _ = self.lstm(x, hidden)
        v = self.fc_v(x)
        return v
      
    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, h_in_lst, h_out_lst, done_lst = [], [], [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, h_in, h_out, done = transition
            
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            h_in_lst.append(h_in)
            h_out_lst.append(h_out)
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
            
        s,a,r,s_prime,done_mask,prob_a = torch.tensor(np.array(s_lst), dtype=torch.float), torch.tensor(np.array(a_lst, dtype=np.int64)), \
                                         torch.tensor(np.array(r_lst), dtype=torch.float), torch.tensor(np.array(s_prime_lst), dtype=torch.float), \
                                         torch.tensor(np.array(done_lst), dtype=torch.float), torch.tensor(np.array(prob_a_lst), dtype=torch.float)
        self.data = []
        return s,a,r,s_prime, done_mask, prob_a, h_in_lst[0], h_out_lst[0]
        
    def train_net(self):
        s,a,r,s_prime,done_mask, prob_a, (h1_in, h2_in), (h1_out, h2_out) = self.make_batch()
        first_hidden  = (h1_in.detach().to(device), h2_in.detach().to(device))
        second_hidden = (h1_out.detach().to(device), h2_out.detach().to(device))

        for i in range(K_epoch):
            v_prime = self.v(s_prime, second_hidden).squeeze(1)
            td_target = r.to(device) + gamma * v_prime * done_mask.to(device)
            v_s = self.v(s.to(device), first_hidden).squeeze(1)
            delta = td_target - v_s
            delta = delta.detach().cpu().numpy()
            
            advantage_lst = []
            advantage = 0.0
            for item in delta[::-1]:
                advantage = gamma * lmbda * advantage + item[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float).to(device)

            pi, _ = self.pi(s, first_hidden)
            pi_a = pi.squeeze(1).gather(1,a.to(device))
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a.to(device)))  # a/b == log(exp(a)-exp(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(v_s, td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward(retain_graph=True)
            self.optimizer.step()
