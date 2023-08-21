import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from PIL import Image

from linformer import Linformer
from epsim.utils import img2cnn
from epsim.core.consts import *
#Hyperparameters
learning_rate = 2.5e-4
gamma = 0.98
lmbda = 0.95
eps_clip = 0.1
K_epoch = 8
T_horizon = 800
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_data(data):
    img = data[:ROWS * CELL_SIZE, :CELL_SIZE * COLS, :]
    return img2cnn(img, 224)


class PPO(nn.Module):

    def __init__(self):
        super().__init__()
        self.data = []
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(6*10, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True))
        self.fc_pi = nn.Linear(128, 3)
        self.fc_v = nn.Linear(128, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x, softmax_dim=0):
        x = x.to(device)
        x =self.net(x)
        x = self.fc_pi(x)
        #print(x.cpu().numpy())
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    def v(self, x):
        x = x.to(device)
        x =self.net(x)
        v = self.fc_v(x)
        return v

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition

            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])

        s = torch.tensor(np.array(s_lst), dtype=torch.float)
        s_prime = torch.tensor(np.array(s_prime_lst), dtype=torch.float)
        a = torch.tensor(np.array(a_lst), dtype=torch.int64)
        r = torch.tensor(np.array(r_lst), dtype=torch.float)
        done_mask = torch.tensor(np.array(done_lst), dtype=torch.float)
        prob_a = torch.tensor(np.array(prob_a_lst), dtype=torch.float)
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a

    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()

        for i in range(K_epoch):
            td_target = r.to(
                device) + gamma * self.v(s_prime) * done_mask.to(device)
            delta = td_target - self.v(s)
            delta = delta.detach().cpu().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = gamma * lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst,
                                     dtype=torch.float).to(device)

            pi = self.pi(s, softmax_dim=1)
            pi_a = pi.gather(1, a.to(device))
            ratio = torch.exp(
                torch.log(pi_a) -
                torch.log(prob_a.to(device)))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(
                self.v(s), td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
