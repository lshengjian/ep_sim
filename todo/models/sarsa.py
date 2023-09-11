from collections import namedtuple
import numpy as np 

Args = namedtuple('Args', ['env', 'Q','alpha','gamma','epsilon','n_episodes','verbose','record_training'])

class SARSA(object):
	def __init__(self, args):
		self.env = args.env
		self.Q = args.Q
		self.alpha = args.alpha
		self.gamma = args.gamma
		self.epsilon = args.epsilon
		self.n_episodes = args.n_episodes
		self.verbose = args.verbose
		self.record_training = args.record_training
		self.checkpoint = self.n_episodes * 0.1


	def eps_greedy(self, obs):
		if np.random.uniform() < self.epsilon:
			return np.random.randint(self.env.action_space.n)
		else:
			action_values = [self.Q[obs, a] for a in 
							 range(self.env.action_space.n)]
			greedy_idx = np.argwhere(action_values == np.max(action_values))
			greedy_act_idx = np.random.choice(greedy_idx.flatten())
			return greedy_act_idx

	def greedy_action(self, obs):
		action_values = [self.Q[obs, a] for a in 
						 range(self.env.action_space.n)]
		greedy_idx = np.argmax(action_values)
		return greedy_idx

	def train(self, idx=None, q=None):
		if self.record_training:
			self.all_rewards = []

		for episode in range(self.n_episodes):
			done = False
			tranc = False
			obs ,info= self.env.reset()
			if self.record_training:
				episode_reward = 0
			a = self.eps_greedy(obs)

			while not (done or tranc):
				obs_prime, reward, done, tranc,info = self.env.step(a)
				a_prime = self.eps_greedy(obs_prime)
				self.Q[obs,a] += self.alpha * (reward + self.Q[obs_prime, a_prime] -
												   self.Q[obs, a])
				if self.record_training:
					episode_reward += reward
				obs = obs_prime
				a = a_prime
				
				
			if self.record_training:
				self.all_rewards.append(episode_reward)
			if self.verbose and episode % self.checkpoint == 0:
				if not idx is None:
					print(f'Agent: {idx} Episode: {episode}')
				else:
					print(f'Episode: {episode}')
		if not q is None:
			q.put(self)
		if not idx is None:
			print(f'Agent: {idx} - Training complete.')
		else:
			print('Training complete.')
