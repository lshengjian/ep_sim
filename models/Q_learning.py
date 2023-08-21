import numpy as np 
Args = namedtuple('Args', ['env', 'Q','alpha','gamma','epsilon','n_episodes','verbose','record_training'])
class QLearning(object):
	
	def __init__(self, args):

		self.env = args[0]
		self.Q = args[1]
		self.alpha = args[2]
		self.gamma = args[3]
		self.epsilon = args[4]
		self.n_episodes = args[5]
		self.verbose = args[6]
		self.record_training = args[7]
		self.checkpoint = self.n_episodes * 0.1
		self.eps_decay = None

			

	def eps_greedy(self, obs):
		if np.random.uniform() < self.epsilon:
			return np.random.randint(self.env.action_space.n)
		else:
			action_values = [self.Q[obs, a] for a in 
							 range(self.env.action_space.n)]
			greedy_idx = np.argwhere(action_values == np.max(action_values))
			greedy_act_idx = np.random.choice(greedy_idx.flatten())
			return greedy_act_idx


	def opt_action_val(self, obs):
		action_values = [self.Q[obs, a] for a in 
						 range(self.env.action_space.n)]
		return np.max(action_values)


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
			obs ,info= self.env.reset()
			if self.record_training:
				episode_reward = 0
			
			while not done:
				a = self.eps_greedy(obs)
				obs_prime, reward, done, _,info = self.env.step(a)
				self.Q[obs,a] += self.alpha * (reward + self.gamma *
									   self.opt_action_val(obs_prime) - 
									   self.Q[obs,a])
				obs = obs_prime
				if self.record_training:
					episode_reward += reward
			if self.record_training:
				self.all_rewards.append(episode_reward)
			if not self.eps_decay is None:
				self.epsilon = self.eps_decay(self.epsilon)
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







