import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import gym
from gym import spaces
import torch.nn.functional as F

class PPOMemory:
    def __init__(self, batch_size, device):
        self.states = None
        self.probs = None
        self.actions = None
        self.vals = None
        self.rewards = None
        self.dones = None
        self.batch_size = batch_size

        self.clear_memory()
        self.device = device

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = torch.arange(0, n_states, self.batch_size)
        indices = torch.arange(n_states, dtype=torch.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i + self.batch_size] for i in batch_start]

        return self.states, self.actions, self.probs, self.vals, self.rewards, self.dones, batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(torch.tensor(state, dtype=torch.float).unsqueeze(0))
        self.actions.append(torch.tensor(action, dtype=torch.float).unsqueeze(0))
        self.probs.append(torch.tensor(probs, dtype=torch.float).unsqueeze(0) if probs is not None else torch.zeros((1, len(action)), dtype=torch.float))
        self.vals.append(torch.tensor(vals, dtype=torch.float).unsqueeze(0))
        self.rewards.append(torch.tensor(reward, dtype=torch.float).unsqueeze(0))
        self.dones.append(torch.tensor(done, dtype=torch.bool).unsqueeze(0))

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.vals = []
        self.rewards = []
        self.dones = []

    def stack_tensors(self):
        self.states = torch.cat(self.states, dim=0).to(self.device)
        self.actions = torch.cat(self.actions, dim=0).to(self.device)
        self.probs = torch.cat(self.probs, dim=0).to(self.device)
        self.vals = torch.cat(self.vals, dim=0).to(self.device)
        self.rewards = torch.cat(self.rewards, dim=0).to(self.device)
        self.dones = torch.cat(self.dones, dim=0).to(self.device)

class EconomicEnv(gym.Env):
    def __init__(self):
        super(EconomicEnv, self).__init__()
        self.action_space = spaces.Box(low=np.array([0.0]), high=np.array([1.0]), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([0, 0]), high=np.array([1, 1]), dtype=np.float32)
        self.current_step = 0

        self.production = 0.5  # Fixed production value
        self.price = 0
        self.demand = 0

    def calculate_demand(self, price):
        return max(2 - 1.5 * price, 0)

    def calculate_cost(self, production):
        return 0.2 + 0.01 * production if production > 0 else 0

    def step(self, action):
        if len(action) != 1:
            raise ValueError("Action must contain exactly one element.")

        price_ratio = action[0]
        self.price = price_ratio
        self.demand = self.calculate_demand(self.price)

        cost = self.calculate_cost(self.production)
        revenue = min(self.production, self.demand) * self.price

        profit = revenue - cost
        reward = profit * 1000

        state = np.array([self.price, self.demand])
        terminated = self.current_step >= 63

        self.current_step += 1

        return state, reward, terminated, {'profit': profit, 'revenue': revenue, 'cost': cost}

    def reset(self, **kwargs):
        self.price = np.random.uniform(0, 1)
        self.demand = self.calculate_demand(self.price)
        self.current_step = 0
        state = np.array([self.price, self.demand])
        return state, {}

class ActorNetwork(nn.Module):
    def __init__(self, input_dims):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dims, 4)
        self.mean = nn.Linear(4, 1)
        self.log_std = nn.Parameter(torch.zeros(1))  # Log standard deviation as a learnable parameter
        self.relu = nn.ReLU()

    def forward(self, state):
        x = self.relu(self.fc1(state))
        mean = torch.sigmoid(self.mean(x))
        log_std = self.log_std.expand_as(mean)  # Ensure log_std has the same shape as mean
        std = torch.exp(log_std)
        return mean, std

    def act(self, state, uniform_sampling=False):
        if uniform_sampling:
            action = torch.rand(state.shape[0], 1).to(state.device)
            log_prob = torch.zeros(state.shape[0], 1).to(state.device)  # Uniform distribution log-prob is zero
        else:
            mean, std = self.forward(state)
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample().clamp(0, 1)
            log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

class CriticNetwork(nn.Module):
    def __init__(self, input_dims):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dims, 4)
        self.value = nn.Linear(4, 1)
        self.relu = nn.ReLU()

    def forward(self, state):
        x = self.relu(self.fc1(state))
        value = self.value(x)
        return value

class PPO:
    def __init__(self, input_dims, alpha=0.0003, policy_clip=0.2, batch_size=64, n_epochs=10, entropy_coefficient=0.01):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.actor = ActorNetwork(input_dims).to(self.device)
        self.critic = CriticNetwork(input_dims).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=alpha)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=alpha)

        self.memory = PPOMemory(batch_size, self.device)

        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.entropy_coefficient = entropy_coefficient
        self.uniform_sampling = True  # Start with uniform sampling

    def store_transition(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def learn(self):
        print('\n', "-" * 100)
        self.actor.train()
        self.critic.train()

        self.memory.stack_tensors()

        state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, batches = self.memory.generate_batches()

        advantages = reward_arr - vals_arr

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

        for _ in range(self.n_epochs):
            for batch in batches:
                batch_states = state_arr[batch]
                batch_actions = action_arr[batch]
                batch_advantages = advantages[batch]
                batch_returns = reward_arr[batch]
                batch_old_probs = old_prob_arr[batch]

                mean, std = self.actor(batch_states)
                dist = torch.distributions.Normal(mean, std)
                new_probs = dist.log_prob(batch_actions).sum(dim=-1)
                dist_entropy = dist.entropy().sum(dim=-1).mean()

                new_vals = self.critic(batch_states).squeeze()

                ratio = (new_probs.exp() / (batch_old_probs.exp() + 1e-10))

                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.policy_clip, 1 + self.policy_clip) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coefficient * dist_entropy

                critic_loss = 0.5 * (new_vals - batch_returns).pow(2).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

        self.memory.clear_memory()

    @torch.no_grad()
    def choose_action(self, observation):
        state = torch.tensor(observation, dtype=torch.float).to(self.device)
        action, log_prob = self.actor.act(state, self.uniform_sampling)
        value = self.critic(state).cpu().numpy().flatten()
        return action.cpu().numpy().flatten(), log_prob.cpu().numpy().flatten(), value

    def update_uniform_sampling(self, episode, switch_episode=2):
        if episode > switch_episode:
            self.uniform_sampling = False

if __name__ == '__main__':
    input_dims = 2
    alpha = 0.00005
    policy_clip = 0.2
    batch_size = 64
    n_epochs = 1000
    entropy_coefficient = 0.1

    agent = PPO(input_dims, alpha, policy_clip=policy_clip, batch_size=batch_size, n_epochs=n_epochs, entropy_coefficient=entropy_coefficient)
    env = EconomicEnv()

    num_episodes = 11

    for episode in tqdm(range(num_episodes)):
        agent.update_uniform_sampling(episode, switch_episode=2)
        observation, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action, log_prob, value = agent.choose_action(observation)
            next_observation, reward, terminated, info = env.step([action])

            agent.store_transition(observation, action, log_prob, value, reward, done)

            observation = next_observation
            done = terminated

            if len(agent.memory.states) >= agent.memory.batch_size:
                agent.learn()

            print(f'Episode {episode}, Actions: {action}, Reward: {reward}')
