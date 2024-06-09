import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import gym
from gym import spaces
import time

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
        self.action_space = spaces.Box(low=np.array([0.0, 0.0, 0.0]), high=np.array([1.0, 1.0, 1.0]), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([0, 0, 0]), high=np.array([1, 1, 1]), dtype=np.float32)
        self.current_step = 0

        self.production = 0
        self.price = 0
        self.demand = 0

    def calculate_demand(self, price):
        return max(2 - 1.5 * price, 0)

    def calculate_cost(self, production):
        return 0.2 + 0.01 * production if production > 0 else 0

    def step(self, action):
        if len(action) != 3:
            raise ValueError("Action must contain exactly three elements.")

        production_ratio, price_ratio, start_production = action
        self.production = production_ratio if start_production > 0.5 else 0
        self.price = price_ratio
        self.demand = self.calculate_demand(self.price)

        cost = self.calculate_cost(self.production)
        revenue = min(self.production, self.demand) * self.price

        profit = revenue - cost
        reward = profit*1000

        state = np.array([self.production, self.price, self.demand])
        terminated = self.current_step >= 63

        self.current_step += 1

        return state, reward, terminated, {'profit': profit, 'revenue': revenue, 'cost': cost}

    def reset(self, **kwargs):
        self.production = np.random.uniform(0, 1)
        self.price = np.random.uniform(0, 1)
        self.demand = self.calculate_demand(self.price)
        self.current_step = 0
        state = np.array([self.production, self.price, self.demand])
        return state, {}

class BaseNetwork(nn.Module):
    def __init__(self, input_dims, output_dims, dropout_rate=0.1):
        super(BaseNetwork, self).__init__()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        layer_dims = [32, 32]
        for i in range(len(layer_dims)):
            layer = nn.Linear(input_dims if i == 0 else layer_dims[i-1], layer_dims[i])
            nn.init.kaiming_normal_(layer.weight)  # He initialization
            self.layers.append(layer)
            self.batch_norms.append(nn.BatchNorm1d(layer_dims[i]))

        self.final_layer = nn.Linear(layer_dims[-1], output_dims)
        nn.init.kaiming_normal_(self.final_layer.weight)  # He initialization
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, state):
        x = state
        for i, (layer, bn) in enumerate(zip(self.layers, self.batch_norms)):
            x = layer(x)
            if x.size(0) > 1:  # Apply batch normalization only if batch size > 1
                x = bn(x)
            x = self.relu(x)
            x = self.dropout(x)
            # Debugging print to check for NaNs
            if torch.isnan(x).any():
                print(f"NaN detected in layer {i}: {x}")
        return x

class CriticNetwork(BaseNetwork):
    def __init__(self, input_dims, dropout_rate=0.1):
        super(CriticNetwork, self).__init__(input_dims, 1, dropout_rate)

    def forward(self, state):
        x = super(CriticNetwork, self).forward(state)
        q = self.final_layer(x)
        return q

class ActorNetwork(nn.Module):
    def __init__(self, input_dims, dropout_rate=0.2):
        super(ActorNetwork, self).__init__()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        layer_dims = [32, 32]
        for i in range(len(layer_dims)):
            layer = nn.Linear(input_dims if i == 0 else layer_dims[i-1], layer_dims[i])
            nn.init.kaiming_normal_(layer.weight)  # He initialization
            self.layers.append(layer)
            self.batch_norms.append(nn.BatchNorm1d(layer_dims[i]))

        self.final_layer_continuous = nn.Linear(layer_dims[-1], 2)
        self.final_layer_binary = nn.Linear(layer_dims[-1], 1)
        nn.init.kaiming_normal_(self.final_layer_continuous.weight)  # He initialization
        nn.init.kaiming_normal_(self.final_layer_binary.weight)  # He initialization
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, state):
        x = state
        for i, (layer, bn) in enumerate(zip(self.layers, self.batch_norms)):
            x = layer(x)
            if x.size(0) > 1:  # Apply batch normalization only if batch size > 1
                x = bn(x)
            x = self.relu(x)
            x = self.dropout(x)
        continuous_actions = torch.sigmoid(self.final_layer_continuous(x))
        binary_action = torch.sigmoid(self.final_layer_binary(x))
        return continuous_actions, binary_action

    def act(self, state):
        continuous_actions, binary_action = self.forward(state)
        dist_continuous = torch.distributions.Beta(continuous_actions, 1 - continuous_actions)
        dist_binary = torch.distributions.Bernoulli(probs=binary_action)
        actions_continuous = dist_continuous.sample()
        actions_binary = dist_binary.sample()
        actions = torch.cat((actions_continuous, actions_binary), dim=-1)
        return actions, dist_continuous.log_prob(actions_continuous).sum(dim=-1) + dist_binary.log_prob(actions_binary).sum(dim=-1)

class PPO:
    def __init__(self, input_dims, n_actions, alpha=0.0003, policy_clip=0.2, batch_size=64, n_epochs=10, entropy_coefficient=0.01, weight_decay=0.01, mini_batch_size=64):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.actor = ActorNetwork(input_dims).to(self.device)
        self.critic = CriticNetwork(input_dims).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=alpha, weight_decay=weight_decay)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=alpha, weight_decay=weight_decay)

        self.memory = PPOMemory(batch_size, self.device)

        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.entropy_coefficient = entropy_coefficient
        self.mini_batch_size = mini_batch_size

    def store_transition(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def learn(self):
        start_time = time.time()
        print('\n', "-" * 100)
        self.actor.train()
        self.critic.train()

        self.memory.stack_tensors()

        state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, batches = self.memory.generate_batches()

        state_arr = state_arr.clone().detach().to(self.device)
        action_arr = action_arr.clone().detach().to(self.device)
        vals_arr = vals_arr.clone().detach().to(self.device)
        reward_arr = reward_arr.clone().detach().to(self.device)
        dones_arr = dones_arr.clone().detach().to(self.device)

        advantages = reward_arr - vals_arr

        for _ in range(self.n_epochs):
            num_samples = len(state_arr)
            indices = np.arange(num_samples)
            np.random.shuffle(indices)

            for start_idx in range(0, num_samples, self.mini_batch_size):
                minibatch_indices = indices[start_idx:start_idx + self.mini_batch_size]

                batch_states = state_arr[minibatch_indices].clone().detach().to(self.device)
                batch_actions = action_arr[minibatch_indices].clone().detach().to(self.device)
                batch_advantages = advantages[minibatch_indices].clone().detach().to(self.device)
                batch_returns = reward_arr[minibatch_indices].clone().detach().to(self.device)
                batch_old_probs = old_prob_arr[minibatch_indices].clone().detach().to(self.device)

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()

                new_probs, dist_entropy, actor_loss, critic_loss = self.calculate_loss(batch_states, batch_actions, batch_advantages, batch_returns, batch_old_probs)

                actor_loss.backward()
                self.actor_optimizer.step()
                critic_loss.backward()
                self.critic_optimizer.step()

        self.memory.clear_memory()
        end_time = time.time()
        episode_time = end_time - start_time
        print(f"Learning completed in {episode_time} seconds")
        print("-" * 100)

    def calculate_loss(self, batch_states, batch_actions, batch_advantages, batch_returns, batch_old_probs):
        continuous_actions, binary_action = self.actor(batch_states)
        dist_continuous = torch.distributions.Beta(continuous_actions, 1 - continuous_actions)
        dist_binary = torch.distributions.Bernoulli(probs=binary_action)

        new_probs_continuous = dist_continuous.log_prob(batch_actions[:, :2]).sum(dim=-1)
        new_probs_binary = dist_binary.log_prob(batch_actions[:, 2]).sum(dim=-1)
        new_probs = new_probs_continuous + new_probs_binary
        dist_entropy = dist_continuous.entropy().sum(dim=-1).mean() + dist_binary.entropy().mean()

        new_vals = self.critic(batch_states).squeeze()

        batch_old_probs = batch_old_probs.sum(dim=-1)

        ratio = (new_probs / (batch_old_probs + 1e-10))

        batch_advantages = batch_advantages.unsqueeze(1) if batch_advantages.ndim == 1 else batch_advantages

        surr1 = ratio * batch_advantages
        surr2 = torch.clamp(ratio, 1 - self.policy_clip, 1 + self.policy_clip) * batch_advantages
        actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coefficient * dist_entropy

        critic_loss = 0.5 * (new_vals - batch_returns).pow(2).mean()

        return new_probs, dist_entropy, actor_loss, critic_loss

    @torch.no_grad()
    def choose_action(self, observation):
        if not isinstance(observation, np.ndarray):
            observation = np.array(observation)

        observation = np.array(observation).reshape(1, -1)
        state = torch.tensor(observation, dtype=torch.float).to(self.device)
        actions, log_probs = self.actor.act(state)

        if torch.isnan(actions).any():
            print("NaN detected in actions: ", actions)

        log_prob = log_probs.cpu().numpy().flatten()
        value = self.critic(state).cpu().numpy().flatten()

        return actions.cpu().numpy().flatten(), log_prob, value

if __name__ == '__main__':
    input_dims = 3
    n_actions = 3
    alpha = 0.0005
    policy_clip = 0.5
    batch_size = 64
    n_epochs = 2000
    entropy_coefficient = 0.25
    weight_decay = 0.0000001
    mini_batch_size = 64

    agent = PPO(input_dims, n_actions, alpha, policy_clip=policy_clip, batch_size=batch_size, n_epochs=n_epochs, entropy_coefficient=entropy_coefficient, weight_decay=weight_decay, mini_batch_size=mini_batch_size)
    env = EconomicEnv()

    num_episodes = 11

    for episode in tqdm(range(num_episodes)):
        observation, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action, log_prob, value = agent.choose_action(observation)
            next_observation, reward, terminated, info = env.step(action)

            agent.store_transition(observation, action, log_prob, value, reward, done)

            observation = next_observation
            done = terminated

            if len(agent.memory.states) >= agent.memory.batch_size:
                agent.learn()
                agent.memory.clear_memory()


            print(f'Episode {episode}, Actions: {action}, Reward: {reward}')
