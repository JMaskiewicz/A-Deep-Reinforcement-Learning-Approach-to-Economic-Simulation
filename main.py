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
        batches = [indices[i:i + self.batch_size] for i in batch_start]

        return self.states, self.actions, self.probs, self.vals, self.rewards, self.dones, batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(torch.tensor(state, dtype=torch.float).unsqueeze(0))
        self.actions.append(torch.tensor(action, dtype=torch.float).unsqueeze(0))
        if probs is not None:
            self.probs.append(torch.tensor(probs, dtype=torch.float).unsqueeze(0))
        else:
            self.probs.append(None)
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
        self.probs = torch.cat([p for p in self.probs if p is not None], dim=0).to(self.device) if any(p is not None for p in self.probs) else None
        self.vals = torch.cat(self.vals, dim=0).to(self.device)
        self.rewards = torch.cat(self.rewards, dim=0).to(self.device)
        self.dones = torch.cat(self.dones, dim=0).to(self.device)

class BaseNetwork(nn.Module):
    def __init__(self, input_dims, output_dims, dropout_rate=0.2):
        super(BaseNetwork, self).__init__()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        layer_dims = [64, 64]
        for i in range(len(layer_dims)):
            self.layers.append(nn.Linear(input_dims if i == 0 else layer_dims[i-1], layer_dims[i]))
            self.batch_norms.append(nn.BatchNorm1d(layer_dims[i]))

        self.final_layer = nn.Linear(layer_dims[-1], output_dims)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, state):
        x = state
        for layer, bn in zip(self.layers, self.batch_norms):
            x = layer(x)
            if x.size(0) > 1:  # Apply batch normalization only if batch size > 1
                x = bn(x)
            x = self.relu(x)
            x = self.dropout(x)
        return x

class ActorNetwork(BaseNetwork):
    def __init__(self, n_actions, input_dims, dropout_rate=0.2):
        super(ActorNetwork, self).__init__(input_dims, n_actions, dropout_rate)

    def forward(self, state):
        x = super(ActorNetwork, self).forward(state)
        x = self.final_layer(x)
        x = torch.sigmoid(x)
        return x

class CriticNetwork(BaseNetwork):
    def __init__(self, input_dims, dropout_rate=0.2):
        super(CriticNetwork, self).__init__(input_dims, 1, dropout_rate)

    def forward(self, state):
        x = super(CriticNetwork, self).forward(state)
        q = self.final_layer(x)
        return q

class PPO:
    def __init__(self, input_dims, n_actions, alpha=0.0003, policy_clip=0.2, batch_size=64, n_epochs=10, entropy_coefficient=0.01, weight_decay=0.01, mini_batch_size=64):
        self.device = torch.device("cpu")
        print(f"Using device: {self.device}")

        self.actor = ActorNetwork(n_actions, input_dims).to(self.device)
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
        self.actor.train()
        self.critic.train()

        # Step 1: Prepare all data from memory and calculate advantages
        self.memory.stack_tensors()
        state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, batches = self.memory.generate_batches()

        advantages = reward_arr - vals_arr

        # Step 2: Iterate over training epochs
        for _ in range(self.n_epochs):
            # Shuffle the batches to randomize the learning process
            indices = torch.randperm(len(state_arr))

            for batch in batches:
                batch_indices = indices[batch]
                batch_states = state_arr[batch_indices]
                batch_actions = action_arr[batch_indices]
                batch_old_probs = old_prob_arr[batch_indices]
                batch_vals = vals_arr[batch_indices]
                batch_advantages = advantages[batch_indices]

                # Calculate new probabilities, values, and entropy
                new_probs, dist_entropy = self.actor(batch_states), None  # Assuming a distribution is returned
                new_vals = self.critic(batch_states)

                # Calculate ratio for clipped surrogate objective
                ratio = (new_probs / batch_old_probs).exp()  # Ensure old_probs are log probs if this line is used
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.policy_clip, 1 + self.policy_clip) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coefficient * dist_entropy.mean()

                # Calculate critic loss as mean squared error
                critic_loss = 0.5 * (batch_vals - reward_arr[batch_indices]).pow(2).mean()

                # Step 3: Take gradient step
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

        # Clear memory after updating
        self.memory.clear_memory()
        print("Learning step completed")

    @torch.no_grad()
    def choose_action(self, observation):
        if not isinstance(observation, np.ndarray):
            observation = np.array(observation)

        observation = np.array(observation).reshape(1, -1)
        state = torch.tensor(observation, dtype=torch.float).to(self.device)
        actions = self.actor(state).cpu().numpy().flatten()
        actions = actions * 100  # Scale actions to the range [0, 100]
        log_prob = None  # Not applicable for continuous actions
        value = self.critic(state).cpu().numpy().flatten()

        return actions, log_prob, value

class EconomicEnv(gym.Env):
    def __init__(self):
        super(EconomicEnv, self).__init__()
        self.action_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([0, 0, 0]), high=np.array([100, 100, 200]), dtype=np.float32)
        self.current_step = 0

        self.production = 0
        self.price = 0
        self.demand = 0

    def calculate_demand(self, price):
        return max(200 - 5 * price, 0)

    def calculate_cost(self, production):
        return 20 + 1.5 * production if production > 0 else 0

    def step(self, action):
        if len(action) != 2:
            raise ValueError("Action must contain exactly two elements.")
        production, price = map(int, np.round(action))
        self.production, self.price = production, price
        self.demand = self.calculate_demand(price)

        cost = self.calculate_cost(production)
        revenue = min(production, self.demand) * price

        profit = revenue - cost
        reward = profit

        state = np.array([self.production, self.price, self.demand])
        terminated = self.current_step >= 500
        truncated = False

        self.current_step += 1

        return state, reward, terminated, truncated, {'profit': profit, 'revenue': revenue, 'cost': cost}

    def reset(self, **kwargs):
        self.production = np.random.randint(0, 101)
        self.price = np.random.randint(0, 101)
        self.demand = self.calculate_demand(self.price)
        self.current_step = 0
        state = np.array([self.production, self.price, self.demand])
        return state, {}


if __name__ == '__main__':
    input_dims = 3
    n_actions = 2
    alpha = 0.0003
    policy_clip = 0.2
    batch_size = 64
    n_epochs = 10
    entropy_coefficient = 0.01
    weight_decay = 0.01
    mini_batch_size = 64

    agent = PPO(input_dims, n_actions, alpha, policy_clip=policy_clip, batch_size=batch_size, n_epochs=n_epochs, entropy_coefficient=entropy_coefficient, weight_decay=weight_decay, mini_batch_size=mini_batch_size)
    env = EconomicEnv()

    num_episodes = 500

    for episode in tqdm(range(num_episodes)):
        observation, _ = env.reset()
        done = False

        while not done:
            action, log_prob, value = agent.choose_action(observation)
            next_observation, reward, terminated, truncated, info = env.step(action)

            agent.store_transition(observation, action, log_prob, value, reward, done)

            observation = next_observation
            done = terminated or truncated
            print('Action:', action, 'Reward:', reward)

            if len(agent.memory.states) >= agent.memory.batch_size:
                agent.learn()
                agent.memory.clear_memory()

        print(f"Episode {episode} finished with profit: {info['profit']}")