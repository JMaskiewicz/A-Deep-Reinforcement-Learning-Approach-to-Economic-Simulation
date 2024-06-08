import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import gym
from gym import spaces

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
        self.probs.append(torch.tensor(probs, dtype=torch.float).unsqueeze(0))
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
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state):
        x = super(ActorNetwork, self).forward(state)
        x = self.final_layer(x)
        x = self.softmax(x)
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
        start_time = time.time()
        print('\n', "-" * 100)
        self.actor.train()
        self.critic.train()

        self.memory.stack_tensors()

        state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, batches = self.memory.generate_batches()

        state_arr = state_arr.clone().detach().to(self.device)
        action_arr = action_arr.clone().detach().to(self.device)
        old_prob_arr = old_prob_arr.clone().detach().to(self.device)
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
                batch_old_probs = old_prob_arr[minibatch_indices].clone().detach().to(self.device)
                batch_advantages = advantages[minibatch_indices].clone().detach().to(self.device)
                batch_returns = reward_arr[minibatch_indices].clone().detach().to(self.device)

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()

                new_probs, dist_entropy, actor_loss, critic_loss = self.calculate_loss(batch_states, batch_actions, batch_old_probs, batch_advantages, batch_returns)

                actor_loss.backward()
                self.actor_optimizer.step()
                critic_loss.backward()
                self.critic_optimizer.step()

        self.memory.clear_memory()
        end_time = time.time()
        episode_time = end_time - start_time
        print(f"Learning completed in {episode_time} seconds")
        print("-" * 100)

    def calculate_loss(self, batch_states, batch_actions, batch_old_probs, batch_advantages, batch_returns):
        new_probs = self.actor(batch_states)
        new_dist = torch.distributions.Categorical(new_probs)
        new_log_probs = new_dist.log_prob(batch_actions)

        prob_ratios = torch.exp(new_log_probs - batch_old_probs)

        surr1 = prob_ratios * batch_advantages
        surr2 = torch.clamp(prob_ratios, 1.0 - self.policy_clip, 1.0 + self.policy_clip) * batch_advantages
        actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coefficient * new_dist.entropy().mean()

        values = self.critic(batch_states).squeeze(-1)
        critic_loss = (batch_returns - values).pow(2).mean()

        return new_probs, new_dist.entropy(), actor_loss, critic_loss

    @torch.no_grad()
    def choose_action(self, observation):
        if not isinstance(observation, np.ndarray):
            observation = np.array(observation)

        observation = np.array(observation).reshape(1, -1)
        state = torch.tensor(observation, dtype=torch.float).to(self.device)
        probs = self.actor(state)

        dist = torch.distributions.Categorical(probs)
        action = dist.sample().cpu().numpy()
        log_prob = dist.log_prob(dist.sample()).cpu().numpy()
        value = self.critic(state).cpu().numpy().flatten()

        return action, log_prob, value

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
        return max(200 - 2 * price, 0)

    def calculate_cost(self, production):
        return 20 + 1.5 * production if production > 0 else 0

    def step(self, action):
        production, price = map(int, np.round(action * 100))
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
    update_interval = 10

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
