import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random

# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(4, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 2)
        self.sigmoid = nn.Sigmoid()
        self.bankrupt = False

    def forward(self, state):
        if self.bankrupt:
            return torch.tensor([0.0, 0.0], dtype=torch.float32)
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return 100 * self.sigmoid(x)


class EconomicEnv:
    def __init__(self):
        self.c = 1

    def demand(self, total_price):
        return torch.clamp(150 - 2 * total_price, min=0)

    def step(self, actions):
        price1, production1 = actions[0, 0], actions[0, 1]
        price2, production2 = actions[1, 0], actions[1, 1]

        if price1 < price2:
            demand1 = self.demand(price1)
            actual_sell1 = torch.min(production1, demand1)
            remaining_demand = self.demand(price2) - actual_sell1
            actual_sell2 = torch.min(production2, remaining_demand)
        else:
            demand2 = self.demand(price2)
            actual_sell2 = torch.min(production2, demand2)
            remaining_demand = self.demand(price1) - actual_sell2
            actual_sell1 = torch.min(production1, remaining_demand)

        revenue1 = price1 * actual_sell1
        revenue2 = price2 * actual_sell2
        cost1 = 10 * production1 + 100
        cost2 = 10 * production2 + 100
        profit1 = revenue1 - cost1
        profit2 = revenue2 - cost2
        return profit1 / 10, profit2 / 10


class ReplayMemory:
    def __init__(self):
        self.memory = []

    def push(self, state, action1, action2, reward1, reward2):
        self.memory.append((state, action1, action2, reward1, reward2))

    def sample(self):
        return random.sample(self.memory, len(self.memory))

    def clear(self):
        self.memory = []

# Initialize actor-critics and optimizers
actor1 = Actor()
actor2 = Actor()
actor_opt1 = optim.Adam(actor1.parameters(), lr=0.0005)
actor_opt2 = optim.Adam(actor2.parameters(), lr=0.0005)

env = EconomicEnv()
memory = ReplayMemory()

num_games = 100
steps_per_game = 100
gamma = 0.99  # Discount factor for future rewards
initial_sigma = 0.5  # Standard deviation for exploration noise

# Track prices, productions, and profits over games
prices1 = []
prices2 = []
productions1 = []
productions2 = []
profits1 = []
profits2 = []

bankruptcy_threshold = 25

for game in range(num_games):
    sigma = initial_sigma * (0.99 ** game)  # Decrease sigma after each game
    prev_actions1 = torch.tensor([0.0, 0.0], dtype=torch.float32, requires_grad=True)
    prev_actions2 = torch.tensor([0.0, 0.0], dtype=torch.float32, requires_grad=True)

    # Tracking for bankruptcy
    consecutive_negatives1 = 0
    consecutive_negatives2 = 0

    actor1.bankrupt = False
    actor2.bankrupt = False

    for step in range(steps_per_game):
        state = torch.cat([prev_actions1, prev_actions2]).unsqueeze(0)  # State includes previous actions

        actions1 = actor1(state)
        actions2 = actor2(state)

        noisy_actions1 = actions1 + sigma * torch.randn_like(actions1)
        noisy_actions2 = actions2 + sigma * torch.randn_like(actions2)

        noisy_actions1 = torch.clamp(noisy_actions1, 0, 100)  # Ensure actions stay within valid range
        noisy_actions2 = torch.clamp(noisy_actions2, 0, 100)  # Ensure actions stay within valid range

        # Remove the extra dimension by squeezing the tensors
        noisy_actions1 = noisy_actions1.squeeze()
        noisy_actions2 = noisy_actions2.squeeze()

        actions = torch.stack([noisy_actions1, noisy_actions2])  # Ensure actions are stacked correctly
        profit1, profit2 = env.step(actions)

        # Track prices, productions, and profits
        prices1.append(noisy_actions1[0].item())
        prices2.append(noisy_actions2[0].item())
        productions1.append(noisy_actions1[1].item())
        productions2.append(noisy_actions2[1].item())
        profits1.append(profit1.item())
        profits2.append(profit2.item())

        # Store experience in memory
        memory.push(state, noisy_actions1, noisy_actions2, profit1.item(), profit2.item())

        # Update previous actions
        prev_actions1 = noisy_actions1.detach().clone().requires_grad_(True)
        prev_actions2 = noisy_actions2.detach().clone().requires_grad_(True)

        sigma *= 0.99  # Decrease sigma over time to reduce exploration as learning progresses

        # Update bankruptcy status based on profit
        consecutive_negatives1 = 0 if profit1.item() >= 0 else consecutive_negatives1 + 1
        consecutive_negatives2 = 0 if profit2.item() >= 0 else consecutive_negatives2 + 1

        if consecutive_negatives1 >= bankruptcy_threshold:
            actor1.bankrupt = True
        if consecutive_negatives2 >= bankruptcy_threshold:
            actor2.bankrupt = True
        if step % 10 == 0:
            print('Game:', game, 'Step:', step)
            print(f"Actions 1 {noisy_actions1.detach().numpy()}, Profit 1 {profit1.item():.2f}")
            print(f'Actions 2 {noisy_actions2.detach().numpy()}', f'Profit 2 {profit2.item():.2f}')

    # Update actor networks at the end of each game using experiences in memory
    experiences = memory.sample()

    states, actions1, actions2, rewards1, rewards2 = zip(*experiences)

    states = torch.cat(states)
    actions1 = torch.stack(actions1)
    actions2 = torch.stack(actions2)
    rewards1 = torch.tensor(rewards1)
    rewards2 = torch.tensor(rewards2)

    # Ensure actions1 and actions2 have shape [1, 2] before passing to env.step
    actions1 = [action.view(1, 2) if action.dim() == 1 else action.unsqueeze(0) if action.dim() == 1 else action for
                action in actions1]
    actions2 = [action.view(1, 2) if action.dim() == 1 else action.unsqueeze(0) if action.dim() == 1 else action for
                action in actions2]

    # Compute loss for actor1
    actor_opt1.zero_grad()
    action_preds1 = actor1(states)
    future_rewards1 = torch.stack([env.step(torch.cat([action_pred1.view(1, 2), action2], dim=0))[0]
                                   for action_pred1, action2 in zip(action_preds1, actions2)])
    future_rewards1 = future_rewards1.view(-1)  # Ensure it's a 1D tensor
    loss1 = -rewards1 + gamma * future_rewards1
    loss1 = loss1.mean()
    loss1.backward()
    actor_opt1.step()

    # Compute loss for actor2
    actor_opt2.zero_grad()
    action_preds2 = actor2(states)
    future_rewards2 = torch.stack([env.step(torch.cat([action1, action_pred2.view(1, 2)], dim=0))[1]
                                   for action_pred2, action1 in zip(action_preds2, actions1)])
    future_rewards2 = future_rewards2.view(-1)  # Ensure it's a 1D tensor
    loss2 = -rewards2 + gamma * future_rewards2
    loss2 = loss2.mean()
    loss2.backward()
    actor_opt2.step()

    memory.clear()  # Clear memory after each game


# Plot profits
plt.figure(figsize=(12, 6))
plt.plot(profits1, label='Firm 1 Profit')
plt.plot(profits2, label='Firm 2 Profit')
plt.xlabel('Episode')
plt.ylabel('Profit')
plt.legend()
plt.title('Profit over Episodes')
plt.savefig(r'D:\studia\WNE\2023_2024\symulacje\zdj\oligopol4\profits.png')


# Plot prices
plt.figure(figsize=(12, 6))
plt.plot(prices1, label='Firm 1 Price')
plt.plot(prices2, label='Firm 2 Price')
plt.xlabel('Episode')
plt.ylabel('Price')
plt.legend()
plt.title('Price over Episodes')
plt.savefig(r'D:\studia\WNE\2023_2024\symulacje\zdj\oligopol4\prices.png')


# Plot productions
plt.figure(figsize=(12, 6))
plt.plot(productions1, label='Firm 1 Production')
plt.plot(productions2, label='Firm 2 Production')
plt.xlabel('Episode')
plt.ylabel('Production')
plt.legend()
plt.title('Production over Episodes')
plt.savefig(r'D:\studia\WNE\2023_2024\symulacje\zdj\oligopol4\productions.png')
