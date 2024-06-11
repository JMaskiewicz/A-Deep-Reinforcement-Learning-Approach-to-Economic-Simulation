import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

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


# Economic Environment
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
        cost1 = 20 * production1 + 100
        cost2 = 10 * production2 + 100
        profit1 = revenue1 - cost1
        profit2 = revenue2 - cost2
        return profit1 / 10, profit2 / 10


# Initialize actor-critics and optimizers
actor1 = Actor()
actor2 = Actor()
actor_opt1 = optim.Adam(actor1.parameters(), lr=0.0005)
actor_opt2 = optim.Adam(actor2.parameters(), lr=0.0005)

env = EconomicEnv()

num_games = 100
steps_per_game = 100
gamma = 0  # Discount factor for future rewards
initial_sigma = 1  # Standard deviation for exploration noise

# Track prices, productions, and profits over games
prices1 = []
prices2 = []
productions1 = []
productions2 = []
profits1 = []
profits2 = []

for game in range(num_games):
    sigma = initial_sigma * (0.99 ** game)  # Decrease sigma after each game
    prev_actions1 = torch.tensor([0.0, 0.0], dtype=torch.float32, requires_grad=True)
    prev_actions2 = torch.tensor([0.0, 0.0], dtype=torch.float32, requires_grad=True)

    # Track prices, productions, and profits over games
    prices1 = []
    prices2 = []
    productions1 = []
    productions2 = []
    profits1 = []
    profits2 = []

    # Tracking for bankruptcy
    consecutive_negatives1 = 0
    consecutive_negatives2 = 0
    bankruptcy_threshold = 25

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

        actions = torch.stack([noisy_actions1.squeeze(), noisy_actions2.squeeze()])
        profit1, profit2 = env.step(actions)

        # Track prices, productions, and profits
        prices1.append(noisy_actions1[0, 0].item())
        prices2.append(noisy_actions2[0, 0].item())
        productions1.append(noisy_actions1[0, 1].item())
        productions2.append(noisy_actions2[0, 1].item())
        profits1.append(profit1.item())
        profits2.append(profit2.item())

        # Update previous actions
        prev_actions1 = noisy_actions1.detach().clone().requires_grad_(True).squeeze()
        prev_actions2 = noisy_actions2.detach().clone().requires_grad_(True).squeeze()

        # Update actor1
        actor_opt1.zero_grad()
        action_pred1 = actor1(state)  # Predicted actions
        loss1 = -env.step(torch.stack([action_pred1.squeeze(), prev_actions2]))[0]
        loss1.backward()
        actor_opt1.step()

        # Update actor2
        actor_opt2.zero_grad()
        action_pred2 = actor2(state)  # Predicted actions
        loss2 = -env.step(torch.stack([prev_actions1, action_pred2.squeeze()]))[1]
        loss2.backward()
        actor_opt2.step()

        sigma *= 0.95  # Decrease sigma over time to reduce exploration as learning progresses

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
