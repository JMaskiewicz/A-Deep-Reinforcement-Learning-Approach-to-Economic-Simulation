import torch
import torch.nn as nn
import torch.optim as optim

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(1, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 2)  # Two outputs: price and production
        self.sigmoid = nn.Sigmoid()

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return 100 * self.sigmoid(x)  # Outputs range [0, 100]

class EconomicEnv:
    def __init__(self):
        self.c = 1
    def demand(self, price):
        return torch.clamp(150 - 2 * price, min=0)

    def cost(self, production):
        return self.c * production + self.f

    def step(self, actions):
        price, production = actions[:, 0], actions[:, 1]
        demand = self.demand(price)
        actual_sell = torch.min(production, demand)
        revenue = price * actual_sell
        cost = 100 + 10 * production
        profit = revenue - cost

        return profit

actor = Actor()
env = EconomicEnv()
actor_opt = optim.Adam(actor.parameters(), lr=0.005)
num_episodes = 500
sigma = 1  # Standard deviation for exploration noise

# Assume these are filled during the training loop
profits = []
prices = []
productions = []

for episode in range(num_episodes):
    state = torch.tensor([[0.0]], dtype=torch.float32)  # Dummy state
    actions = actor(state)
    noisy_actions = actions + sigma * torch.randn_like(actions)
    noisy_actions = torch.clamp(noisy_actions, 0, 100)  # Ensure actions stay within valid range
    profit = env.step(noisy_actions)

    # Update actor
    actor_opt.zero_grad()
    action_pred = actor(state)  # Predicted actions
    loss = -env.step(action_pred)  # Use the negative of the profit as loss
    loss.backward()
    actor_opt.step()

    profits.append(profit.item())
    prices.append(noisy_actions[0, 0].item())
    productions.append(noisy_actions[0, 1].item())

    sigma *= 0.99  # Decrease sigma over time to reduce exploration as learning progresses

    if episode % 1 == 0:
        print(f"Episode {episode}: Actions {noisy_actions.detach().numpy()}, Profit {profit.item():.2f}")

# Testing the trained actor
test_actions = actor(torch.tensor([[0.0]], dtype=torch.float32))
print(f"Optimal Actions: Price {test_actions[0,0].item():.2f}, Production {test_actions[0,1].item():.2f}")

import matplotlib.pyplot as plt

# Create a figure and a set of subplots
fig, axs = plt.subplots(2, 1, figsize=(10, 10))  # 2 rows, 1 column, figure size 10x10 inches

# Plotting the profit over time on the first subplot
axs[0].plot(profits, label='Profit')
axs[0].set_title('Profit Over Episodes - Monopoly 1')
axs[0].set_xlabel('Episode')
axs[0].set_ylabel('Profit')
axs[0].legend()

# Plotting price and production actions over time on the second subplot
axs[1].plot(prices, label='Price')
axs[1].plot(productions, label='Production')
axs[1].set_title('Action Trends Over Episodes - Monopoly 1')
axs[1].set_xlabel('Episode')
axs[1].set_ylabel('Actions')
axs[1].legend()

# Adjust layout so the subplots do not overlap
plt.tight_layout()


# Save the figure to a file
plt.savefig(r'D:\studia\WNE\2023_2024\symulacje\zdj\Monopoly1_combined_plot.png')