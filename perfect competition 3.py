import torch
import torch.nn as nn
import torch.optim as optim

class Actor(nn.Module):
    def __init__(self, num_agents):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_agents * 2, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 2)  # Two outputs: price and production
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

    def step(self, actions):
        # Sort actions by price
        actions = sorted(actions, key=lambda x: x[0])
        total_demand = 150
        remaining_demand = total_demand
        sales = []

        for price, production in actions:
            demand = self.demand(price)
            actual_sell = torch.min(production, torch.tensor(remaining_demand, dtype=torch.float32))
            sales.append(actual_sell)
            remaining_demand -= actual_sell.item()

        profits = []
        for i, action in enumerate(actions):
            price, production = action
            actual_sell = sales[i]
            revenue = price * actual_sell
            cost = 100 + 10 * production
            profit = revenue - cost
            profits.append(profit)

        return torch.stack(profits)  # Return tensor with requires_grad=True

num_agents = 5
actors = [Actor(num_agents) for _ in range(num_agents)]
optimizers = [optim.Adam(actor.parameters(), lr=0.000075) for actor in actors]

env = EconomicEnv()
num_episodes = 1000
sigma = 0.5  # Standard deviation for exploration noise

# Initialize previous actions
prev_actions = [torch.tensor([0.0, 0.0], dtype=torch.float32, requires_grad=True) for _ in range(num_agents)]

for episode in range(num_episodes):
    state = torch.cat(prev_actions).unsqueeze(0)  # State includes previous actions of all agents

    actions = [actor(state) for actor in actors]
    noisy_actions = [torch.clamp(action + sigma * torch.randn_like(action), 0, 100) for action in actions]

    actions_tensor = torch.stack([action.squeeze() for action in noisy_actions])
    profits = env.step(actions_tensor)

    # Update previous actions
    prev_actions = [action.detach().squeeze() for action in noisy_actions]

    # Update each actor
    for i, actor in enumerate(actors):
        optimizers[i].zero_grad()
        action_pred = actor(state)  # Predicted actions
        all_actions = [prev_actions[j] if j != i else action_pred.squeeze() for j in range(num_agents)]
        loss = -env.step(torch.stack(all_actions))[i]  # Use the negative of the profit as loss
        loss.backward()
        optimizers[i].step()

    sigma *= 0.99  # Decrease sigma over time to reduce exploration as learning progresses

    if episode % 10 == 0:
        print(f"Episode {episode}:")
        for i in range(num_agents):
            print(f"Actions {i + 1}: {noisy_actions[i].detach().numpy()}, Profit {profits[i].item():.2f}")

# Testing the trained actors
test_state = torch.cat(prev_actions).unsqueeze(0)
test_actions = [actor(test_state) for actor in actors]
for i, test_action in enumerate(test_actions):
    print(f"Optimal Actions for Firm {i + 1}: Price {test_action[0,0].item():.2f}, Production {test_action[0,1].item():.2f}")
