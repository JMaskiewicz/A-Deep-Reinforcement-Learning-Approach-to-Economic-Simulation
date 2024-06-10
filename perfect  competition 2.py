import torch
import torch.nn as nn
import torch.optim as optim

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(10, 16)
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

    def step(self, actions):
        profits = []
        for action in actions:
            price, production = action[0], action[1]
            demand = self.demand(price)
            actual_sell = torch.min(production, demand)
            revenue = price * actual_sell
            cost = 10 * production + 100
            profit = revenue - cost
            profits.append(profit / 10)
        return torch.tensor(profits)

# Create actors for the 5 firms
actors = [Actor() for _ in range(5)]
optimizers = [optim.Adam(actor.parameters(), lr=0.00025) for actor in actors]

env = EconomicEnv()

# Initialize previous actions
prev_actions = [torch.tensor([0.0, 0.0], dtype=torch.float32) for _ in range(5)]

num_episodes = 500
sigma = 0.5  # Standard deviation for exploration noise

for episode in range(num_episodes):
    state = torch.cat(prev_actions).unsqueeze(0)  # State includes previous actions

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
        all_actions = [prev_actions[j] if j != i else action_pred.squeeze() for j in range(5)]
        loss = -env.step(torch.stack(all_actions))[i]  # Use the negative of the profit as loss
        loss.backward()
        optimizers[i].step()

    sigma *= 0.99  # Decrease sigma over time to reduce exploration as learning progresses

    print(f"Episode {episode}:")
    for i in range(5):
        print(f"Actions {i + 1} {noisy_actions[i].detach().numpy()}, Profit {i + 1} {profits[i].item():.2f}")

# Testing the trained actors
test_state = torch.cat(prev_actions).unsqueeze(0)
test_actions = [actor(test_state) for actor in actors]
for i, test_action in enumerate(test_actions):
    print(f"Optimal Actions for Firm {i + 1}: Price {test_action[0,0].item():.2f}, Production {test_action[0,1].item():.2f}")
