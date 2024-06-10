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
        return torch.clamp(150 - 2 * price, min=0)  #

    def step(self, actions):
        sorted_indices = torch.argsort(actions[:, 0])
        sorted_actions = actions[sorted_indices]
        sales = torch.zeros_like(actions[:, 1])
        remaining_demand = self.demand(sorted_actions[0][0])

        for i in range(len(sorted_actions)):
            price, production = sorted_actions[i]
            if i > 0:
                remaining_demand = self.demand(price) - torch.sum(sales[:i])
            actual_sell = torch.min(production, remaining_demand)
            sales[i] = actual_sell

        profits = torch.zeros(len(actions))
        for i in range(len(sorted_actions)):
            price, production = sorted_actions[i]
            actual_sell = sales[i]
            revenue = price * actual_sell
            cost = 10 * production + 100  # fixed cost plus variable cost
            profit = revenue - cost
            profits[i] = profit

        unsorted_profits = profits[torch.argsort(sorted_indices)]
        return unsorted_profits

num_agents = 10
actors = [Actor(num_agents) for _ in range(num_agents)]
optimizers = [optim.Adam(actor.parameters(), lr=0.000075) for actor in actors]

env = EconomicEnv()
num_episodes = 2000
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

    sigma *= 0.995  # Decrease sigma over time to reduce exploration as learning progresses

    if episode % 100 == 0:
        print(f"Episode {episode}:")
        for i in range(num_agents):
            print(f"Actions {i + 1}: {noisy_actions[i].detach().numpy()}, Profit {profits[i].item():.2f}")

        # Print detailed debug information for the episode
        for i in range(num_agents):
            price, production = actions_tensor[i]
            print(f"Agent {i + 1}: Price={price.item():.2f}, Production={production.item():.2f}, "
                  f"Profit={profits[i].item():.2f}")

# Testing the trained actors
test_state = torch.cat(prev_actions).unsqueeze(0)
test_actions = [actor(test_state) for actor in actors]
for i, test_action in enumerate(test_actions):
    print(f"Optimal Actions for Firm {i + 1}: Price {test_action[0,0].item():.2f}, Production {test_action[0,1].item():.2f}")
