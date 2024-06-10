import torch
import torch.nn as nn
import torch.optim as optim

# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)

class Actor(nn.Module):
    def __init__(self, num_agents):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(2 * num_agents, 16)  # 2*num_agents inputs: price and production of all agents
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

    def demand(self, total_price):
        return torch.clamp(150 - 2 * total_price, min=0)

    def step(self, actions):
        # Sort agents by price
        sorted_indices = torch.argsort(actions[:, 0])
        sorted_actions = actions[sorted_indices]

        total_demand = self.demand(sorted_actions[0, 0])
        actual_sells = torch.zeros(len(actions))
        remaining_demand = total_demand

        for i in range(len(actions)):
            price = sorted_actions[i, 0]
            production = sorted_actions[i, 1]

            demand_at_price = self.demand(price)
            actual_sell = torch.min(production, remaining_demand)
            actual_sells[sorted_indices[i]] = actual_sell

            remaining_demand = remaining_demand - actual_sell
            remaining_demand = torch.clamp(remaining_demand, min=0)

        revenues = sorted_actions[:, 0] * actual_sells
        costs = 10 * sorted_actions[:, 1] + 100

        profits = revenues - costs

        return profits / 10

# Parameters
num_agents = 2
num_episodes = 100
sigma = 0.5  # Standard deviation for exploration noise
lr = 0.00075

# Create actors and optimizers for each agent
actors = [Actor(num_agents) for _ in range(num_agents)]
optimizers = [optim.Adam(actor.parameters(), lr=lr) for actor in actors]
env = EconomicEnv()

# Initialize previous actions
# Initialize previous actions
prev_actions = [torch.tensor([0.0, 0.0], dtype=torch.float32) for _ in range(num_agents)]

for episode in range(num_episodes):
    # Create state as concatenation of prices and productions of all agents
    state = torch.cat(prev_actions).unsqueeze(0)  # State includes previous actions of all agents

    actions = []
    for actor in actors:
        actions.append(actor(state))

    noisy_actions = []
    for action in actions:
        noisy_action = action + sigma * torch.randn_like(action)
        noisy_action = torch.clamp(noisy_action, 0, 100)  # Ensure actions stay within valid range
        noisy_actions.append(noisy_action)

    actions_tensor = torch.stack([action.squeeze() for action in noisy_actions])
    profits = env.step(actions_tensor)

    # Update previous actions
    prev_actions = [action.detach().squeeze() for action in noisy_actions]

    for i in range(num_agents):
        optimizers[i].zero_grad()
        action_pred = actors[i](state)  # Predicted actions
        action_list = [prev_actions[j] if j != i else action_pred.squeeze() for j in range(num_agents)]
        profit = env.step(torch.stack(action_list))[i]
        loss = -profit  # Use the negative of the profit as loss
        loss.backward()
        optimizers[i].step()

    sigma *= 0.99  # Decrease sigma over time to reduce exploration as learning progresses

    print(f"Episode {episode}:")
    for i in range(num_agents):
        print(f"Actions {i + 1} {noisy_actions[i].detach().numpy()}, Profit {i + 1} {profits[i].item():.2f}")
