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

    def demand(self, total_price):
        return torch.clamp(150 - 2 * total_price, min=0)

    def step(self, actions):
        prices = actions[:, 0]
        productions = actions[:, 1]

        # Sort the firms by their prices
        sorted_indices = torch.argsort(prices)
        sorted_prices = prices[sorted_indices]
        sorted_productions = productions[sorted_indices]

        total_demand = self.demand(torch.sum(sorted_prices))
        remaining_demand = total_demand
        actual_sells = torch.zeros_like(sorted_productions)

        for i in range(len(sorted_prices)):
            if remaining_demand > 0:
                actual_sell = torch.min(sorted_productions[i], remaining_demand)
                actual_sells[i] = actual_sell
                remaining_demand -= actual_sell
            else:
                actual_sells[i] = 0

        revenues = sorted_prices * actual_sells
        costs = 10 * sorted_productions + 100
        profits = revenues - costs

        # Re-order the profits according to the original order of the firms
        unsorted_profits = profits[torch.argsort(sorted_indices)]

        return unsorted_profits / 10

# Create two actors for the 5 firms
actor1 = Actor()
actor2 = Actor()
actor3 = Actor()
actor4 = Actor()
actor5 = Actor()

env = EconomicEnv()
actor_opt1 = optim.Adam(actor1.parameters(), lr=0.00075)
actor_opt2 = optim.Adam(actor2.parameters(), lr=0.00075)
actor_opt3 = optim.Adam(actor3.parameters(), lr=0.00075)
actor_opt4 = optim.Adam(actor4.parameters(), lr=0.00075)
actor_opt5 = optim.Adam(actor5.parameters(), lr=0.00075)

# Initialize previous actions
prev_actions1 = torch.tensor([0.0, 0.0], dtype=torch.float32)
prev_actions2 = torch.tensor([0.0, 0.0], dtype=torch.float32)
prev_actions3 = torch.tensor([0.0, 0.0], dtype=torch.float32)
prev_actions4 = torch.tensor([0.0, 0.0], dtype=torch.float32)
prev_actions5 = torch.tensor([0.0, 0.0], dtype=torch.float32)


num_episodes = 500
sigma = 0.5  # Standard deviation for exploration noise

for episode in range(num_episodes):
    state = torch.cat([prev_actions1, prev_actions2]).unsqueeze(0)  # State includes previous actions

    actions1 = actor1(state)
    actions2 = actor2(state)
    actions3 = actor3(state)
    actions4 = actor4(state)
    actions5 = actor5(state)

    noisy_actions1 = actions1 + sigma * torch.randn_like(actions1)
    noisy_actions2 = actions2 + sigma * torch.randn_like(actions2)
    noisy_actions3 = actions3 + sigma * torch.randn_like(actions3)
    noisy_actions4 = actions4 + sigma * torch.randn_like(actions4)
    noisy_actions5 = actions5 + sigma * torch.randn_like(actions5)

    noisy_actions1 = torch.clamp(noisy_actions1, 0, 100)
    noisy_actions2 = torch.clamp(noisy_actions2, 0, 100)
    noisy_actions3 = torch.clamp(noisy_actions3, 0, 100)
    noisy_actions4 = torch.clamp(noisy_actions4, 0, 100)
    noisy_actions5 = torch.clamp(noisy_actions5, 0, 100)


    actions = torch.stack([noisy_actions1.squeeze(), noisy_actions2.squeeze(), noisy_actions3.squeeze(), noisy_actions4.squeeze(), noisy_actions5.squeeze()])
    profit1, profit2, profit3, profit4, profit5 = env.step(actions)

    # Update previous actions
    prev_actions1 = noisy_actions1.detach().squeeze()
    prev_actions2 = noisy_actions2.detach().squeeze()
    prev_actions3 = noisy_actions3.detach().squeeze()
    prev_actions4 = noisy_actions4.detach().squeeze()
    prev_actions5 = noisy_actions5.detach().squeeze()

    # Update actor1
    actor_opt1.zero_grad()
    action_pred1 = actor1(state)  # Predicted actions
    all_actions = torch.stack([action_pred1.squeeze(), prev_actions2, prev_actions3, prev_actions4, prev_actions5])
    loss1 = -env.step(all_actions)[0]  # Use the negative of the profit as loss
    loss1.backward()
    actor_opt1.step()

    # Update actor2
    actor_opt2.zero_grad()
    action_pred2 = actor2(state)  # Predicted actions
    all_actions = torch.stack([prev_actions1, action_pred2.squeeze(), prev_actions3, prev_actions4, prev_actions5])
    loss2 = -env.step(all_actions)[1]  # Use the negative of the profit as loss
    loss2.backward()
    actor_opt2.step()

    # Update actor3
    actor_opt3.zero_grad()
    action_pred3 = actor3(state)  # Predicted actions
    all_actions = torch.stack([prev_actions1, prev_actions2, action_pred3.squeeze(), prev_actions4, prev_actions5])
    loss3 = -env.step(all_actions)[2]  # Use the negative of the profit as loss
    loss3.backward()
    actor_opt3.step()

    # Update actor4
    actor_opt4.zero_grad()
    action_pred4 = actor4(state)  # Predicted actions
    all_actions = torch.stack([prev_actions1, prev_actions2, prev_actions3, action_pred4.squeeze(), prev_actions5])
    loss4 = -env.step(all_actions)[3]  # Use the negative of the profit as loss
    loss4.backward()
    actor_opt4.step()

    # Update actor5
    actor_opt5.zero_grad()
    action_pred5 = actor5(state)  # Predicted actions
    all_actions = torch.stack([prev_actions1, prev_actions2, prev_actions3, prev_actions4, action_pred5.squeeze()])
    loss5 = -env.step(all_actions)[4]  # Use the negative of the profit as loss
    loss5.backward()
    actor_opt5.step()

    sigma *= 0.99  # Decrease sigma over time to reduce exploration as learning progresses

    print(f"Episode {episode}:\nActions 1 {noisy_actions1.detach().numpy()}, Profit 1 {profit1.item():.2f}")
    print(f'Actions 2 {noisy_actions2.detach().numpy()}', f'Profit 2 {profit2.item():.2f}')
    print(f'Actions 3 {noisy_actions3.detach().numpy()}', f'Profit 3 {profit3.item():.2f}')
    print(f'Actions 4 {noisy_actions4.detach().numpy()}', f'Profit 4 {profit4.item():.2f}')
    print(f'Actions 5 {noisy_actions5.detach().numpy()}', f'Profit 5 {profit5.item():.2f}')

# Testing the trained actors
test_state = torch.cat([prev_actions1, prev_actions2]).unsqueeze(0)
test_actions1 = actor1(test_state)
test_actions2 = actor2(test_state)
print(f"Optimal Actions for Firm 1: Price {test_actions1[0,0].item():.2f}, Production {test_actions1[0,1].item():.2f}")
print(f"Optimal Actions for Firm 2: Price {test_actions2[0,0].item():.2f}, Production {test_actions2[0,1].item():.2f}")
print(f"Optimal Actions for Firm 3: Price {test_actions2[0,0].item():.2f}, Production {test_actions2[0,1].item():.2f}")
print(f"Optimal Actions for Firm 4: Price {test_actions2[0,0].item():.2f}, Production {test_actions2[0,1].item():.2f}")
print(f"Optimal Actions for Firm 5: Price {test_actions2[0,0].item():.2f}, Production {test_actions2[0,1].item():.2f}")
