import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

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

        cost1 = 10 * production1 + 100  # provide also with 1
        cost2 = 10 * production2 + 100

        profit1 = revenue1 - cost1
        profit2 = revenue2 - cost2

        return profit1/10, profit2/10

env = EconomicEnv()

# Initialize actors and optimizers
actor1 = Actor()
actor2 = Actor()
actor_opt1 = optim.Adam(actor1.parameters(), lr=0.0005)
actor_opt2 = optim.Adam(actor2.parameters(), lr=0.0005)

num_episodes = 200
bankruptcy_threshold = 25
sigma = 0.5  # Standard deviation for exploration noise

for episode in range(num_episodes):
    actor1.bankrupt = False
    actor2.bankrupt = False

    # Tracking for bankruptcy
    consecutive_negatives1 = 0
    consecutive_negatives2 = 0

    cumulative_loss1 = 0
    cumulative_loss2 = 0
    state_history = []
    action_history1 = []
    action_history2 = []
    profit_history1 = []
    profit_history2 = []

    prev_actions1 = torch.tensor([0.0, 0.0], dtype=torch.float32)
    prev_actions2 = torch.tensor([0.0, 0.0], dtype=torch.float32)

    for step in range(100):  # Assuming 100 steps per episode
        state = torch.cat([prev_actions1, prev_actions2]).unsqueeze(0)
        actions1 = actor1(state)
        actions2 = actor2(state)

        # Exploration noise
        noisy_actions1 = torch.clamp(actions1 + sigma * torch.randn_like(actions1), 0, 100)
        noisy_actions2 = torch.clamp(actions2 + sigma * torch.randn_like(actions2), 0, 100)

        actions = torch.stack([noisy_actions1.squeeze(), noisy_actions2.squeeze()])
        profit1, profit2 = env.step(actions)

        # Save history for updating at the end of the episode
        state_history.append(state)
        action_history1.append(noisy_actions1)
        action_history2.append(noisy_actions2)
        profit_history1.append(profit1)
        profit_history2.append(profit2)

        # Update previous actions for the next state
        prev_actions1 = noisy_actions1.detach().squeeze()
        prev_actions2 = noisy_actions2.detach().squeeze()

        # Update bankruptcy status based on profit
        consecutive_negatives1 = 0 if profit1.item() >= 0 else consecutive_negatives1 + 1
        consecutive_negatives2 = 0 if profit2.item() >= 0 else consecutive_negatives2 + 1

        if consecutive_negatives1 >= bankruptcy_threshold:
            actor1.bankrupt = True
        if consecutive_negatives2 >= bankruptcy_threshold:
            actor2.bankrupt = True

        if step % 10 == 0:
            print('Episode:', episode, 'Step:', step)
            print(f"Actions 1 {noisy_actions1.detach().numpy()}, Profit 1 {profit1.item():.2f}")
            print(f'Actions 2 {noisy_actions2.detach().numpy()}', f'Profit 2 {profit2.item():.2f}')

    # Compute cumulative loss for the episode
    for i in range(100):
        state = state_history[i]
        predicted_actions1 = actor1(state).squeeze()
        predicted_actions2 = actor2(state).squeeze()

        loss1 = -env.step(torch.stack([predicted_actions1, action_history2[i].detach().squeeze()]))[0]
        loss2 = -env.step(torch.stack([action_history1[i].detach().squeeze(), predicted_actions2]))[1]

        cumulative_loss1 = cumulative_loss1 + loss1.requires_grad_()
        cumulative_loss2 = cumulative_loss2 + loss2.requires_grad_()

        # Apply gradients
    actor_opt1.zero_grad()
    cumulative_loss1.backward()
    actor_opt1.step()

    actor_opt2.zero_grad()
    cumulative_loss2.backward()
    actor_opt2.step()

    sigma *= 0.99
    print(f"Episode {episode}: Profit1 {sum(profit_history1)}, Profit2 {sum(profit_history2)}")