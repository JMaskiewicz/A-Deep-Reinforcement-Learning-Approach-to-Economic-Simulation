import torch
import torch.nn as nn
import torch.optim as optim

# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)

# Actor for Firm 1
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

# Actor-Critic for Firm 2
class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc_actor = nn.Linear(128, 2)
        self.fc_critic = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.bankrupt = False

    def forward(self, state):
        if self.bankrupt:
            return torch.tensor([0.0, 0.0], dtype=torch.float32), torch.tensor([0.0], dtype=torch.float32)
        x = self.relu(self.fc1(state))
        action = 100 * self.sigmoid(self.fc_actor(x))
        value = self.fc_critic(x)
        return action, value

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
        cost2 = 5 * production2 + 100
        profit1 = revenue1 - cost1
        profit2 = revenue2 - cost2
        return profit1 / 10, profit2 / 10


# Initialize actors and optimizers
actor1 = Actor()
actor_critic2 = ActorCritic()
opt_actor1 = optim.Adam(actor1.parameters(), lr=0.00075)
opt_actor_critic2 = optim.Adam(actor_critic2.parameters(), lr=0.0005)

env = EconomicEnv()

num_episodes = 2000
gamma = 0.99  # Discount factor for future rewards
sigma = 0.5   # Standard deviation for exploration noise

# Tracking for bankruptcy
consecutive_negatives1 = 0
consecutive_negatives2 = 0
bankruptcy_threshold = 50

for episode in range(num_episodes):
    state = torch.cat([torch.tensor([50.0, 30.0]), torch.tensor([50.0, 30.0])]).unsqueeze(0)

    # Get actions from actor1
    actions1 = actor1(state)  # Ensure actions1 has shape [2]
    actions1 = actions1.squeeze()

    # Get actions and state values from actor_critic2
    actions2, state_value2 = actor_critic2(state)
    actions2 = actions2.squeeze()

    # Add exploration noise and clamp
    noisy_actions1 = torch.clamp(actions1 + sigma * torch.randn_like(actions1), 0, 100)
    noisy_actions2 = torch.clamp(actions2 + sigma * torch.randn_like(actions2), 0, 100)

    # Combine actions into a single tensor
    actions = torch.stack([noisy_actions1, noisy_actions2])

    # Debugging: Print the shape of actions
    print(f"Episode {episode}: Actions shape: {actions.shape}")

    # Get profits for both firms
    profit1, profit2 = env.step(actions)

    # Ensure profits require gradients
    profit1 = profit1.requires_grad_()
    profit2 = profit2.requires_grad_()

    # Prepare for the next state's value estimate
    next_state = state  # In a real case, you'd get this from the environment
    _, next_value2 = actor_critic2(next_state)  # Get next state value prediction

    # Calculate TD target and error
    td_target = profit2 + gamma * next_value2.detach()  # Detach to prevent connection to next graph
    td_error = td_target - state_value2

    # Backward and optimize for actor1
    opt_actor1.zero_grad()
    loss1 = -profit1  # Convert profit to loss
    loss1.backward(retain_graph=True)  # Retain graph for subsequent backward pass
    opt_actor1.step()

    # Backward and optimize for actor_critic2
    opt_actor_critic2.zero_grad()
    actor_loss2 = (-state_value2 * actions2).mean()  # Policy gradient part
    total_loss2 = td_error.pow(2).mean() + actor_loss2  # Total loss
    total_loss2.backward()  # No need for retain_graph=True because this is the only backward pass
    opt_actor_critic2.step()

    # Decay exploration noise
    sigma *= 0.99

    # Update bankruptcy status based on profit
    consecutive_negatives1 = 0 if profit1.item() >= 0 else consecutive_negatives1 + 1
    consecutive_negatives2 = 0 if profit2.item() >= 0 else consecutive_negatives2 + 1

    if consecutive_negatives1 >= bankruptcy_threshold:
        actor1.bankrupt = True
    if consecutive_negatives2 >= bankruptcy_threshold:
        actor_critic2.bankrupt = True

    # Optional: Print episode results
    if episode % 100 == 0:  # Print every 100 episodes
        print(f"Episode {episode}:\nActions 1 {noisy_actions1.detach().numpy()}, Profit 1 {profit1.item():.2f}")
        print(f"Actions 2 {noisy_actions2.detach().numpy()}, Profit 2 {profit2.item():.2f}")


# Final testing to check actions at the end of training
test_state = torch.cat([torch.tensor([50.0, 30.0]), torch.tensor([50.0, 30.0])]).unsqueeze(0)
test_actions1 = actor1(test_state)
test_actions2, _ = actor_critic2(test_state)
print(f"Final Actions for Firm 1: Price {test_actions1[0].item():.2f}, Production {test_actions1[1].item():.2f}")
print(f"Final Actions for Firm 2: Price {test_actions2[0].item():.2f}, Production {test_actions2[1].item():.2f}")
