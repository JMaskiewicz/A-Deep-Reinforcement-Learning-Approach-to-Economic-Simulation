import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)

# Actor-Critic for Firm 1 (Replacing Actor with ActorCritic)
class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(4, 32)
        self.fc_actor = nn.Linear(32, 2)
        self.fc_critic = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.bankrupt = False
        self.bankrupt_episodes = 0
        self.sigma = 10  # Initial sigma value for exploration noise

    def forward(self, state):
        x = self.relu(self.fc1(state))
        action = 100 * self.sigmoid(self.fc_actor(x))
        value = self.fc_critic(x)
        return action, value

    def reset_actor_weights(self):
        # Reinitialize actor weights
        self.fc_actor.reset_parameters()
        self.sigma = 10  # Reset sigma to its initial value

    def get_noisy_action(self, state):
        if self.bankrupt:
            return torch.tensor([[0.0, 0.0]], dtype=torch.float32), torch.tensor([0.0], dtype=torch.float32)
        action, value = self.forward(state)
        noisy_action = torch.clamp(action + self.sigma * torch.randn_like(action), 0, 100)
        return noisy_action.unsqueeze(0), value


# Economic Environment
class EconomicEnv:
    def __init__(self):
        self.c = 1

    def demand(self, total_price):
        return torch.clamp(torch.exp(-total_price / 20) * 100, min=0)

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
        cost2 = 30 * production2 + 100
        profit1 = revenue1 - cost1
        profit2 = revenue2 - cost2
        return profit1 / 10, profit2 / 10


# Initialize actor-critics and optimizers
actor_critic1 = ActorCritic()
actor_critic2 = ActorCritic()
opt_actor_critic1 = optim.Adam(actor_critic1.parameters(), lr=0.00025)
opt_actor_critic2 = optim.Adam(actor_critic2.parameters(), lr=0.00025)

env = EconomicEnv()

num_episodes = 2000
gamma = 0.5  # Discount factor for future rewards

# Initialize previous actions
prev_actions1 = torch.tensor([0.0, 0.0], dtype=torch.float32, requires_grad=True)
prev_actions2 = torch.tensor([0.0, 0.0], dtype=torch.float32, requires_grad=True)

# Track prices, productions, and profits over episodes
prices1 = []
prices2 = []
productions1 = []
productions2 = []
profits1 = []
profits2 = []

# Tracking for bankruptcy
consecutive_negatives1 = 0
consecutive_negatives2 = 0
bankruptcy_threshold = 100
return_from_bankruptcy_threshold = 250

for episode in range(num_episodes):
    state = torch.cat([prev_actions1, prev_actions2]).unsqueeze(0)

    # Get noisy actions and state values from actor_critic1
    noisy_actions1, state_value1 = actor_critic1.get_noisy_action(state)

    # Get noisy actions and state values from actor_critic2
    noisy_actions2, state_value2 = actor_critic2.get_noisy_action(state)

    # Combine actions into a single tensor
    actions = torch.cat([noisy_actions1, noisy_actions2], dim=0)

    # Get profits for both firms
    profit1, profit2 = env.step(actions)

    # Track prices, productions, and profits
    prices1.append(noisy_actions1[0, 0].item())
    prices2.append(noisy_actions2[0, 0].item())
    productions1.append(noisy_actions1[0, 1].item())
    productions2.append(noisy_actions2[0, 1].item())
    profits1.append(profit1.item())
    profits2.append(profit2.item())

    # Ensure profits require gradients by creating them directly from operations involving requires_grad=True tensors
    profit1 = torch.tensor(profit1, requires_grad=True)
    profit2 = torch.tensor(profit2, requires_grad=True)

    # Prepare for the next state's value estimate
    next_state = state  # In a real case, you'd get this from the environment
    _, next_value1 = actor_critic1(next_state)  # Get next state value prediction
    _, next_value2 = actor_critic2(next_state)  # Get next state value prediction

    # Calculate TD targets and errors
    td_target1 = profit1 + gamma * next_value1.detach()  # Detach to prevent connection to next graph
    td_error1 = td_target1 - state_value1

    td_target2 = profit2 + gamma * next_value2.detach()  # Detach to prevent connection to next graph
    td_error2 = td_target2 - state_value2

    # Backward and optimize for actor_critic1
    opt_actor_critic1.zero_grad()
    actor_loss1 = (-state_value1 * noisy_actions1).mean()  # Policy gradient part
    total_loss1 = td_error1.pow(2).mean() + actor_loss1  # Total loss
    total_loss1.backward(retain_graph=True)  # Retain graph for subsequent backward pass
    opt_actor_critic1.step()

    # Backward and optimize for actor_critic2
    opt_actor_critic2.zero_grad()
    actor_loss2 = (-state_value2 * noisy_actions2).mean()  # Policy gradient part
    total_loss2 = td_error2.pow(2).mean() + actor_loss2  # Total loss
    total_loss2.backward()  # No need for retain_graph=True because this is the only backward pass
    opt_actor_critic2.step()

    # Decay exploration noise
    actor_critic1.sigma = actor_critic1.sigma * 0.999
    actor_critic2.sigma = actor_critic2.sigma * 0.999

    # Update bankruptcy status based on profit
    consecutive_negatives1 = 0 if profit1.item() >= 0 else consecutive_negatives1 + 1
    consecutive_negatives2 = 0 if profit2.item() >= 0 else consecutive_negatives2 + 1

    if consecutive_negatives1 >= bankruptcy_threshold:
        actor_critic1.bankrupt = True
        actor_critic1.bankrupt_episodes += 1
    else:
        actor_critic1.bankrupt_episodes = 0

    if consecutive_negatives2 >= bankruptcy_threshold:
        actor_critic2.bankrupt = True
        actor_critic2.bankrupt_episodes += 1
    else:
        actor_critic2.bankrupt_episodes = 0

    # Reset bankruptcy status after 250 episodes of bankruptcy
    if actor_critic1.bankrupt_episodes >= return_from_bankruptcy_threshold:
        print(f'Firm 1 returns from bankruptcy at episode {episode}')
        actor_critic1.bankrupt = False
        actor_critic1.bankrupt_episodes = 0
        actor_critic1.reset_actor_weights()

    if actor_critic2.bankrupt_episodes >= return_from_bankruptcy_threshold:
        print(f'Firm 2 returns from bankruptcy at episode {episode}')
        actor_critic2.bankrupt = False
        actor_critic2.bankrupt_episodes = 0
        actor_critic2.reset_actor_weights()

    # Optional: Print episode results
    if episode % 10 == 0:  # Print every 10 episodes
        print(f"Episode {episode}:\nActions 1 {noisy_actions1.detach().numpy()}, Profit 1 {profit1.item():.2f}")
        print(f"Actions 2 {noisy_actions2.detach().numpy()}, Profit 2 {profit2.item():.2f}")

# Plot profits
plt.figure(figsize=(12, 6))
plt.plot(profits1, label='Firm 1 Profit')
plt.plot(profits2, label='Firm 2 Profit')
plt.xlabel('Episode')
plt.ylabel('Profit')
plt.legend()
plt.title('Profit over Episodes')
plt.savefig(r'D:\studia\WNE\2023_2024\symulacje\zdj\oligopol3\profits.png')

# Plot prices
plt.figure(figsize=(12, 6))
plt.plot(prices1, label='Firm 1 Price')
plt.plot(prices2, label='Firm 2 Price')
plt.xlabel('Episode')
plt.ylabel('Price')
plt.legend()
plt.title('Price over Episodes')
plt.savefig(r'D:\studia\WNE\2023_2024\symulacje\zdj\oligopol3\prices.png')

# Plot productions
plt.figure(figsize=(12, 6))
plt.plot(productions1, label='Firm 1 Production')
plt.plot(productions2, label='Firm 2 Production')
plt.xlabel('Episode')
plt.ylabel('Production')
plt.legend()
plt.title('Production over Episodes')
plt.savefig(r'D:\studia\WNE\2023_2024\symulacje\zdj\oligopol3\productions.png')
