import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

class EconomicEnv(gym.Env):
    def __init__(self):
        super(EconomicEnv, self).__init__()
        self.action_space = spaces.Box(low=np.array([0, 0]), high=np.array([100, 100]), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([0, 0, 0]), high=np.array([100, 100, 200]), dtype=np.float32)
        self.current_step = 0

        self.production = 0
        self.price = 0
        self.demand = 0

    def calculate_demand(self, price):
        return max(200 - 2 * price, 0)

    def step(self, action):
        # Discretize the continuous action by rounding both production and price
        production, price = map(int, np.round(action))
        self.production, self.price = production, price
        self.demand = self.calculate_demand(price)

        # Calculate costs and revenue
        if production > 0:
            cost = 20 + 1.5 * production
        else:
            cost = 0
        revenue = min(production, self.demand) * price

        # Compute profit
        profit = revenue - cost
        reward = profit

        # Prepare the next state
        state = np.array([self.production, self.price, self.demand])
        terminated = self.current_step >= 500
        truncated = False  # In this example, we don't use truncation

        self.current_step += 1

        return state, reward, terminated, truncated, {'profit': profit, 'revenue': revenue, 'cost': cost}

    def reset(self, **kwargs):
        self.production = 0
        self.price = 0
        self.demand = self.calculate_demand(self.price)
        self.current_step = 0
        state = np.array([self.production, self.price, self.demand])
        info = {}
        return state, info

# Wrap the environment with DummyVecEnv for compatibility with Stable Baselines3
env = DummyVecEnv([lambda: EconomicEnv()])

model = PPO("MlpPolicy", env, verbose=1, ent_coef=0.01)

# Continuous training loop
num_episodes = 500
for episode in range(num_episodes):
    obs, _ = env.reset()  # Unpack the returned tuple
    done = [False]
    while not done[0]:  # Vectorized environments return lists/arrays for done
        action, _states = model.predict(obs, deterministic=False)  # Ensure stochastic policy for exploration
        obs, rewards, dones, infos = env.step(action)
        model.learn(total_timesteps=1)  # Update the model based on the most recent step
        done = dones

    if episode % 10 == 0:
        print(f"Episode {episode}: Last state: {obs}, Last action: {action}, Last profit: {infos[0]['profit']}")
