import gymnasium as gym
from gymnasium import spaces
import numpy as np

class EconomicEnv(gym.Env):
    def __init__(self):
        super(EconomicEnv, self).__init__()
        # Continuous action space which will be discretized in step
        self.action_space = spaces.Box(low=np.array([0, 0]), high=np.array([100, 100]), dtype=np.float32)
        # Observations are tuples: (Last Production, Last Price, Last Demand)
        self.observation_space = spaces.Box(low=np.array([0, 0, 0]), high=np.array([100, 100, 200]), dtype=np.float32)

        self.production = 0
        self.price = 0
        self.demand = 0

    def calculate_demand(self, price):
        return max(200 - 2 * price, 0)

    def step(self, action):
        production, price = map(int, np.round(action))
        self.production, self.price = production, price
        self.demand = self.calculate_demand(price)

        if production > 0:
            cost = 20 + 1.5 * production
        else:
            cost = 0
        revenue = min(production, self.demand) * price

        profit = revenue - cost
        reward = profit  # Reward is the profit

        state = np.array([self.production, self.price, self.demand])
        done = False
        truncated = False
        info = {'profit': profit, 'revenue': revenue, 'cost': cost}

        return state, reward, done, truncated, info

    def reset(self, **kwargs):
        # Random initialization or other logic
        self.production = np.random.randint(0, 101)
        self.price = np.random.randint(0, 101)
        self.demand = self.calculate_demand(self.price)
        state = np.array([self.production, self.price, self.demand])
        info = {}
        return state, info

from stable_baselines3 import PPO

env = EconomicEnv()

model = PPO("MlpPolicy", env, verbose=1, ent_coef=0.2)
model.learn(total_timesteps=10000)

# Test the trained model
obs, _ = env.reset()
for _ in range(10):
    print(f"Current state: {obs}")
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, _, info = env.step(action)
    print(f"Action: {action}, Profit: {info['profit']}, Revenue: {info['revenue']}, Cost: {info['cost']}")