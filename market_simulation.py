import random
import numpy as np
import matplotlib.pyplot as plt
import copy

price_genome_length = 10
initial_wealth = 1000
population_size = 100

steps_per_iteration = 252 # average number of trading days per year
num_iterations = 15

MEMORY_SIZE = 10
MAX_STOCKS_PER_TRADE = 10000
MIN_SHARE_PRICE = 0.1

random_agents = 20
trend_following_agents = 20
mean_reversion_agents = 10
genetic_agents = 60

#region Utils
def mutate(agent, mutation_rate=0.1, mutation_amount=0.2):
    for i in range(len(agent.price_genome)):
        if random.random() < mutation_rate:
            agent.price_genome[i] += random.uniform(-mutation_amount, mutation_amount)
            agent.price_genome[i] = max(-1, min(1, agent.price_genome[i]))  # Clamp between -1 and 1
#endregion

class Agent:
    def __init__(self, strategy, initial_wealth):
        self.strategy = strategy
        self.wealth = initial_wealth
        self.shares = 0
        self.memory = []
        self.memory_size = 10
        self.fitness = 0

        # for generic agents only
        self.price_genome = [random.uniform(-1, 1) for _ in range(price_genome_length)]
    
    def make_decision(self, current_price, volume):
        if self.strategy == "random":
            return random.choice(["buy", "sell", "hold"])
        elif self.strategy == "trend_following":
            return self.trend_following_strategy(current_price, volume)
        elif self.strategy == "mean_reversion":
            return self.mean_reversion_strategy(current_price)

        self.fitness = self.shares * current_price

    def trend_following_strategy(self, current_price, volume):
        self.memory.append((current_price, volume))
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)
        
        if len(self.memory) < self.memory_size:
            return "hold"  # Not enough data to make a decision

        price_change = (current_price - self.memory[0][0]) / self.memory[0][0]
        avg_volume = np.mean([v for _, v in self.memory])
        
        # Buy if price is trending up and volume is above average
        if price_change > 0.02 and volume > avg_volume:
            return "buy"
        # Sell if price is trending down and volume is above average
        elif price_change < -0.02 and volume > avg_volume:
            return "sell"
        else:
            return "hold"

    def mean_reversion_strategy(self, current_price):
        self.memory.append(current_price)
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)
        
        if len(self.memory) < self.memory_size:
            return "hold"  # Not enough data to make a decision

        mean_price = np.mean(self.memory)
        
        if current_price < mean_price * 0.95:  # Buy if price is 5% below the mean price
            return "buy"
        elif current_price > mean_price * 1.05:  # Sell if price is 5% above the mean price
            return "sell"
        else:
            return "hold"
    
    def genetic_strategy(self, market_data, _):
        price_decision = sum(g * d for g, d in zip(self.price_genome, market_data))

        if price_decision > 0:
            return "buy"
        elif price_decision < 0:
            return "sell"
        else:
            return "hold"
        
class Market:
    def __init__(self, initial_price, num_agents):
        self.price = initial_price
        #self.agents = [Agent(random.choice(["random", "trend_following", "mean_reversion"]), initial_wealth) for _ in range(num_agents)]
        self.agents = [Agent("random", initial_wealth) for _ in range(random_agents)] + \
                      [Agent("trend_following", initial_wealth) for _ in range(trend_following_agents)] + \
                      [Agent("mean_reversion", initial_wealth) for _ in range(mean_reversion_agents)] + \
                      [Agent("genetic", initial_wealth) for _ in range(genetic_agents)]
        
        self.volume = 0

    def update(self):
        buy_orders = 0
        sell_orders = 0
        for agent in self.agents:
            decision = agent.make_decision(self.price, self.volume)
            if decision == "buy":
                if agent.wealth > self.price:
                    buy_orders += 1
                    agent.wealth -= self.price
                    agent.shares += 1
            elif decision == "sell":
                if agent.shares > 0:
                    sell_orders += 1
                    agent.wealth += self.price
                    agent.shares -= 1
        
        self.volume = buy_orders + sell_orders
        
        # Price update mechanism with volume influence
        price_change = 0.01 * (buy_orders - sell_orders) + 0.005 * (random.random() - 0.45) 
        # why 0.45 above? if it was 0.5, we are saying that there is an equal chance that market goes up and down
        # but I believe a more viable model is that markets tend to go up, so we are slightly skew it to growth  
        
        self.price *= (1 + price_change)
        self.price = max(MIN_SHARE_PRICE, self.price)

    def run_simulation(self, num_steps):
        price_history = []
        volume_history = []
        for _ in range(num_steps):
            self.update()
            price_history.append(self.price)
            volume_history.append(self.volume)
            print(f"Price: {self.price:.2f}, Volume: {self.volume}")
        
        return price_history, volume_history
    
    def run_iterations(self, num_iterations, steps_per_iteration):
        price_history = []
        volume_history = []
        for _ in range(num_iterations):
            prices, volumes = self.run_simulation(steps_per_iteration)
            price_history = price_history + prices
            volume_history = volume_history + volumes

            self.agents.sort(key=lambda x: x.fitness, reverse=True)
            
            new_agents = []
            while len(new_agents) < population_size:
                agent = random.choice(self.agents)
                child1 = copy.deepcopy(agent)
                child2 = copy.deepcopy(agent)
                if agent.strategy == "genetic":
                    mutate(child1)
                    mutate(child2)
                    child1.money = initial_wealth
                    child2.money = initial_wealth
                    child1.stocks = 0
                    child2.stocks = 0
                new_agents.append(child1)
                new_agents.append(child2)

            self.agents = new_agents

        return price_history, volume_history

# Run the simulation
market = Market(initial_price=1000, num_agents=100)
price_history, volume_history = market.run_iterations(num_iterations=10, steps_per_iteration=steps_per_iteration)

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

window_size = 5

smoothed_price = moving_average(price_history, window_size)
smoothed_volume = moving_average(volume_history, window_size)

plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.semilogy(smoothed_price)  # Use semilogy for log scale on y-axis
plt.title("Price History (Log Scale) - Smoothed")
plt.ylabel("Price")
plt.grid(True, which="both", ls="-", alpha=0.2)

plt.subplot(2, 1, 2)
plt.semilogy(smoothed_volume)  # Use semilogy for log scale on y-axis
plt.title("Volume History (Log Scale) - Smoothed")
plt.xlabel("Time Steps")
plt.ylabel("Volume")
plt.grid(True, which="both", ls="-", alpha=0.2)

plt.tight_layout()
plt.show()