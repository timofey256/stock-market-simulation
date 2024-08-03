import random
import numpy as np
import matplotlib.pyplot as plt
from time import sleep

DO_LOG = False
def log(message):
    if DO_LOG:
        print(message)
        sleep(0.05)

MAX_STOCKS_PER_TRADE = 100000000

class TradingAgent:
    def __init__(self, initial_money, initial_stocks, price_genome_length, amount_genome_length):
        self.money = initial_money
        self.stocks = initial_stocks
        self.price_genome = [random.uniform(-1, 1) for _ in range(price_genome_length)]
        self.amount_genome = [random.uniform(0, 1) for _ in range(amount_genome_length+1)]
        self.fitness = 0

    def decide(self, market_data):
        price_decision = sum(g * d for g, d in zip(self.price_genome, market_data))

        extended_by_stocks = market_data + [self.stocks] # take into account how much shares does agent have
        amount_decision = sum(g * d for g, d in zip(self.amount_genome, extended_by_stocks))
        amount_factor = min(self.stocks, amount_decision)/self.stocks # Normalize to 0-1 range
        
        if price_decision > 0:  # Buy
            max_buyable = self.money / market_data[-1]
            max_buyable = min(max_buyable, MAX_STOCKS_PER_TRADE)
            amount = int(max_buyable * amount_factor)
        elif price_decision < 0:  # Sell
            amount = int(-self.stocks * amount_factor)
        else:
            amount = 0
        
        return amount

class Market:
    def __init__(self, initial_price, price_impact_factor):
        self.price = initial_price
        self.price_impact_factor = price_impact_factor
        self.price_history = [initial_price]
        self.volume_history = [0]

    def update(self, net_demand):
        price_change = net_demand * self.price_impact_factor
        self.price *= (1 + price_change)
        self.price = max(0.01, self.price)  # Ensure price doesn't go negative
        self.price_history.append(self.price)
        self.volume_history.append(abs(net_demand))

def run_simulation(agents, market, days, genome_length):
    for _ in range(days):
        net_demand = 0
        market_data = market.price_history[-genome_length:]
        
        if len(market_data) < genome_length:
            market_data = [market.price] * (genome_length - len(market_data)) + market_data
        
        for i, agent in enumerate(agents):
            num_of_stocks = agent.decide(market_data)
            if num_of_stocks > 0:  # Buy
                agent.stocks += num_of_stocks
                agent.money -= num_of_stocks * market.price
                net_demand += num_of_stocks
                log(f"Agent {i} decided to buy {abs(num_of_stocks)} stocks. He is left with ${agent.money}.")

            elif num_of_stocks < 0:  # Sell
                agent.stocks -= num_of_stocks
                agent.money += abs(num_of_stocks) * market.price
                net_demand -= abs(num_of_stocks)
                log(f"Agent {i} decided to sell {abs(num_of_stocks)} stocks. He is left with ${agent.money}.")
            
        market.update(net_demand)
        log(f'===Final price of a simulation day is ${market.price}===') 
    
    # Calculate final fitness (total assets)
    for agent in agents:
        agent.fitness = agent.money + agent.stocks * market.price

def crossover(parent1, parent2):
    child = TradingAgent(parent1.money, parent1.stocks, len(parent1.price_genome), len(parent1.amount_genome))
    
    # Crossover for price genome
    split = random.randint(0, len(parent1.price_genome))
    child.price_genome = parent1.price_genome[:split] + parent2.price_genome[split:]
    
    # Crossover for amount genome
    split = random.randint(0, len(parent1.amount_genome))
    child.amount_genome = parent1.amount_genome[:split] + parent2.amount_genome[split:]
    
    return child

def mutate(agent, mutation_rate=0.01, mutation_amount=0.1):
    for genome in [agent.price_genome, agent.amount_genome]:
        for i in range(len(genome)):
            if random.random() < mutation_rate:
                genome[i] += random.uniform(-mutation_amount, mutation_amount)
                genome[i] = max(-1, min(1, genome[i]))  # Clamp between -1 and 1

def genetic_algorithm(population_size, price_genome_length, amount_genome_length, initial_money, initial_stocks, initial_price, price_impact_factor, days, iterations, survival_rate):
    market = Market(initial_price, price_impact_factor)
    agents = [TradingAgent(initial_money, initial_stocks, price_genome_length, amount_genome_length) for _ in range(population_size)]
    
    for iteration in range(iterations):
        run_simulation(agents, market, days, max(price_genome_length, amount_genome_length))
        agents.sort(key=lambda x: x.fitness, reverse=True)
        
        survivors = int(population_size * survival_rate)
        agents = agents[:survivors]
        
        while len(agents) < population_size:
            parent1, parent2 = random.sample(agents, 2)
            child = crossover(parent1, parent2)
            mutate(child)
            child.money = initial_money
            child.stocks = initial_stocks
            agents.append(child)
        
        print(f"Iteration {iteration + 1}: Best fitness = {agents[0].fitness:.2f}, Final price = {market.price:.2f}")
    
    return agents[0], market

# Run the genetic algorithm
population_size = 100
price_genome_length = 10
amount_genome_length = 5
initial_money = 10000
initial_stocks = 100
initial_price = 100
price_impact_factor = 0.01
days = 252  # Approximately one trading year
iterations = 50
survival_rate = 0.2

best_agent, market = genetic_algorithm(population_size, price_genome_length, amount_genome_length, initial_money, initial_stocks, initial_price, price_impact_factor, days, iterations, survival_rate)

print(f"\nBest agent's price genome: {best_agent.price_genome}")
print(f"Best agent's amount genome: {best_agent.amount_genome}")
print(f"Best agent's final fitness: {best_agent.fitness:.2f}")

# Plot the price history
plt.figure(figsize=(12, 6))
plt.plot(market.price_history)
plt.title('Stock Price History')
plt.xlabel('Day')
plt.ylabel('Price')
plt.show()

# Plot the volume history
plt.figure(figsize=(12, 6))
plt.plot(market.volume_history)
plt.title('Trading Volume History')
plt.xlabel('Day')
plt.ylabel('Volume')
plt.show()