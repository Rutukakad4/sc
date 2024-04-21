'''Write a program to implement basic genetic algorithm. Generate a population of size
100. Each element in the population is a 20 bit string with objective function of -sum(x)
and number of iterations > 100 '''

import random

# Objective function
def objective_function(individual):
    return -sum(individual)

# Generate initial population
def generate_population(pop_size, chrom_length):
    population = []
    for _ in range(pop_size):
        individual = [random.randint(0, 1) for _ in range(chrom_length)]
        population.append(individual)
    return population

# Tournament selection
def tournament_selection(population, tournament_size):
    selected_parents = []
    for _ in range(len(population)):
        tournament = random.sample(population, tournament_size)
        winner = max(tournament, key=objective_function)
        selected_parents.append(winner)
    return selected_parents

# Crossover (single point crossover)
def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

# Mutation (bit flip mutation)
def mutation(individual, mutation_rate):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = 1 - individual[i]
    return individual

# Genetic algorithm
def genetic_algorithm(population, pop_size, chrom_length, mutation_rate, num_generations):
    for generation in range(num_generations):
        # Select parents
        selected_parents = tournament_selection(population, tournament_size=2)

        # Create next generation
        next_generation = []
        while len(next_generation) < pop_size:
            parent1, parent2 = random.sample(selected_parents, 2)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutation(child1, mutation_rate)
            child2 = mutation(child2, mutation_rate)
            next_generation.extend([child1, child2])

        # Update population
        population = next_generation

        # Output best individual in each generation
        best_individual = max(population, key=objective_function)
        print(f"Generation {generation+1}: Best individual = {best_individual}, Objective function value = {objective_function(best_individual)}")

    return population

# Parameters
pop_size = 100
chrom_length = 20
mutation_rate = 0.01
num_generations = 100

# Generate initial population
population = generate_population(pop_size, chrom_length)

# Run genetic algorithm
final_population = genetic_algorithm(population, pop_size, chrom_length, mutation_rate, num_generations)

