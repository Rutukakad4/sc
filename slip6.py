'''Write a program to implement basic genetic algorithm with the following parameters.
Population size = 500, Target = ‘AGT’ and initial population is the string
' TTCGATGAGTCCATAATTGGCAGT'. Assume a 1 bit mutation or
0.1%probability'''

import random
import string

# Objective function to calculate fitness
def fitness(individual, target):
    return sum(1 for i in range(len(target)) if individual[i] == target[i])

# Generate initial population
def generate_initial_population(pop_size, target_length):
    population = []
    for _ in range(pop_size):
        individual = ''.join(random.choice(string.ascii_uppercase) for _ in range(target_length))
        population.append(individual)
    return population

# Mutation function
def mutate(individual, mutation_rate):
    mutated_individual = ''
    for bit in individual:
        if random.random() < mutation_rate:
            mutated_individual += random.choice(string.ascii_uppercase.replace(bit, ''))
        else:
            mutated_individual += bit
    return mutated_individual

# Genetic algorithm
def genetic_algorithm(population, target, mutation_rate, num_generations):
    pop_size = len(population)
    target_length = len(target)
    for generation in range(num_generations):
        # Calculate fitness for each individual
        fitness_scores = [fitness(individual, target) for individual in population]

        # Select parents for mating
        selected_parents = []
        for _ in range(pop_size):
            parent1, parent2 = random.choices(population, weights=fitness_scores, k=2)
            selected_parents.append((parent1, parent2))

        # Create next generation by crossover and mutation
        next_generation = []
        for parent1, parent2 in selected_parents:
            crossover_point = random.randint(0, target_length - 1)
            child = parent1[:crossover_point] + parent2[crossover_point:]
            child = mutate(child, mutation_rate)
            next_generation.append(child)

        # Update population
        population = next_generation

        # Output best individual in each generation
        best_individual = max(population, key=lambda x: fitness(x, target))
        best_fitness = fitness(best_individual, target)
        print(f"Generation {generation+1}: Best individual = {best_individual}, Fitness = {best_fitness}/{target_length}")

        # Check for convergence
        if best_fitness == target_length:
            print("Target achieved!")
            break

    return population

# Parameters
population_size = 500
target_string = 'AGT'
initial_individual = 'TTCGATGAGTCCATAATTGGCAGT'
mutation_rate = 0.001
num_generations = 1000

# Generate initial population
population = generate_initial_population(population_size, len(target_string))

# Replace some individuals with the initial individual
population[random.randint(0, population_size-1)] = initial_individual

# Run genetic algorithm
genetic_algorithm(population, target_string, mutation_rate, num_generations)

