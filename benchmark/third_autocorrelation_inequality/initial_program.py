import numpy as np
import scipy.integrate

def calculate_c3_upper_bound(height_sequence):

    N = len(height_sequence)
    delta_x = 1 / (2 * N)

    def f(x):
        if -0.25 <= x <= 0.25:
            index = int((x - (-0.25)) / delta_x)
            if index == N:
                index -= 1
            return height_sequence[index]
        else:
            return 0.0

    integral_f = np.sum(height_sequence) * delta_x
    integral_sq = integral_f**2

    if integral_sq < 1e-18:
        return 0.0

    t_points = np.linspace(-0.5, 0.5, 2 * N + 1)
    
    max_conv_val = 0.0
    for t_val in t_points:

        lower_bound = max(-0.25, t_val - 0.25)
        upper_bound = min(0.25, t_val + 0.25)

        if upper_bound <= lower_bound:
            convolution_val = 0.0
        else:
            def integrand(x):
                return f(x) * f(t_val - x)
            
            convolution_val, _ = scipy.integrate.quad(integrand, lower_bound, upper_bound, limit=100)
        
        if abs(convolution_val) > max_conv_val:
            max_conv_val = abs(convolution_val)

    return max_conv_val / integral_sq

def genetic_algorithm(population_size, num_intervals, generations, mutation_rate, crossover_rate):

    population = np.random.rand(population_size, num_intervals) * 2 - 1

    best_solution = None
    best_fitness = 0.0

    for gen in range(generations):

        fitness_scores = np.array([calculate_c3_upper_bound(individual) for individual in population])

        current_best_idx = np.argmax(fitness_scores)
        if fitness_scores[current_best_idx] > best_fitness:
            best_fitness = fitness_scores[current_best_idx]
            best_solution = population[current_best_idx].copy()
            # print(f"Generation {gen}: New best fitness = {best_fitness}")


        new_population = np.zeros_like(population)
        for i in range(population_size):

            competitors_indices = np.random.choice(population_size, 2, replace=False)
            winner_idx = competitors_indices[np.argmax(fitness_scores[competitors_indices])]
            new_population[i] = population[winner_idx].copy()
            
        for i in range(0, population_size, 2):
            if np.random.rand() < crossover_rate:
                parent1 = new_population[i]
                parent2 = new_population[i+1]
                crossover_point = np.random.randint(1, num_intervals - 1)
                new_population[i] = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
                new_population[i+1] = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))

        for i in range(population_size):
            if np.random.rand() < mutation_rate:
                mutation_point = np.random.randint(num_intervals)
                new_population[i, mutation_point] += np.random.normal(0, 0.1) 

                new_population[i, mutation_point] = np.clip(new_population[i, mutation_point], -2, 2)

        population = new_population
    
    return best_solution

def find_better_c3_upper_bound():

    NUM_INTERVALS = 4
    POPULATION_SIZE = 2
    GENERATIONS = 10
    MUTATION_RATE = 0.1
    CROSSOVER_RATE = 0.8

    height_sequence_3 = genetic_algorithm(POPULATION_SIZE, NUM_INTERVALS, GENERATIONS, MUTATION_RATE, CROSSOVER_RATE)
    
    return height_sequence_3