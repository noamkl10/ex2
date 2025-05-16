import random
import copy
import time
import matplotlib.pyplot as plt
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

class MagicSquareGA:
    def __init__(self, n, population_size=100, elite_ratio=0.1, 
                 tournament_size=3, mutation_rate=0.2, optimization_steps=10):
        self.n = n
        self.population_size = population_size
        self.elite_count = max(1, int(population_size * elite_ratio))
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate
        self.optimization_steps = optimization_steps
        self.magic_sum = n * (n * n + 1) // 2  # Magic constant
        
        # Metrics tracking
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.evaluation_count = 0
        
    def generate_individual(self):
        """Generate a random magic square candidate"""
        numbers = list(range(1, self.n * self.n + 1))
        random.shuffle(numbers)
        return [numbers[i * self.n:(i + 1) * self.n] for i in range(self.n)]
    
    def flatten(self, square):
        """Convert 2D square to 1D list"""
        return [cell for row in square for cell in row]
    
    def unflatten(self, flat_list):
        """Convert 1D list to 2D square"""
        return [flat_list[i * self.n:(i + 1) * self.n] for i in range(self.n)]
    
    def calculate_fitness(self, square):
        self.evaluation_count += 1
        score = 0
        
        # Row, column sums
        row_sums = [sum(row) for row in square]
        col_sums = [sum(square[i][j] for i in range(self.n)) for j in range(self.n)]
        
        # Diagonals
        diag1_sum = sum(square[i][i] for i in range(self.n))
        diag2_sum = sum(square[i][self.n - i - 1] for i in range(self.n))
        
        # Deviation from magic sum
        score += sum(abs(self.magic_sum - s) for s in row_sums + col_sums)
        score += abs(self.magic_sum - diag1_sum)
        score += abs(self.magic_sum - diag2_sum)
        
        # Penalty for duplicates
        flat = self.flatten(square)
        duplicates = len(flat) - len(set(flat))
        score += 10 * duplicates  # Big penalty for each duplicate
        
        return score

    def crossover(self, parent1, parent2):
        flat_p1 = self.flatten(parent1)
        flat_p2 = self.flatten(parent2)
        size = len(flat_p1)

        cx1, cx2 = sorted(random.sample(range(size), 2))
        child = [None] * size

        # Copy middle slice
        child[cx1:cx2] = flat_p1[cx1:cx2]
        used = set(child[cx1:cx2])

        # Fill rest from parent2 in order
        pos = 0
        for val in flat_p2:
            if val not in used:
                while child[pos] is not None:
                    pos += 1
                child[pos] = val
                used.add(val)

        return self.unflatten(child)

    def mutate(self, square):
        """Mutate a solution by swapping random positions"""
        if random.random() >= self.mutation_rate:
            return square
        
        # Make a copy to avoid modifying the original
        mutated = copy.deepcopy(square)
        
        # Choose number of swaps based on square size
        num_swaps = random.randint(1, max(1, self.n // 2))
        
        for _ in range(num_swaps):
            # Choose positions to swap
            i1, j1 = random.randint(0, self.n - 1), random.randint(0, self.n - 1)
            i2, j2 = random.randint(0, self.n - 1), random.randint(0, self.n - 1)
            
            # Avoid swapping the same position
            while i1 == i2 and j1 == j2:
                i2, j2 = random.randint(0, self.n - 1), random.randint(0, self.n - 1)
            
            # Swap values
            mutated[i1][j1], mutated[i2][j2] = mutated[i2][j2], mutated[i1][j1]
        
        return mutated

    def tournament_selection(self, population, fitness_values):
        """Select an individual using tournament selection"""
        selected_indices = random.sample(range(len(population)), self.tournament_size)
        selected_index = min(selected_indices, key=lambda i: fitness_values[i])
        return population[selected_index]
    
    def optimize_individual(self, individual, steps=None):
        if steps is None:
            steps = self.optimization_steps

        current = copy.deepcopy(individual)
        current_fitness = self.calculate_fitness(current)
        best = copy.deepcopy(current)
        best_fitness = current_fitness

        for _ in range(steps * self.n):  # More iterations
            # Random swap
            i1, j1 = random.randint(0, self.n - 1), random.randint(0, self.n - 1)
            i2, j2 = random.randint(0, self.n - 1), random.randint(0, self.n - 1)
            current[i1][j1], current[i2][j2] = current[i2][j2], current[i1][j1]

            new_fitness = self.calculate_fitness(current)
            if new_fitness < best_fitness:
                best = copy.deepcopy(current)
                best_fitness = new_fitness
            else:
                # Undo swap
                current[i1][j1], current[i2][j2] = current[i2][j2], current[i1][j1]

            if best_fitness == 0:
                break

        return best, best_fitness

    def run(self, mode="regular", generations=1000, early_stop=True, 
            min_convergence_gens=50, patience=100):
        """
        Run the genetic algorithm
        
        Parameters:
        - mode: "regular", "darwin" or "lamarck"
        - generations: maximum number of generations
        - early_stop: whether to stop early if solution found
        - min_convergence_gens: minimum generations before checking convergence
        - patience: stop if no improvement for this many generations
        
        Returns:
        - best solution found, fitness, statistics
        """
        # Initialize population
        start_time = time.time()
        population = [self.generate_individual() for _ in range(self.population_size)]
        best_fitness = float('inf')
        best_solution = None
        stagnation_counter = 0
        
        for generation in range(generations):
            # Evaluate population
            fitness_values = []
            improved_population = []
            
            for individual in population:
                original = copy.deepcopy(individual)
                
                if mode == "regular":
                    # Standard GA - no local optimization
                    fitness = self.calculate_fitness(individual)
                    improved_population.append(individual)
                
                elif mode == "darwin":
                    # Darwinian GA - apply optimization but don't modify genotype
                    optimized, fitness = self.optimize_individual(copy.deepcopy(individual))
                    improved_population.append(individual)  # Keep original
                
                elif mode == "lamarck":
                    # Lamarckian GA - apply optimization and modify genotype
                    optimized, fitness = self.optimize_individual(copy.deepcopy(individual))
                    improved_population.append(optimized)  # Use optimized version
                
                else:
                    raise ValueError(f"Unknown mode: {mode}")
                
                fitness_values.append(fitness)
            
            # Update population with possibly improved individuals
            population = improved_population
            
            # Track metrics
            avg_fitness = sum(fitness_values) / len(fitness_values)
            generation_best_fitness = min(fitness_values)
            self.best_fitness_history.append(generation_best_fitness)
            self.avg_fitness_history.append(avg_fitness)
            
            # Print progress
            if generation % 10 == 0 or generation_best_fitness == 0:
                print(f"Generation {generation}: Best = {generation_best_fitness}, Avg = {avg_fitness:.2f}, Evals = {self.evaluation_count}")
            
            # Update best solution
            if generation_best_fitness < best_fitness:
                best_fitness = generation_best_fitness
                best_index = fitness_values.index(best_fitness)
                best_solution = copy.deepcopy(population[best_index])
                stagnation_counter = 0
            else:
                stagnation_counter += 1
            
            # Check for early stopping
            if early_stop and best_fitness == 0:
                print(f"\n✅ Perfect magic square found in generation {generation}!")
                break
                
            # Check for convergence (no improvement for a while)
            if (generation > min_convergence_gens and 
                stagnation_counter > patience):
                print(f"\nStopping early: No improvement for {patience} generations")
                break
            
            # Create next generation
            next_generation = []
            
            # Elitism: Keep best individuals
            sorted_indices = sorted(range(len(fitness_values)), key=lambda i: fitness_values[i])
            elites = [copy.deepcopy(population[i]) for i in sorted_indices[:self.elite_count]]
            next_generation.extend(elites)
            
            # Generate offspring
            while len(next_generation) < self.population_size:
                # Tournament selection
                parent1 = self.tournament_selection(population, fitness_values)
                parent2 = self.tournament_selection(population, fitness_values)
                
                # Apply crossover
                child = self.crossover(parent1, parent2)
                
                # Apply mutation
                child = self.mutate(child)
                
                next_generation.append(child)
            
            population = next_generation
        
        # Get final best solution if we didn't find a perfect one
        if best_fitness > 0 and best_solution is None:
            best_index = fitness_values.index(min(fitness_values))
            best_solution = copy.deepcopy(population[best_index])
            best_fitness = fitness_values[best_index]
        
        duration = time.time() - start_time
        print(f"\nFinished in {duration:.2f} seconds with {self.evaluation_count} evaluations")
        print(f"Best fitness: {best_fitness}")
        
        return best_solution, best_fitness, {
            'best_fitness_history': self.best_fitness_history,
            'avg_fitness_history': self.avg_fitness_history,
            'evaluations': self.evaluation_count,
            'duration': duration
        }
    
    def verify_solution(self, square):
        """Verify if the square is a valid magic square"""
        # Check if it contains all numbers from 1 to n²
        flat = self.flatten(square)
        if set(flat) != set(range(1, self.n * self.n + 1)):
            return False
        
        # Check rows, columns, and diagonals
        row_sums = [sum(row) for row in square]
        col_sums = [sum(square[i][j] for i in range(self.n)) for j in range(self.n)]
        diag1_sum = sum(square[i][i] for i in range(self.n))
        diag2_sum = sum(square[i][self.n - i - 1] for i in range(self.n))
        
        all_sums = row_sums + col_sums + [diag1_sum, diag2_sum]
        return all(s == self.magic_sum for s in all_sums)
    
    def print_square(self, square):
        """Pretty print a magic square"""
        width = len(str(self.n * self.n))
        horizontal_line = "+" + "-" * (self.n * (width + 2) + 1) + "+"
        
        print(horizontal_line)
        for row in square:
            print("| " + " ".join(f"{num:{width}d}" for num in row) + " |")
        print(horizontal_line)
        
        # Print verification
        is_valid = self.verify_solution(square)
        print(f"Magic sum: {self.magic_sum}")
        print(f"Valid magic square: {'✅' if is_valid else '❌'}")


def run_experiment(n, algorithm_type, generations=1000, population_size=100):
    """Run a single experiment and return results"""
    ga = MagicSquareGA(n=n, population_size=population_size, 
                      elite_ratio=0.1, tournament_size=3, 
                      mutation_rate=0.2, optimization_steps=n)
    
    best_solution, best_fitness, stats = ga.run(
        mode=algorithm_type, 
        generations=generations,
        early_stop=True,
        min_convergence_gens=50,
        patience=100
    )
    
    return best_solution, best_fitness, stats, ga


def compare_algorithms(n, generations=500, population_size=100, runs=3):
    """Compare different GA types on the magic square problem"""
    algorithms = ["regular", "darwin", "lamarck"]
    results = {alg: {"best_fitness": [], "evaluations": [], "success_rate": 0} 
               for alg in algorithms}
    
    for alg in algorithms:
        print(f"\n{'=' * 50}")
        print(f"Running {alg.capitalize()} GA for n={n}")
        print(f"{'=' * 50}\n")
        
        success_count = 0
        best_solution = None
        best_ga = None
        
        for run in range(runs):
            print(f"\n--- Run {run+1}/{runs} ---")
            solution, fitness, stats, ga = run_experiment(
                n, alg, generations, population_size
            )
            
            # Track metrics
            results[alg]["best_fitness"].append(fitness)
            results[alg]["evaluations"].append(stats["evaluations"])
            
            if fitness == 0:
                success_count += 1
                
            # Keep track of the best solution
            if best_solution is None or fitness < results[alg]["best_fitness"][-1]:
                best_solution = solution
                best_ga = ga
        
        # Calculate success rate
        results[alg]["success_rate"] = success_count / runs
        
        # Print best solution for this algorithm
        if best_solution:
            print(f"\nBest solution for {alg.capitalize()} GA:")
            best_ga.print_square(best_solution)
    
    # Print comparison
    print("\n" + "=" * 50)
    print("ALGORITHM COMPARISON")
    print("=" * 50)
    
    print(f"{'Algorithm':<10} | {'Success %':<10} | {'Avg Fitness':<12} | {'Avg Evals':<12}")
    print("-" * 50)
    
    for alg in algorithms:
        avg_fitness = sum(results[alg]["best_fitness"]) / runs
        avg_evals = sum(results[alg]["evaluations"]) / runs
        success = results[alg]["success_rate"] * 100
        
        print(f"{alg.capitalize():<10} | {success:<10.1f}% | {avg_fitness:<12.2f} | {avg_evals:<12.0f}")
    
    return results


def plot_comparison(results, n):
    """Plot comparison graphs for different algorithms"""
    algorithms = list(results.keys())
    
    # Create figure with subplots
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Bar chart for evaluations and success rate
    x = np.arange(len(algorithms))
    width = 0.35
    
    # Calculate averages
    avg_evals = [sum(results[alg]["evaluations"]) / len(results[alg]["evaluations"]) 
                for alg in algorithms]
    success_rates = [results[alg]["success_rate"] * 100 for alg in algorithms]
    
    # Plot bars
    bars1 = ax.bar(x - width/2, avg_evals, width, label='Avg. Evaluations')
    ax.set_ylabel('Average Evaluations')
    ax.set_xticks(x)
    ax.set_xticklabels([alg.capitalize() for alg in algorithms])
    
    # Create second y-axis for success rate
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width/2, success_rates, width, color='orange', label='Success Rate (%)')
    ax2.set_ylabel('Success Rate (%)')
    ax2.set_ylim([0, 100])
    
    # Add title and legend
    ax.set_title(f'Comparison of Genetic Algorithms for Magic Square (n={n})')
    fig.tight_layout()
    
    # Add combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper center')
    
    plt.savefig(f'magic_square_comparison_n{n}.png')
    plt.close()

def run_full_experiment():
        sizes = [4, 5]
        generations = 1000
        population = 1000
        runs = 3  # You can increase this for more robust results

        for n in sizes:
            print(f"\n===== Running experiments for n = {n} =====")
            results = compare_algorithms(n=n, generations=generations, population_size=population, runs=runs)
            plot_comparison(results, n)
            print(f"Results for n={n} saved to: magic_square_comparison_n{n}.png")

def main():
    print("Magic Square Genetic Algorithm")
    print("=" * 30)

    mode = input("Choose mode (regular/darwin/lamarck/compare/batch): ").strip().lower()

    if mode == "batch":
        run_full_experiment()
        return

    if mode == "compare":
        # Existing comparison flow
        runs = int(input("Number of runs per algorithm (default 3): ") or 3)
        gen = int(input("Max generations per run (default 500): ") or 500)
        pop = int(input("Population size (default 100): ") or 100)

        results = compare_algorithms(n, generations=gen, population_size=pop, runs=runs)
        plot_comparison(results, n)
        print("\nComparison completed. See graph file for visual comparison.")
        return

    # For single algorithm
    n = int(input("Enter size of magic square (n >= 3): "))
    if n < 3:
        print("Error: Magic square size must be at least 3")
        return

    if mode in ["regular", "darwin", "lamarck"]:
        gen = int(input("Max generations (default 1000): ") or 1000)
        pop = int(input("Population size (default 100): ") or 100)

        ga = MagicSquareGA(n=n, population_size=pop, mutation_rate=0.2, optimization_steps=n)

        print(f"\nRunning {mode.capitalize()} GA for {n}x{n} magic square...")
        best_solution, best_fitness, stats = ga.run(
            mode=mode,
            generations=gen,
            early_stop=True
        )

        print("\nFinal solution:")
        ga.print_square(best_solution)

        # Plot fitness history
        plt.figure(figsize=(10, 5))
        plt.plot(stats['best_fitness_history'], label='Best Fitness')
        plt.plot(stats['avg_fitness_history'], label='Average Fitness')
        plt.xlabel('Generation')
        plt.ylabel('Fitness (lower is better)')
        plt.title(f'{mode.capitalize()} GA for {n}x{n} Magic Square')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'magic_square_{mode}_n{n}.png')
        plt.close()

        print(f"\nFitness history saved to magic_square_{mode}_n{n}.png")
    else:
        print("Invalid mode. Please choose 'regular', 'darwin', 'lamarck', 'compare', or 'batch'.")

    
if __name__ == "__main__":
    main()