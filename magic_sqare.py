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
    final_gen = len(stats['best_fitness_history']) - 1
    stats['final_generation'] = final_gen
    
    
    return best_solution, best_fitness, stats, ga

def plot_fitness_histories(results, n, max_gens):
    """Plot average fitness history across runs for each algorithm"""
    algorithms = list(results.keys())
    colors = ['blue', 'green', 'red']
    
    # Plot 1: Best fitness history
    plt.figure(figsize=(12, 5))
    for i, alg in enumerate(algorithms):
        plt.plot(results[alg]["fitness_history"], label=f"{alg.capitalize()} GA", color=colors[i])
    
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness (lower is better)')
    plt.title(f'Average Best Fitness History Across {len(results[algorithms[0]]["best_fitness"])} Runs (n={n})')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'magic_square_best_fitness_n{n}.png')
    plt.close()
    
    # Plot 2: Average fitness history
    plt.figure(figsize=(12, 5))
    for i, alg in enumerate(algorithms):
        plt.plot(results[alg]["avg_fitness_history"], label=f"{alg.capitalize()} GA", color=colors[i])
    
    plt.xlabel('Generation')
    plt.ylabel('Average Population Fitness (lower is better)')
    plt.title(f'Average Population Fitness History Across {len(results[algorithms[0]]["best_fitness"])} Runs (n={n})')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'magic_square_avg_fitness_n{n}.png')
    plt.close()

    # Plot 3: Combined view with log scale for better visualization
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    for i, alg in enumerate(algorithms):
        plt.plot(results[alg]["fitness_history"], label=f"{alg.capitalize()} Best", color=colors[i])
        plt.plot(results[alg]["avg_fitness_history"], label=f"{alg.capitalize()} Avg", color=colors[i], linestyle='--')
    
    plt.ylabel('Fitness (lower is better)')
    plt.title(f'Fitness History Comparison (n={n})')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    for i, alg in enumerate(algorithms):
        plt.semilogy(results[alg]["fitness_history"], label=f"{alg.capitalize()} Best", color=colors[i])
        plt.semilogy(results[alg]["avg_fitness_history"], label=f"{alg.capitalize()} Avg", color=colors[i], linestyle='--')
    
    plt.xlabel('Generation')
    plt.ylabel('Fitness (log scale)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'magic_square_fitness_comparison_n{n}.png')
    plt.close()


def run_full_experiment():
    """Run experiments for different square sizes and compare algorithms"""
    sizes = [3, 4, 5]  # Try different square sizes
    generations = 1000
    population = 500
    runs = 3  # Number of runs per algorithm
    
    # Store results for all sizes
    all_results = {}
    
    for n in sizes:
        print(f"\n{'=' * 60}")
        print(f"RUNNING EXPERIMENTS FOR {n}x{n} MAGIC SQUARE")
        print(f"{'=' * 60}")
        
        results = compare_algorithms(n=n, generations=generations, 
                                    population_size=population, runs=runs)
        all_results[n] = results
        
        # Create algorithm comparison plot for this size
        plot_comparison(results, n)
    
    # Create a comparison across different square sizes
    plot_size_comparison(all_results, sizes)
    
    print(f"\nAll experiments completed. Results saved to image files.")


def plot_size_comparison(all_results, sizes):
    """Plot comparison of algorithms across different square sizes"""
    algorithms = ["regular", "darwin", "lamarck"]
    colors = ['blue', 'green', 'red']
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Success rates by size
    ax1 = axes[0, 0]
    width = 0.25
    x = np.arange(len(sizes))
    
    for i, alg in enumerate(algorithms):
        success_rates = [all_results[n][alg]["success_rate"] * 100 for n in sizes]
        ax1.bar(x + (i-1)*width, success_rates, width, label=f"{alg.capitalize()}")
    
    ax1.set_xlabel('Square Size (n)')
    ax1.set_ylabel('Success Rate (%)')
    ax1.set_title('Success Rate by Algorithm and Square Size')
    ax1.set_xticks(x)
    ax1.set_xticklabels(sizes)
    ax1.legend()
    ax1.grid(True, axis='y')
    
    # Plot 2: Average generations by size
    ax2 = axes[0, 1]
    
    for i, alg in enumerate(algorithms):
        avg_gens = [sum(all_results[n][alg]["generations"]) / len(all_results[n][alg]["generations"]) 
                   for n in sizes]
        ax2.bar(x + (i-1)*width, avg_gens, width, label=f"{alg.capitalize()}")
    
    ax2.set_xlabel('Square Size (n)')
    ax2.set_ylabel('Average Generations')
    ax2.set_title('Average Generations by Algorithm and Square Size')
    ax2.set_xticks(x)
    ax2.set_xticklabels(sizes)
    ax2.legend()
    ax2.grid(True, axis='y')
    
    # Plot 3: Average evaluations by size
    ax3 = axes[1, 0]
    
    for i, alg in enumerate(algorithms):
        avg_evals = [sum(all_results[n][alg]["evaluations"]) / len(all_results[n][alg]["evaluations"]) 
                    for n in sizes]
        ax3.bar(x + (i-1)*width, avg_evals, width, label=f"{alg.capitalize()}")
    
    ax3.set_xlabel('Square Size (n)')
    ax3.set_ylabel('Average Evaluations')
    ax3.set_title('Average Evaluations by Algorithm and Square Size')
    ax3.set_xticks(x)
    ax3.set_xticklabels(sizes)
    ax3.legend()
    ax3.grid(True, axis='y')
    
    # Plot 4: Average fitness by size
    ax4 = axes[1, 1]
    
    for i, alg in enumerate(algorithms):
        avg_fitness = [sum(all_results[n][alg]["best_fitness"]) / len(all_results[n][alg]["best_fitness"]) 
                      for n in sizes]
        ax4.bar(x + (i-1)*width, avg_fitness, width, label=f"{alg.capitalize()}")
    
    ax4.set_xlabel('Square Size (n)')
    ax4.set_ylabel('Average Best Fitness')
    ax4.set_title('Average Best Fitness by Algorithm and Square Size')
    ax4.set_xticks(x)
    ax4.set_xticklabels(sizes)
    ax4.legend()
    ax4.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig('magic_square_size_comparison.png')
    plt.close()
    
def compare_algorithms(n, generations=500, population_size=100, runs=10):
    """Compare different GA types on the magic square problem"""
    algorithms = ["regular", "darwin", "lamarck"]
    results = {alg: {
        "best_fitness": [], 
        "evaluations": [], 
        "generations": [],
        "success_rate": 0,
        "fitness_history": [],  # List to store fitness history for each run
        "avg_fitness_history": []  # List to store average fitness history for each run
    } for alg in algorithms}
    
    max_gens = 0  # Track the maximum number of generations across all runs
    
    for alg in algorithms:
        print(f"\n{'=' * 50}")
        print(f"Running {alg.capitalize()} GA for n={n}")
        print(f"{'=' * 50}\n")
        
        success_count = 0
        best_solution = None
        best_ga = None
        best_fitness_overall = float('inf')
        
        # Initialize combined history arrays for this algorithm
        fitness_histories = []
        avg_fitness_histories = []
        
        for run in range(runs):
            print(f"\n--- Run {run+1}/{runs} ---")
            solution, fitness, stats, ga = run_experiment(
                n, alg, generations, population_size
            )
            
            # Track metrics
            results[alg]["best_fitness"].append(fitness)
            results[alg]["evaluations"].append(stats["evaluations"])
            results[alg]["generations"].append(stats["final_generation"])
            
            # Store fitness histories for this run
            fitness_histories.append(stats['best_fitness_history'])
            avg_fitness_histories.append(stats['avg_fitness_history'])
            
            # Update max generations if this run went longer
            max_gens = max(max_gens, len(stats['best_fitness_history']))
            
            if fitness == 0:
                success_count += 1
                
            if best_solution is None or fitness < best_fitness_overall:
                best_solution = solution
                best_ga = ga
                best_fitness_overall = fitness
        
        # Calculate success rate
        results[alg]["success_rate"] = success_count / runs
        
        # Normalize and average fitness histories across runs
        # First pad shorter histories with their final values
        padded_best = []
        padded_avg = []
        
        for history in fitness_histories:
            padded = history + [history[-1]] * (max_gens - len(history))
            padded_best.append(padded)
            
        for history in avg_fitness_histories:
            padded = history + [history[-1]] * (max_gens - len(history))
            padded_avg.append(padded)
        
        # Calculate average fitness across runs for each generation
        avg_best_fitness = []
        avg_avg_fitness = []
        
        for gen in range(max_gens):
            gen_best_fitness = sum(hist[gen] for hist in padded_best) / runs
            gen_avg_fitness = sum(hist[gen] for hist in padded_avg) / runs
            avg_best_fitness.append(gen_best_fitness)
            avg_avg_fitness.append(gen_avg_fitness)
        
        results[alg]["fitness_history"] = avg_best_fitness
        results[alg]["avg_fitness_history"] = avg_avg_fitness
        
        # Print best solution for this algorithm
        if best_solution:
            print(f"\nBest solution for {alg.capitalize()} GA:")
            best_ga.print_square(best_solution)
    
    # Print comparison with added generation info
    print("\n" + "=" * 65)
    print("ALGORITHM COMPARISON")
    print("=" * 65)
    
    print(f"{'Algorithm':<10} | {'Success %':<10} | {'Avg Fitness':<12} | {'Avg Gens':<10} | {'Avg Evals':<12}")
    print("-" * 65)
    
    for alg in algorithms:
        avg_fitness = sum(results[alg]["best_fitness"]) / runs
        avg_evals = sum(results[alg]["evaluations"]) / runs
        avg_gens = sum(results[alg]["generations"]) / runs
        success = results[alg]["success_rate"] * 100
        
        print(f"{alg.capitalize():<10} | {success:<10.1f}% | {avg_fitness:<12.2f} | {avg_gens:<10.1f} | {avg_evals:<12.0f}")
    
    # Plot fitness histories
    plot_fitness_histories(results, n, max_gens)
    
    return results

def plot_comparison(results, n):
    """Plot comparison graphs for different algorithms with generation data"""
    algorithms = list(results.keys())
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Bar chart for evaluations and success rate
    x = np.arange(len(algorithms))
    width = 0.2
    
    # Calculate averages
    avg_evals = [sum(results[alg]["evaluations"]) / len(results[alg]["evaluations"]) 
                for alg in algorithms]
    avg_gens = [sum(results[alg]["generations"]) / len(results[alg]["generations"])
               for alg in algorithms]
    success_rates = [results[alg]["success_rate"] * 100 for alg in algorithms]
    
    # Plot 1: Success rates and generations
    ax1 = axes[0]
    bars1 = ax1.bar(x - width, success_rates, width, label='Success Rate (%)')
    ax1.set_ylabel('Success Rate (%)')
    ax1.set_ylim([0, 100])
    
    ax1_twin = ax1.twinx()
    bars2 = ax1_twin.bar(x + width, avg_gens, width, color='green', label='Avg. Generations')
    ax1_twin.set_ylabel('Average Generations')
    
    ax1.set_title(f'Success Rate and Generations (n={n})')
    ax1.set_xticks(x)
    ax1.set_xticklabels([alg.capitalize() for alg in algorithms])
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center')
    
    # Plot 2: Evaluations and fitness
    ax2 = axes[1]
    bars3 = ax2.bar(x - width, avg_evals, width, label='Avg. Evaluations')
    ax2.set_ylabel('Average Evaluations')
    
    ax2_twin = ax2.twinx()
    avg_fitness = [sum(results[alg]["best_fitness"]) / len(results[alg]["best_fitness"]) 
                  for alg in algorithms]
    bars4 = ax2_twin.bar(x + width, avg_fitness, width, color='orange', label='Avg. Fitness')
    ax2_twin.set_ylabel('Average Fitness (lower is better)')
    
    ax2.set_title(f'Evaluations and Fitness (n={n})')
    ax2.set_xticks(x)
    ax2.set_xticklabels([alg.capitalize() for alg in algorithms])
    
    # Combine legends
    lines3, labels3 = ax2.get_legend_handles_labels()
    lines4, labels4 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines3 + lines4, labels3 + labels4, loc='upper center')
    
    # Finalize figure
    fig.tight_layout()
    plt.savefig(f'magic_square_comparison_n{n}.png')
    plt.close()


def run_full_experiment():
        sizes = [4, 5]
        generations = 1000
        population = 1000
        runs = 10  # You can increase this for more robust results

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