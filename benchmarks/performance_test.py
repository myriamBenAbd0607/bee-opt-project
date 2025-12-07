"""
Performance Benchmarking for ABC Algorithm
==========================================
Compare ABC with PSO, GA and other optimization algorithms.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.abc_algorithm import ArtificialBeeColony
from src.benchmark_functions import *

# Try to import other optimization libraries
try:
    from scipy.optimize import differential_evolution
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: SciPy not installed. Some comparisons will be skipped.")

try:
    # Try to install and use PSO and GA from pyswarms and geneticalgorithm
    import pyswarms as ps
    HAS_PSO = True
except ImportError:
    print("Warning: pyswarms not installed. Installing...")
    try:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyswarms"])
        import pyswarms as ps
        HAS_PSO = True
    except:
        HAS_PSO = False
        print("Could not install pyswarms. PSO comparison skipped.")

try:
    from geneticalgorithm import geneticalgorithm as ga
    HAS_GA = True
except ImportError:
    print("Warning: geneticalgorithm not installed. Installing...")
    try:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "geneticalgorithm"])
        from geneticalgorithm import geneticalgorithm as ga
        HAS_GA = True
    except:
        HAS_GA = False
        print("Could not install geneticalgorithm. GA comparison skipped.")


def run_abc_optimization(func, bounds, dim, seed=42):
    """Run ABC algorithm."""
    abc = ArtificialBeeColony(
        objective_func=func,
        bounds=bounds,
        colony_size=min(50, 10 * dim),
        max_iterations=100 * dim,
        seed=seed,
        verbose=False
    )
    
    start_time = time.time()
    best_sol, best_val = abc.optimize()
    elapsed = time.time() - start_time
    
    return best_val, elapsed, best_sol


def run_pso_optimization(func, bounds, dim, seed=42):
    """Run PSO algorithm using pyswarms."""
    if not HAS_PSO:
        return float('inf'), float('inf'), None
    
    np.random.seed(seed)
    
    # Define bounds for PSO
    lower_bounds = np.array([b[0] for b in bounds])
    upper_bounds = np.array([b[1] for b in bounds])
    bounds_tuple = (lower_bounds, upper_bounds)
    
    # PSO options
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
    
    # Create optimizer
    optimizer = ps.single.GlobalBestPSO(
        n_particles=30,
        dimensions=dim,
        options=options,
        bounds=bounds_tuple
    )
    
    # Define objective function for PSO (minimization)
    def pso_objective(x):
        return func(x)
    
    start_time = time.time()
    best_cost, best_pos = optimizer.optimize(pso_objective, iters=100)
    elapsed = time.time() - start_time
    
    return best_cost, elapsed, best_pos


def run_ga_optimization(func, bounds, dim, seed=42):
    """Run Genetic Algorithm using geneticalgorithm."""
    if not HAS_GA:
        return float('inf'), float('inf'), None
    
    np.random.seed(seed)
    
    # Define variable bounds for GA
    varbound = np.array(bounds)
    
    # Define algorithm parameters
    algorithm_param = {
        'max_num_iteration': 100,
        'population_size': 50,
        'mutation_probability': 0.1,
        'elit_ratio': 0.01,
        'crossover_probability': 0.5,
        'parents_portion': 0.3,
        'crossover_type': 'uniform',
        'max_iteration_without_improv': None
    }
    
    # Create model
    model = ga(
        function=func,
        dimension=dim,
        variable_type='real',
        variable_boundaries=varbound,
        algorithm_parameters=algorithm_param
    )
    
    start_time = time.time()
    model.run(no_plot=True)
    elapsed = time.time() - start_time
    
    return model.output_dict['function'], elapsed, model.output_dict['variable']


def run_de_optimization(func, bounds, dim, seed=42):
    """Run Differential Evolution."""
    if not HAS_SCIPY:
        return float('inf'), float('inf'), None
    
    np.random.seed(seed)
    bounds_list = [(b[0], b[1]) for b in bounds]
    
    start_time = time.time()
    result = differential_evolution(
        func, 
        bounds_list, 
        maxiter=100, 
        popsize=15,
        seed=seed
    )
    elapsed = time.time() - start_time
    
    return result.fun, elapsed, result.x


def compare_all_algorithms(func_name="rastrigin", dim=2, n_runs=5):
    """Compare ABC with PSO, GA and other algorithms."""
    
    func = get_function_by_name(func_name)
    bounds = get_function_bounds(func_name, dim)
    
    algorithms = [
        ('ABC', run_abc_optimization),
        ('DE', run_de_optimization),
    ]
    
    if HAS_PSO:
        algorithms.append(('PSO', run_pso_optimization))
    if HAS_GA:
        algorithms.append(('GA', run_ga_optimization))
    
    results = []
    
    print(f"\n{'='*60}")
    print(f"COMPARING ALGORITHMS ON {func_name.upper()} (DIM={dim})")
    print('='*60)
    
    for algo_name, algo_func in algorithms:
        print(f"\n{algo_name}:")
        
        values = []
        times = []
        
        for run in range(n_runs):
            seed = run * 100
            try:
                value, elapsed, _ = algo_func(func, bounds, dim, seed)
                values.append(value)
                times.append(elapsed)
            except Exception as e:
                print(f"  Run {run+1} failed: {str(e)[:50]}...")
                values.append(float('inf'))
                times.append(float('inf'))
        
        # Remove failed runs
        valid_values = [v for v in values if v != float('inf')]
        valid_times = [t for t in times if t != float('inf')]
        
        if valid_values:
            results.append({
                'name': algo_name,
                'mean_value': np.mean(valid_values),
                'std_value': np.std(valid_values),
                'mean_time': np.mean(valid_times),
                'best_value': np.min(valid_values),
                'success_rate': len(valid_values) / n_runs
            })
            
            print(f"  Best value: {np.min(valid_values):.6f}")
            print(f"  Mean ± std: {np.mean(valid_values):.6f} ± {np.std(valid_values):.6f}")
            print(f"  Mean time: {np.mean(valid_times):.3f}s")
            print(f"  Success rate: {len(valid_values)}/{n_runs} ({len(valid_values)/n_runs*100:.1f}%)")
        else:
            print(f"  All runs failed for {algo_name}")
    
    return results


def plot_comprehensive_comparison(results, func_name, dim):
    """Plot comprehensive comparison results."""
    if not results:
        print("No results to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    names = [r['name'] for r in results]
    mean_values = [r['mean_value'] for r in results]
    std_values = [r['std_value'] for r in results]
    mean_times = [r['mean_time'] for r in results]
    best_values = [r['best_value'] for r in results]
    success_rates = [r['success_rate'] for r in results]
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(names)))
    
    # 1. Bar plot of mean values with error bars
    ax1 = axes[0, 0]
    x_pos = np.arange(len(names))
    bars = ax1.bar(x_pos, mean_values, yerr=std_values, 
                  capsize=10, color=colors, alpha=0.7)
    ax1.set_xlabel('Algorithm')
    ax1.set_ylabel('Mean Objective Value')
    ax1.set_title('Solution Quality (lower is better)', fontweight='bold', fontsize=12)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(names, rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, mean_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{value:.3e}', ha='center', va='bottom', fontsize=9)
    
    # 2. Execution times
    ax2 = axes[0, 1]
    bars = ax2.bar(x_pos, mean_times, color=colors, alpha=0.7)
    ax2.set_xlabel('Algorithm')
    ax2.set_ylabel('Mean Execution Time (seconds)')
    ax2.set_title('Computational Cost', fontweight='bold', fontsize=12)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(names, rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Best values comparison
    ax3 = axes[1, 0]
    bars = ax3.bar(x_pos, best_values, color=colors, alpha=0.7)
    ax3.set_xlabel('Algorithm')
    ax3.set_ylabel('Best Objective Value')
    ax3.set_title('Best Found Solution (lower is better)', fontweight='bold', fontsize=12)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(names, rotation=45)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Success rates
    ax4 = axes[1, 1]
    bars = ax4.bar(x_pos, success_rates, color=colors, alpha=0.7)
    ax4.set_xlabel('Algorithm')
    ax4.set_ylabel('Success Rate')
    ax4.set_title('Algorithm Reliability', fontweight='bold', fontsize=12)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(names, rotation=45)
    ax4.set_ylim([0, 1.1])
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add percentage labels
    for bar, rate in zip(bars, success_rates):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{rate*100:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle(f'Algorithm Comparison on {func_name.upper()} Function (Dimension {dim})', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save figure
    os.makedirs('../assets', exist_ok=True)
    plt.savefig(f'../assets/comparison_{func_name}_dim{dim}.png', dpi=150, bbox_inches='tight')
    plt.show()


def convergence_analysis(func_name="sphere", dim=2):
    """Analyze convergence of different algorithms."""
    func = get_function_by_name(func_name)
    bounds = get_function_bounds(func_name, dim)
    
    # We'll track convergence for ABC only for now
    # You can extend this to other algorithms
    
    print(f"\nConvergence Analysis for {func_name} (dim={dim})")
    print("-" * 50)
    
    # Run ABC with history tracking
    abc = ArtificialBeeColony(
        objective_func=func,
        bounds=bounds,
        colony_size=30,
        max_iterations=100,
        seed=42,
        verbose=False
    )
    
    # Monkey patch to track convergence
    convergence_history = []
    original_optimize = abc.optimize
    
    def track_convergence():
        best_values = []
        for i in range(abc.max_iterations):
            # Run one iteration
            # This is a simplified version - you need to adapt based on your ABC implementation
            if i % 10 == 0:
                # Get current best value (this needs to be implemented in your ABC class)
                current_best = 0  # Replace with actual tracking
                best_values.append(current_best)
        return best_values
    
    try:
        convergence = track_convergence()
        
        plt.figure(figsize=(10, 6))
        plt.plot(convergence, 'b-', linewidth=2, label='ABC Convergence')
        plt.xlabel('Iteration (every 10th)')
        plt.ylabel('Best Objective Value')
        plt.title(f'Convergence Profile - {func_name.capitalize()} Function', fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
    except Exception as e:
        print(f"Convergence analysis failed: {e}")


def main():
    """Main benchmarking function."""
    print("="*70)
    print("COMPREHENSIVE OPTIMIZATION ALGORITHM BENCHMARK")
    print("ABC vs PSO vs GA vs DE")
    print("="*70)
    
    all_comparisons = {}
    
    # Test on different functions
    test_functions = ['sphere', 'rastrigin', 'ackley', 'rosenbrock']
    
    for func_name in test_functions:
        print(f"\n{'='*70}")
        print(f"BENCHMARKING: {func_name.upper()} FUNCTION")
        print('='*70)
        
        # Compare algorithms in 2D
        results_2d = compare_all_algorithms(func_name, dim=2, n_runs=3)
        all_comparisons[f"{func_name}_2d"] = results_2d
        
        if results_2d:
            plot_comprehensive_comparison(results_2d, func_name, 2)
        
        # Compare algorithms in 5D (more challenging)
        print(f"\n{'='*70}")
        print(f"BENCHMARKING: {func_name.upper()} FUNCTION (5D)")
        print('='*70)
        
        results_5d = compare_all_algorithms(func_name, dim=5, n_runs=3)
        all_comparisons[f"{func_name}_5d"] = results_5d
        
        if results_5d:
            plot_comprehensive_comparison(results_5d, func_name, 5)
    
    # Print final summary
    print("\n" + "="*80)
    print("FINAL BENCHMARK SUMMARY")
    print("="*80)
    
    summary_table = []
    
    for key, results in all_comparisons.items():
        for algo_result in results:
            summary_table.append({
                'Test Case': key,
                'Algorithm': algo_result['name'],
                'Best Value': algo_result['best_value'],
                'Mean Value': algo_result['mean_value'],
                'Mean Time (s)': algo_result['mean_time'],
                'Success Rate': f"{algo_result['success_rate']*100:.1f}%"
            })
    
    # Create and display summary dataframe
    import pandas as pd
    df_summary = pd.DataFrame(summary_table)
    
    print("\nDetailed Results:")
    print(df_summary.to_string(index=False))
    
    # Save results
    os.makedirs('../assets', exist_ok=True)
    df_summary.to_csv('../assets/final_benchmark_summary.csv', index=False)
    print(f"\nResults saved to: ../assets/final_benchmark_summary.csv")
    
    # Create a ranking
    print("\n" + "="*80)
    print("ALGORITHM RANKING (Based on Best Value)")
    print("="*80)
    
    # Group by algorithm and calculate average rank
    algorithms_ranking = {}
    
    for test_case, results in all_comparisons.items():
        # Sort by best value
        sorted_results = sorted(results, key=lambda x: x['best_value'])
        
        for rank, algo in enumerate(sorted_results, 1):
            algo_name = algo['name']
            if algo_name not in algorithms_ranking:
                algorithms_ranking[algo_name] = []
            algorithms_ranking[algo_name].append(rank)
    
    # Calculate average rank
    avg_ranks = []
    for algo_name, ranks in algorithms_ranking.items():
        avg_rank = np.mean(ranks)
        avg_ranks.append((algo_name, avg_rank, len(ranks)))
    
    # Sort by average rank
    avg_ranks.sort(key=lambda x: x[1])
    
    print(f"\n{'Rank':<5} {'Algorithm':<12} {'Avg Rank':<12} {'Tests':<10}")
    print("-"*45)
    
    for i, (algo_name, avg_rank, num_tests) in enumerate(avg_ranks, 1):
        print(f"{i:<5} {algo_name:<12} {avg_rank:<12.2f} {num_tests:<10}")
    
    return all_comparisons


if __name__ == "__main__":
    # Install missing dependencies automatically
    print("Checking and installing required dependencies...")
    
    # Run benchmarks
    print("\nStarting comprehensive benchmarks...")
    all_results = main()
    
    print("\n" + "="*80)
    print("BENCHMARK COMPLETE!")
    print("="*80)
    print("\nCheck the 'assets' folder for:")
    print("  - Comparison plots (PNG files)")
    print("  - Detailed results (CSV files)")
    print("\nTo run individual comparisons:")
    print("  compare_all_algorithms('sphere', dim=2)")
    print("  compare_all_algorithms('rastrigin', dim=5)")