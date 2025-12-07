"""
Simple Demonstration of ABC Algorithm
=====================================
Basic examples showing how to use the ABC algorithm on benchmark functions.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from src.abc_algorithm import ArtificialBeeColony
from src.benchmark_functions import (
    sphere, rosenbrock, rastrigin, ackley,
    get_function_bounds, get_function_info
)


def plot_convergence(history, title="ABC Convergence", save_path=None):
    """Plot convergence curve from optimization history."""
    iterations = [h['iteration'] for h in history]
    best_objectives = [h['best_objective'] for h in history]
    avg_fitness = [h['avg_fitness'] for h in history]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Objective value convergence
    ax1.plot(iterations, best_objectives, 'b-', linewidth=2, label='Best Objective')
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Objective Value', fontsize=12)
    ax1.set_title(f'Objective Convergence: {title}', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_yscale('log')
    
    # Plot 2: Fitness statistics
    ax2.plot(iterations, avg_fitness, 'g-', linewidth=2, label='Average Fitness')
    ax2.fill_between(iterations, 
                     np.array(avg_fitness) - np.array([h['std_fitness'] for h in history]),
                     np.array(avg_fitness) + np.array([h['std_fitness'] for h in history]),
                     alpha=0.2, color='green', label='±1 std')
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Fitness', fontsize=12)
    ax2.set_title('Population Fitness Statistics', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()


def demo_sphere_2d():
    """Demo ABC on 2D Sphere function (easy)."""
    print("\n" + "="*60)
    print("DEMO 1: Sphere Function (2D)")
    print("="*60)
    
    # Get function bounds
    bounds = get_function_bounds('sphere', 2)
    info = get_function_info('sphere')
    
    print(f"Function: {info['description']}")
    print(f"Equation: {info['equation']}")
    print(f"Bounds: {bounds[0]}")
    print(f"Expected minimum: 0 at (0, 0)")
    
    # Create and run ABC optimizer
    abc = ArtificialBeeColony(
        objective_func=sphere,
        bounds=bounds,
        colony_size=20,
        max_iterations=50,
        seed=42,
        verbose=True
    )
    
    best_solution, best_value = abc.optimize()
    stats = abc.get_statistics()
    
    print(f"\nResults:")
    print(f"  Best solution: {best_solution}")
    print(f"  Best value: {best_value:.6e}")
    print(f"  Error from optimum: {best_value:.6e}")
    print(f"  Function evaluations: {stats['function_evaluations']}")
    print(f"  Execution time: {stats['execution_time']:.2f}s")
    
    # Plot convergence
    plot_convergence(abc.history, title="Sphere Function (2D)")
    
    return abc


def demo_rastrigin_2d():
    """Demo ABC on 2D Rastrigin function (difficult, multimodal)."""
    print("\n" + "="*60)
    print("DEMO 2: Rastrigin Function (2D)")
    print("="*60)
    
    bounds = get_function_bounds('rastrigin', 2)
    info = get_function_info('rastrigin')
    
    print(f"Function: {info['description']}")
    print(f"Characteristics: {info['characteristics']}")
    print(f"Equation: {info['equation']}")
    print(f"Bounds: {bounds[0]}")
    
    # Run with more iterations for difficult function
    abc = ArtificialBeeColony(
        objective_func=rastrigin,
        bounds=bounds,
        colony_size=30,
        max_iterations=200,
        limit=100,
        seed=123,
        verbose=True
    )
    
    best_solution, best_value = abc.optimize()
    
    print(f"\nResults:")
    print(f"  Best solution: {best_solution}")
    print(f"  Best value: {best_value:.6f}")
    print(f"  Scout replacements: {abc.stats['scout_replacements']}")
    
    # Plot convergence
    plot_convergence(abc.history, title="Rastrigin Function (2D)")
    
    return abc


def demo_comparison():
    """Compare ABC performance on different functions."""
    print("\n" + "="*60)
    print("DEMO 3: Function Comparison")
    print("="*60)
    
    functions = [
        ('Sphere', sphere),
        ('Rosenbrock', rosenbrock),
        ('Rastrigin', rastrigin),
        ('Ackley', ackley)
    ]
    
    results = []
    
    for name, func in functions:
        print(f"\nOptimizing {name} function...")
        
        bounds = get_function_bounds(name.lower(), 2)
        
        abc = ArtificialBeeColony(
            objective_func=func,
            bounds=bounds,
            colony_size=25,
            max_iterations=100,
            seed=42,
            verbose=False
        )
        
        best_sol, best_val = abc.optimize()
        stats = abc.get_statistics()
        
        results.append({
            'function': name,
            'best_value': best_val,
            'evaluations': stats['function_evaluations'],
            'time': stats['execution_time'],
            'scouts': stats['scout_replacements']
        })
        
        print(f"  Best value: {best_val:.6f}")
        print(f"  Time: {stats['execution_time']:.2f}s")
    
    # Display comparison table
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    print(f"{'Function':<12} {'Best Value':<12} {'Evaluations':<12} {'Time (s)':<10} {'Scouts':<8}")
    print("-"*60)
    
    for res in results:
        print(f"{res['function']:<12} {res['best_value']:<12.6f} "
              f"{res['evaluations']:<12} {res['time']:<10.2f} {res['scouts']:<8}")
    
    return results


def main():
    """Main function to run all demos."""
    print("="*60)
    print("ARTIFICIAL BEE COLONY ALGORITHM - DEMONSTRATION")
    print("="*60)
    
    # Run individual demos
    abc_sphere = demo_sphere_2d()
    abc_rastrigin = demo_rastrigin_2d()
    
    # Run comparison
    results = demo_comparison()
    
    print("\n" + "="*60)
    print("ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    return abc_sphere, abc_rastrigin, results


if __name__ == "__main__":
    # Run the main demonstration
    sphere_result, rastrigin_result, comparison = main()
    
    # Save convergence plots
    import os
    if not os.path.exists('../assets'):
        os.makedirs('../assets')
    
    print("\nGenerating output files...")
    # You could save results to files here
    print("Done!")