"""
Simple and clear algorithm comparison
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import time
import matplotlib.pyplot as plt
from src.abc_algorithm import ArtificialBeeColony
from src.benchmark_functions import *

def simple_abc_vs_others():
    """Simple comparison: ABC vs Random Search vs Gradient Descent"""
    
    print("="*60)
    print("SIMPLE ALGORITHM COMPARISON")
    print("="*60)
    
    # Test on sphere function (2D)
    func = sphere
    dim = 2
    bounds = [(-5, 5), (-5, 5)]
    
    results = []
    
    # 1. ABC Algorithm
    print("\n1. Artificial Bee Colony (ABC):")
    abc_values = []
    abc_times = []
    
    for seed in range(3):
        abc = ArtificialBeeColony(
            objective_func=func,
            bounds=bounds,
            colony_size=20,
            max_iterations=50,
            seed=seed,
            verbose=False
        )
        
        start = time.time()
        best_sol, best_val = abc.optimize()
        abc_times.append(time.time() - start)
        abc_values.append(best_val)
        print(f"   Run {seed+1}: {best_val:.10f}")
    
    results.append({
        'name': 'ABC',
        'best': np.min(abc_values),
        'mean': np.mean(abc_values),
        'time': np.mean(abc_times)
    })
    
    # 2. Random Search (baseline)
    print("\n2. Random Search:")
    random_values = []
    random_times = []
    
    for seed in range(3):
        np.random.seed(seed)
        start = time.time()
        
        best_val = float('inf')
        for i in range(1000):  # 1000 random points
            x = np.random.uniform(-5, 5, size=dim)
            val = func(x)
            if val < best_val:
                best_val = val
        
        random_times.append(time.time() - start)
        random_values.append(best_val)
        print(f"   Run {seed+1}: {best_val:.10f}")
    
    results.append({
        'name': 'Random',
        'best': np.min(random_values),
        'mean': np.mean(random_values),
        'time': np.mean(random_times)
    })
    
    # 3. Simple Local Search
    print("\n3. Local Search:")
    local_values = []
    local_times = []
    
    for seed in range(3):
        np.random.seed(seed)
        start = time.time()
        
        # Start from random point
        current = np.random.uniform(-5, 5, size=dim)
        current_val = func(current)
        
        for i in range(100):  # 100 iterations
            # Try random neighbor
            neighbor = current + np.random.normal(0, 0.5, size=dim)
            neighbor = np.clip(neighbor, -5, 5)
            neighbor_val = func(neighbor)
            
            # Keep if better
            if neighbor_val < current_val:
                current = neighbor
                current_val = neighbor_val
        
        local_times.append(time.time() - start)
        local_values.append(current_val)
        print(f"   Run {seed+1}: {current_val:.10f}")
    
    results.append({
        'name': 'Local',
        'best': np.min(local_values),
        'mean': np.mean(local_values),
        'time': np.mean(local_times)
    })
    
    # Display summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Algorithm':<10} {'Best':<15} {'Mean':<15} {'Time (s)':<10}")
    print("-"*60)
    
    for r in results:
        print(f"{r['name']:<10} {r['best']:<15.10f} {r['mean']:<15.10f} {r['time']:<10.3f}")
    
    # Simple plot
    plot_simple_results(results)
    
    return results

def plot_simple_results(results):
    """Plot simple comparison results"""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    names = [r['name'] for r in results]
    best_values = [r['best'] for r in results]
    mean_values = [r['mean'] for r in results]
    times = [r['time'] for r in results]
    
    colors = ['blue', 'green', 'red']
    
    # 1. Best values
    ax1 = axes[0]
    bars1 = ax1.bar(names, best_values, color=colors)
    ax1.set_ylabel('Best Value Found')
    ax1.set_title('Best Solution (lower is better)', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2e}', ha='center', va='bottom')
    
    # 2. Mean values
    ax2 = axes[1]
    bars2 = ax2.bar(names, mean_values, color=colors)
    ax2.set_ylabel('Mean Value')
    ax2.set_title('Average Performance', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2e}', ha='center', va='bottom')
    
    # 3. Execution times
    ax3 = axes[2]
    bars3 = ax3.bar(names, times, color=colors)
    ax3.set_ylabel('Time (seconds)')
    ax3.set_title('Computational Cost', fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}s', ha='center', va='bottom')
    
    plt.suptitle('ABC vs Random Search vs Local Search', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def run_demo():
    """Run a complete demo"""
    print("Testing on Sphere Function (minimum at [0, 0])")
    print("Goal: Find values close to 0")
    print("\n" + "="*60)
    
    results = simple_abc_vs_others()
    
    # Determine winner
    winner = min(results, key=lambda x: x['best'])
    print(f"\n🏆 WINNER: {winner['name']} with value {winner['best']:.2e}")
    
    if winner['name'] == 'ABC':
        print("✅ VOTRE ALGORITHME ABC EST LE MEILLEUR !")
    else:
        print(f"⚠️  {winner['name']} performed better. ABC needs improvement.")
    
    return results

if __name__ == "__main__":
    run_demo()