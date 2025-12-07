"""
Unit Tests for ABC Algorithm
============================
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from src.abc_algorithm import ArtificialBeeColony
from src.benchmark_functions import sphere


def test_abc_initialization():
    """Test ABC initialization with valid parameters."""
    bounds = np.array([[-5, 5], [-5, 5]])
    abc = ArtificialBeeColony(sphere, bounds, colony_size=10, max_iterations=50, seed=42)
    
    assert abc.n_dim == 2
    assert abc.colony_size == 10
    assert abc.max_iter == 50
    assert abc.population.shape == (10, 2)
    assert abc.fitness.shape == (10,)
    assert abc.trial_counters.shape == (10,)


def test_bounds_validation():
    """Test that invalid bounds raise an error."""
    bounds = np.array([[5, -5], [5, -5]])  # Lower > Upper
    
    with pytest.raises(ValueError):
        ArtificialBeeColony(sphere, bounds)


def test_fitness_calculation():
    """Test fitness calculation for minimization."""
    abc = ArtificialBeeColony(sphere, np.array([[-1, 1], [-1, 1]]), 
                            colony_size=5, max_iterations=10, verbose=False)
    
    # Test positive objective value
    fitness1 = abc._calculate_fitness(1.0)
    assert fitness1 == 0.5  # 1/(1+1)
    
    # Test zero objective value
    fitness2 = abc._calculate_fitness(0.0)
    assert fitness2 == 1.0  # 1/(1+0)
    
    # Test large objective value
    fitness3 = abc._calculate_fitness(100.0)
    assert abs(fitness3 - 1/101) < 1e-10


def test_candidate_generation():
    """Test candidate solution generation."""
    bounds = np.array([[-5, 5], [-5, 5]])
    abc = ArtificialBeeColony(sphere, bounds, colony_size=10, max_iterations=10, 
                            seed=42, verbose=False)
    
    # Generate candidate from first solution
    candidate = abc._generate_candidate(0)
    
    assert candidate.shape == (2,)
    assert np.all(candidate >= bounds[:, 0])
    assert np.all(candidate <= bounds[:, 1])


def test_optimization_convergence():
    """Test that ABC improves solution over iterations."""
    bounds = np.array([[-5, 5], [-5, 5]])
    abc = ArtificialBeeColony(sphere, bounds, colony_size=20, max_iterations=30, 
                            seed=123, verbose=False)
    
    # Run optimization
    best_solution, best_value = abc.optimize()
    
    # Check that we got a valid solution
    assert best_solution.shape == (2,)
    assert isinstance(best_value, float)
    
    # Check that best value is reasonable (close to 0 for sphere)
    assert best_value >= 0
    assert best_value < 10  # Should be much better than random


def test_reproducibility():
    """Test that same seed produces same results."""
    bounds = np.array([[-5, 5], [-5, 5]])
    
    # Run first time with seed 42
    abc1 = ArtificialBeeColony(sphere, bounds, colony_size=10, max_iterations=20, 
                             seed=42, verbose=False)
    best1, val1 = abc1.optimize()
    
    # Run second time with same seed
    abc2 = ArtificialBeeColony(sphere, bounds, colony_size=10, max_iterations=20, 
                             seed=42, verbose=False)
    best2, val2 = abc2.optimize()
    
    # Results should be identical
    assert np.allclose(best1, best2)
    assert abs(val1 - val2) < 1e-10


def test_statistics_collection():
    """Test that statistics are collected correctly."""
    bounds = np.array([[-5, 5], [-5, 5]])
    abc = ArtificialBeeColony(sphere, bounds, colony_size=15, max_iterations=25, 
                            seed=456, verbose=False)
    
    best_solution, best_value = abc.optimize()
    stats = abc.get_statistics()
    
    # Check that statistics are collected
    assert 'execution_time' in stats
    assert 'function_evaluations' in stats
    assert 'improvements' in stats
    assert 'scout_replacements' in stats
    
    # All should be non-negative
    assert stats['execution_time'] >= 0
    assert stats['function_evaluations'] > 0
    assert stats['improvements'] >= 0
    assert stats['scout_replacements'] >= 0


def test_convergence_history():
    """Test that convergence history is recorded."""
    bounds = np.array([[-5, 5], [-5, 5]])
    abc = ArtificialBeeColony(sphere, bounds, colony_size=10, max_iterations=15, 
                            seed=789, verbose=False)
    
    best_solution, best_value = abc.optimize()
    
    # Check history length
    assert len(abc.history) == 15
    
    # Check history structure
    for record in abc.history:
        assert 'iteration' in record
        assert 'best_objective' in record
        assert 'best_fitness' in record
        assert 'avg_fitness' in record
        assert 'std_fitness' in record


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v"])