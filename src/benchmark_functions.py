"""
Benchmark Functions for Optimization Testing
============================================
Common benchmark functions used to evaluate optimization algorithms.
All functions are to be MINIMIZED.
Global minimum is 0 for all functions (at origin, unless specified).
"""

import numpy as np
from typing import Tuple, Dict, Callable, Union


def sphere(x: np.ndarray) -> float:
    """
    Sphere function - convex, unimodal.
    
    Equation: f(x) = Σ_{i=1}^{n} x_i^2
    Domain: x_i ∈ [-5.12, 5.12]
    Global minimum: f(0,...,0) = 0
    """
    return np.sum(x ** 2)


def rosenbrock(x: np.ndarray) -> float:
    """
    Rosenbrock function (Valley/Dejong's function 2) - unimodal but deceptive.
    
    Equation: f(x) = Σ_{i=1}^{n-1} [100(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]
    Domain: x_i ∈ [-2.048, 2.048]
    Global minimum: f(1,...,1) = 0
    """
    return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)


def rastrigin(x: np.ndarray) -> float:
    """
    Rastrigin function - highly multimodal with many local minima.
    
    Equation: f(x) = 10n + Σ_{i=1}^{n} [x_i^2 - 10 cos(2π x_i)]
    Domain: x_i ∈ [-5.12, 5.12]
    Global minimum: f(0,...,0) = 0
    """
    n = len(x)
    return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))


def ackley(x: np.ndarray) -> float:
    """
    Ackley function - multimodal with many shallow local minima.
    
    Equation: f(x) = -20 exp(-0.2√(1/n Σ x_i^2)) 
                    - exp(1/n Σ cos(2π x_i)) 
                    + 20 + e
    Domain: x_i ∈ [-32.768, 32.768]
    Global minimum: f(0,...,0) = 0
    """
    n = len(x)
    sum_sq = np.sum(x**2)
    sum_cos = np.sum(np.cos(2 * np.pi * x))
    
    term1 = -20 * np.exp(-0.2 * np.sqrt(sum_sq / n))
    term2 = -np.exp(sum_cos / n)
    
    return term1 + term2 + 20 + np.e


def griewank(x: np.ndarray) -> float:
    """
    Griewank function - many widespread local minima.
    
    Equation: f(x) = 1 + 1/4000 Σ x_i^2 - Π cos(x_i/√i)
    Domain: x_i ∈ [-600, 600]
    Global minimum: f(0,...,0) = 0
    """
    n = len(x)
    sum_sq = np.sum(x**2)
    prod_cos = np.prod(np.cos(x / np.sqrt(np.arange(1, n + 1))))
    
    return 1 + sum_sq / 4000 - prod_cos


def schwefel(x: np.ndarray) -> float:
    """
    Schwefel function - multimodal with second best minimum far from global optimum.
    
    Equation: f(x) = 418.9829n - Σ x_i sin(√|x_i|)
    Domain: x_i ∈ [-500, 500]
    Global minimum: f(420.9687,...,420.9687) ≈ 0
    """
    n = len(x)
    return 418.9829 * n - np.sum(x * np.sin(np.sqrt(np.abs(x))))


def zakharov(x: np.ndarray) -> float:
    """
    Zakharov function - unimodal with interactions between variables.
    
    Equation: f(x) = Σ x_i^2 + (Σ 0.5 i x_i)^2 + (Σ 0.5 i x_i)^4
    Domain: x_i ∈ [-5, 10]
    Global minimum: f(0,...,0) = 0
    """
    n = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(0.5 * np.arange(1, n + 1) * x)
    
    return sum1 + sum2**2 + sum2**4


def get_function_bounds(func_name: str, dimension: int = 2) -> np.ndarray:
    """
    Return standard bounds for benchmark functions.
    
    Parameters
    ----------
    func_name : str
        Name of the benchmark function.
    dimension : int
        Dimension of the problem.
        
    Returns
    -------
    bounds : np.ndarray
        Array of shape (dimension, 2) with [lower_bound, upper_bound] for each dimension.
    """
    bounds_dict = {
        'sphere': (-5.12, 5.12),
        'rosenbrock': (-2.048, 2.048),
        'rastrigin': (-5.12, 5.12),
        'ackley': (-32.768, 32.768),
        'griewank': (-600, 600),
        'schwefel': (-500, 500),
        'zakharov': (-5, 10)
    }
    
    func_name = func_name.lower()
    if func_name not in bounds_dict:
        available = list(bounds_dict.keys())
        raise ValueError(f"Function '{func_name}' not found. Available: {available}")
    
    lb, ub = bounds_dict[func_name]
    return np.array([[lb, ub]] * dimension)


def get_function_by_name(name: str) -> Callable[[np.ndarray], float]:
    """
    Return function object by name.
    
    Parameters
    ----------
    name : str
        Name of the function.
        
    Returns
    -------
    function : callable
        The benchmark function.
    """
    functions = {
        'sphere': sphere,
        'rosenbrock': rosenbrock,
        'rastrigin': rastrigin,
        'ackley': ackley,
        'griewank': griewank,
        'schwefel': schwefel,
        'zakharov': zakharov
    }
    
    name = name.lower()
    if name not in functions:
        available = list(functions.keys())
        raise ValueError(f"Function '{name}' not found. Available: {available}")
    
    return functions[name]


def get_function_info(name: str) -> Dict[str, Union[str, Tuple[float, float]]]:
    """
    Get comprehensive information about a benchmark function.
    
    Parameters
    ----------
    name : str
        Name of the function.
        
    Returns
    -------
    info : dict
        Dictionary with function information.
    """
    info_dict = {
        'sphere': {
            'description': 'Convex, unimodal, symmetric bowl-shaped function',
            'characteristics': 'Easy, fast convergence',
            'equation': 'f(x) = Σ x_i²',
            'bounds': (-5.12, 5.12)
        },
        'rosenbrock': {
            'description': 'Unimodal but deceptive with narrow curved valley',
            'characteristics': 'Medium difficulty, tests convergence along ridge',
            'equation': 'f(x) = Σ [100(x_{i+1} - x_i²)² + (1 - x_i)²]',
            'bounds': (-2.048, 2.048)
        },
        'rastrigin': {
            'description': 'Highly multimodal with many local minima',
            'characteristics': 'Difficult, tests exploration capability',
            'equation': 'f(x) = 10n + Σ [x_i² - 10 cos(2π x_i)]',
            'bounds': (-5.12, 5.12)
        },
        'ackley': {
            'description': 'Multimodal with many shallow local minima and steep outer walls',
            'characteristics': 'Medium difficulty, tests balance of exploration/exploitation',
            'equation': 'f(x) = -20exp(-0.2√(1/n Σ x_i²)) - exp(1/n Σ cos(2π x_i)) + 20 + e',
            'bounds': (-32.768, 32.768)
        }
    }
    
    name = name.lower()
    if name not in info_dict:
        available = list(info_dict.keys())
        raise ValueError(f"Function '{name}' not found. Available: {available}")
    
    return info_dict[name]


def visualize_function_2d(func: Callable[[np.ndarray], float], 
                         bounds: Tuple[float, float] = (-5, 5),
                         resolution: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a 2D grid for function visualization.
    
    Parameters
    ----------
    func : callable
        Function to visualize (must accept 2D arrays).
    bounds : tuple
        (lower_bound, upper_bound) for both dimensions.
    resolution : int
        Number of points per dimension.
        
    Returns
    -------
    X, Y, Z : np.ndarray
        Meshgrid and function values for 3D plotting.
    """
    x = np.linspace(bounds[0], bounds[1], resolution)
    y = np.linspace(bounds[0], bounds[1], resolution)
    X, Y = np.meshgrid(x, y)
    
    Z = np.zeros_like(X)
    for i in range(resolution):
        for j in range(resolution):
            point = np.array([X[i, j], Y[i, j]])
            Z[i, j] = func(point)
    
    return X, Y, Z