"""
Artificial Bee Colony (ABC) Algorithm
======================================
A Python implementation of the Artificial Bee Colony optimization algorithm,
inspired by the intelligent foraging behavior of honey bee swarms.

Reference: Karaboga, D. (2005). "An idea based on honey bee swarm for 
numerical optimization." Technical Report TR06, Erciyes University.

Author: [Your Name]
Project: BEE-OPT
"""

import numpy as np
from typing import Callable, Tuple, List, Optional, Dict, Any
import time
from tqdm import tqdm


class ArtificialBeeColony:
    """
    Main class implementing the Artificial Bee Colony algorithm for numerical optimization.

    The algorithm simulates three types of bees:
    1. Employed Bees: Exploit known food sources (solutions).
    2. Onlooker Bees: Select promising sources based on fitness.
    3. Scout Bees: Discover new random sources when old ones are exhausted.
    """

    def __init__(self,
                 objective_func: Callable[[np.ndarray], float],
                 bounds: np.ndarray,
                 colony_size: int = 50,
                 max_iterations: int = 1000,
                 limit: Optional[int] = None,
                 seed: Optional[int] = None,
                 verbose: bool = True):
        """
        Initialize the ABC optimizer.

        Parameters
        ----------
        objective_func : callable
            The function to minimize. Signature: f(x) -> float, where x is a 1D array.
        bounds : np.ndarray of shape (n_dim, 2)
            Lower and upper bounds for each dimension: [[lb1, ub1], [lb2, ub2], ...].
        colony_size : int, default=50
            Number of food sources (solutions) in the population (SN).
        max_iterations : int, default=1000
            Maximum number of iterations (cycles).
        limit : int, optional
            Maximum number of trials before abandoning a food source.
            If None, set to colony_size * n_dim (common heuristic).
        seed : int, optional
            Random seed for reproducibility.
        verbose : bool, default=True
            Whether to print progress information.
        """
        self.func = objective_func
        self.bounds = np.array(bounds, dtype=np.float64)
        self.n_dim = len(bounds)
        self.colony_size = colony_size
        self.max_iter = max_iterations
        self.limit = limit if limit is not None else colony_size * self.n_dim
        self.verbose = verbose

        # Validate bounds
        if not np.all(self.bounds[:, 0] < self.bounds[:, 1]):
            raise ValueError("Lower bounds must be less than upper bounds")

        # Random state for reproducibility
        self.rng = np.random.default_rng(seed)

        # Initialize population and tracking variables
        self.population = None          # Food sources (solutions)
        self.fitness = None             # Fitness values (higher is better)
        self.trial_counters = None      # Counters for failures per solution
        self.objective_values = None    # Actual objective function values
        self.global_best_solution = None
        self.global_best_fitness = -np.inf
        self.global_best_objective = np.inf
        self.history = []               # To track best fitness per iteration
        self.execution_time = 0.0

        # Statistics
        self.stats = {
            'function_evaluations': 0,
            'improvements': 0,
            'scout_replacements': 0,
            'phase_times': {'employed': 0.0, 'onlooker': 0.0, 'scout': 0.0}
        }

        self._initialize_population()

    def _initialize_population(self):
        """Randomly initialize the population within bounds."""
        lb = self.bounds[:, 0]
        ub = self.bounds[:, 1]

        # Equation: x_ij = lb_j + rand(0,1) * (ub_j - lb_j)
        self.population = self.rng.uniform(lb, ub, size=(self.colony_size, self.n_dim))
        self.objective_values = np.zeros(self.colony_size)
        self.fitness = np.zeros(self.colony_size)
        
        # Evaluate initial population
        for i in range(self.colony_size):
            self.objective_values[i] = self.func(self.population[i])
            self.fitness[i] = self._calculate_fitness(self.objective_values[i])
            self.stats['function_evaluations'] += 1

        # Initialize trial counters
        self.trial_counters = np.zeros(self.colony_size, dtype=int)

        # Set initial global best
        best_idx = np.argmax(self.fitness)
        self.global_best_solution = self.population[best_idx].copy()
        self.global_best_fitness = self.fitness[best_idx]
        self.global_best_objective = self.objective_values[best_idx]

        if self.verbose:
            print(f"ABC Initialized: D={self.n_dim}, SN={self.colony_size}, limit={self.limit}")

    def _calculate_fitness(self, objective_value: float) -> float:
        """
        Calculate fitness value from objective value.
        For minimization problems: fitness = 1 / (1 + f(x))
        For maximization problems: fitness = f(x) directly.

        Here we assume minimization, so higher fitness means lower objective value.
        """
        if objective_value >= 0:
            return 1.0 / (1.0 + objective_value)
        else:
            return 1.0 + abs(objective_value)

    def _generate_candidate(self, idx: int) -> np.ndarray:
        """
        Generate a candidate solution from food source idx (Employed/Onlooker bee phase).

        Equation: v_ij = x_ij + φ_ij * (x_ij - x_kj)
        where:
            - i: current solution index
            - k: randomly chosen different solution
            - j: randomly chosen dimension
            - φ: random number in [-1, 1]

        Returns
        -------
        candidate : np.ndarray
            New candidate solution.
        """
        # Select a random different solution
        k = idx
        while k == idx:
            k = self.rng.integers(0, self.colony_size)

        # Select a random dimension
        j = self.rng.integers(0, self.n_dim)

        # Generate φ in [-1, 1]
        phi = self.rng.uniform(-1, 1)

        # Create candidate by copying current solution
        candidate = self.population[idx].copy()

        # Apply perturbation: v_ij = x_ij + φ * (x_ij - x_kj)
        candidate[j] = self.population[idx, j] + phi * (self.population[idx, j] - self.population[k, j])

        # Apply bounds (clip to search space)
        lb, ub = self.bounds[j]
        candidate[j] = np.clip(candidate[j], lb, ub)

        return candidate

    def _employed_bees_phase(self):
        """Employed bees phase: each bee tries to improve its assigned food source."""
        start_time = time.time()
        
        for i in range(self.colony_size):
            # Generate candidate from solution i
            candidate = self._generate_candidate(i)
            candidate_objective = self.func(candidate)
            candidate_fitness = self._calculate_fitness(candidate_objective)
            self.stats['function_evaluations'] += 1

            # GREEDY SELECTION
            if candidate_fitness > self.fitness[i]:
                # Accept improvement
                self.population[i] = candidate
                self.fitness[i] = candidate_fitness
                self.objective_values[i] = candidate_objective
                self.trial_counters[i] = 0  # Reset failure counter
                self.stats['improvements'] += 1
            else:
                # Reject candidate
                self.trial_counters[i] += 1  # Increment failure counter
        
        self.stats['phase_times']['employed'] += time.time() - start_time

    def _onlooker_bees_phase(self):
        """Onlooker bees phase: select solutions probabilistically based on fitness."""
        start_time = time.time()
        
        # Calculate selection probabilities (roulette wheel)
        # Equation: p_i = fitness_i / sum(fitness)
        fitness_sum = np.sum(self.fitness)
        if fitness_sum > 0:
            probs = self.fitness / fitness_sum
        else:
            probs = np.ones(self.colony_size) / self.colony_size

        for _ in range(self.colony_size):
            # Select a solution based on probability (roulette wheel selection)
            i = self.rng.choice(self.colony_size, p=probs)

            # Generate candidate from selected solution
            candidate = self._generate_candidate(i)
            candidate_objective = self.func(candidate)
            candidate_fitness = self._calculate_fitness(candidate_objective)
            self.stats['function_evaluations'] += 1

            # Greedy selection
            if candidate_fitness > self.fitness[i]:
                self.population[i] = candidate
                self.fitness[i] = candidate_fitness
                self.objective_values[i] = candidate_objective
                self.trial_counters[i] = 0
                self.stats['improvements'] += 1
            else:
                self.trial_counters[i] += 1
        
        self.stats['phase_times']['onlooker'] += time.time() - start_time

    def _scout_bees_phase(self):
        """Scout bees phase: replace abandoned solutions with random ones."""
        start_time = time.time()
        
        lb = self.bounds[:, 0]
        ub = self.bounds[:, 1]

        for i in range(self.colony_size):
            # If trial counter exceeds limit, abandon the solution
            if self.trial_counters[i] > self.limit:
                # Generate new random solution: x_new = lb + rand*(ub - lb)
                self.population[i] = self.rng.uniform(lb, ub)
                self.objective_values[i] = self.func(self.population[i])
                self.fitness[i] = self._calculate_fitness(self.objective_values[i])
                self.trial_counters[i] = 0  # Reset counter
                self.stats['function_evaluations'] += 1
                self.stats['scout_replacements'] += 1
        
        self.stats['phase_times']['scout'] += time.time() - start_time

    def optimize(self) -> Tuple[np.ndarray, float]:
        """
        Run the complete ABC optimization.

        Returns
        -------
        best_solution : np.ndarray
            The best solution found.
        best_objective : float
            Objective value of the best solution.
        """
        start_time = time.time()
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Starting ABC Optimization")
            print(f"{'='*60}")
            print(f"Dimensions: {self.n_dim}")
            print(f"Colony size: {self.colony_size}")
            print(f"Max iterations: {self.max_iter}")
            print(f"Limit parameter: {self.limit}")
            print(f"{'='*60}")

        # Progress bar for iterations
        iterator = range(self.max_iter)
        if self.verbose:
            iterator = tqdm(iterator, desc="ABC Optimization")

        for iteration in iterator:
            # 1. Employed bees phase
            self._employed_bees_phase()

            # 2. Onlooker bees phase
            self._onlooker_bees_phase()

            # 3. Scout bees phase
            self._scout_bees_phase()

            # 4. Update global best
            current_best_idx = np.argmax(self.fitness)
            if self.fitness[current_best_idx] > self.global_best_fitness:
                self.global_best_solution = self.population[current_best_idx].copy()
                self.global_best_fitness = self.fitness[current_best_idx]
                self.global_best_objective = self.objective_values[current_best_idx]

            # Record history for convergence analysis
            self.history.append({
                'iteration': iteration,
                'best_objective': self.global_best_objective,
                'best_fitness': self.global_best_fitness,
                'avg_fitness': np.mean(self.fitness),
                'std_fitness': np.std(self.fitness),
                'scout_replacements': self.stats['scout_replacements']
            })

        self.execution_time = time.time() - start_time

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Optimization Complete!")
            print(f"{'='*60}")
            print(f"Execution time: {self.execution_time:.2f} seconds")
            print(f"Function evaluations: {self.stats['function_evaluations']}")
            print(f"Improvements accepted: {self.stats['improvements']}")
            print(f"Scout replacements: {self.stats['scout_replacements']}")
            print(f"Best objective value: {self.global_best_objective:.6e}")
            print(f"Best solution: {self.global_best_solution}")
            print(f"{'='*60}")

        return self.global_best_solution, self.global_best_objective

    def get_convergence_data(self) -> Dict[str, np.ndarray]:
        """Return convergence data as arrays for plotting."""
        iterations = [h['iteration'] for h in self.history]
        best_objectives = [h['best_objective'] for h in self.history]
        avg_fitness = [h['avg_fitness'] for h in self.history]
        
        return {
            'iterations': np.array(iterations),
            'best_objectives': np.array(best_objectives),
            'avg_fitness': np.array(avg_fitness)
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Return algorithm statistics."""
        return {
            'execution_time': self.execution_time,
            'function_evaluations': self.stats['function_evaluations'],
            'improvements': self.stats['improvements'],
            'scout_replacements': self.stats['scout_replacements'],
            'phase_times': self.stats['phase_times'],
            'final_population_diversity': np.std(self.population, axis=0).mean()
        }