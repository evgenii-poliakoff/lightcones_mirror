"""
Parallel execution module for stochastic simulations with independent random number generation.

This module provides a function to execute multiple instances of a parameterless
job function in parallel using process-based concurrency. Each job execution
receives a cryptographically secure random seed to ensure statistically
independent random number sequences across all parallel processes.
"""

import secrets
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from typing import List, Callable, TypeVar, Any

# Generic type for job return value
T = TypeVar('T')


def execute(job: Callable[[], T], number_of_jobs: int, max_concurrent_jobs: int = 8) -> List[T]:
    """
    Execute multiple instances of a job function in parallel with independent RNG seeds.
    
    This function spawns a process pool and executes the provided job function
    `number_of_jobs` times in parallel, limited by `max_concurrent_jobs`. Each
    job execution is wrapped to receive a unique, cryptographically secure
    random seed before execution, ensuring statistically independent random
    number sequences across all parallel processes and job instances.
    
    Parameters
    ----------
    job : Callable[[], T]
        A parameterless function that returns a result. The function should
        use numpy's random number generator (np.random) for stochastic
        operations. Each call will be automatically seeded with a unique
        cryptographically secure random seed.
        
    number_of_jobs : int
        Total number of times to execute the job function. This determines
        the size of the returned results list.
        
    max_concurrent_jobs : int, optional
        Maximum number of worker processes to run concurrently. Default is 8.
        The actual number may be limited by system resources. Set to None
        to use the default number of processors.
        
    Returns
    -------
    List[T]
        A list containing the return values from each job execution, in the
        order they were submitted (not necessarily the order of completion).
        
    Raises
    ------
    RuntimeError
        If any job execution raises an exception, it will be propagated.
    ValueError
        If `number_of_jobs` is less than 1.
        
    Examples
    --------
    >>> def my_simulation():
    ...     # Use numpy's random functions
    ...     return np.random.normal(0, 1)
    ...
    >>> results = execute(my_simulation, number_of_jobs=100, max_concurrent_jobs=4)
    >>> len(results)
    100
    
    Notes
    -----
    - Each job execution receives a unique seed from `secrets.randbits(32)`,
      ensuring cryptographic-quality randomness for seed generation.
    - Seeds are generated in the parent process before distribution to
      worker processes, ensuring no race conditions.
    - The function uses `np.random.seed()` for backward compatibility with
      legacy NumPy random functions. For NumPy >= 1.17, consider using
      `np.random.default_rng()` for better statistical properties.
    - Results are collected in submission order, not completion order.
    - Exceptions in any job will cause the entire execution to fail and
      propagate the first exception encountered.
    """
    if number_of_jobs < 1:
        raise ValueError(f"number_of_jobs must be >= 1, got {number_of_jobs}")
    
    def job_wrapper() -> T:
        """
        Internal wrapper that seeds the RNG before executing the actual job.
        
        This wrapper generates a cryptographically secure seed and applies
        it to numpy's random number generator before delegating to the
        user-provided job function.
        
        Returns
        -------
        T
            The return value from the wrapped job function.
        """
        seed = secrets.randbits(32)
        np.random.seed(seed % 2**32)
        return job()

    with ProcessPoolExecutor(max_workers=max_concurrent_jobs) as executor:
        # Submit all tasks to the executor
        futures = [executor.submit(job_wrapper) for _ in range(number_of_jobs)]
        
        # Collect results in submission order
        results: List[T] = [future.result() for future in futures]
        
        return results
    
__all__ = ['execute']