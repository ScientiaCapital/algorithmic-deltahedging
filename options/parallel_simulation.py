"""
Parallel Simulation Module

This module implements multi-threaded and multiprocess simulations
for Monte Carlo pricing, portfolio optimization, and risk analysis.
"""

import numpy as np
from typing import List, Callable, Any, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import time


class ParallelSimulator:
    """
    Parallel Monte Carlo simulator using multi-threading or multiprocessing.
    """

    def __init__(
        self,
        num_workers: Optional[int] = None,
        use_processes: bool = False
    ):
        """
        Initialize parallel simulator.

        Args:
            num_workers: Number of worker threads/processes (default: CPU count)
            use_processes: Use multiprocessing instead of threading
        """
        self.num_workers = num_workers or cpu_count()
        self.use_processes = use_processes

    def run_simulations(
        self,
        simulation_func: Callable,
        num_simulations: int,
        *args,
        **kwargs
    ) -> List[Any]:
        """
        Run simulations in parallel.

        Args:
            simulation_func: Function to run for each simulation
            num_simulations: Total number of simulations
            *args: Positional arguments for simulation_func
            **kwargs: Keyword arguments for simulation_func

        Returns:
            List of results from all simulations
        """
        # Split simulations across workers
        sims_per_worker = num_simulations // self.num_workers
        remainder = num_simulations % self.num_workers

        # Create batches
        batches = [sims_per_worker] * self.num_workers
        for i in range(remainder):
            batches[i] += 1

        # Choose executor
        executor_class = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor

        results = []

        with executor_class(max_workers=self.num_workers) as executor:
            # Submit tasks
            futures = [
                executor.submit(
                    self._run_batch,
                    simulation_func,
                    batch_size,
                    *args,
                    **kwargs
                )
                for batch_size in batches
            ]

            # Collect results
            for future in as_completed(futures):
                batch_results = future.result()
                results.extend(batch_results)

        return results

    @staticmethod
    def _run_batch(
        simulation_func: Callable,
        batch_size: int,
        *args,
        **kwargs
    ) -> List[Any]:
        """
        Run a batch of simulations.

        Args:
            simulation_func: Simulation function
            batch_size: Number of simulations in this batch
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            List of results for this batch
        """
        return [simulation_func(*args, **kwargs) for _ in range(batch_size)]


def parallel_monte_carlo_option_price(
    asset_price: float,
    strike_price: float,
    volatility: float,
    time_to_expiration: float,
    risk_free_rate: float,
    option_type: str = 'call',
    num_simulations: int = 100000,
    num_workers: Optional[int] = None
) -> tuple:
    """
    Calculate option price using parallel Monte Carlo simulation.

    Args:
        asset_price: Current asset price
        strike_price: Strike price
        volatility: Volatility
        time_to_expiration: Time to expiration in years
        risk_free_rate: Risk-free rate
        option_type: 'call' or 'put'
        num_simulations: Number of simulations
        num_workers: Number of parallel workers

    Returns:
        Tuple of (price, standard_error, computation_time)
    """
    start_time = time.time()

    def single_simulation():
        # Simulate final price using GBM
        z = np.random.normal(0, 1)
        drift = (risk_free_rate - 0.5 * volatility**2) * time_to_expiration
        diffusion = volatility * np.sqrt(time_to_expiration) * z
        final_price = asset_price * np.exp(drift + diffusion)

        # Calculate payoff
        if option_type == 'call':
            payoff = max(final_price - strike_price, 0)
        else:
            payoff = max(strike_price - final_price, 0)

        return payoff

    # Run parallel simulations
    simulator = ParallelSimulator(num_workers=num_workers, use_processes=False)
    payoffs = simulator.run_simulations(single_simulation, num_simulations)

    # Calculate price
    discount_factor = np.exp(-risk_free_rate * time_to_expiration)
    payoffs_array = np.array(payoffs)
    price = discount_factor * np.mean(payoffs_array)
    std_error = discount_factor * np.std(payoffs_array) / np.sqrt(num_simulations)

    computation_time = time.time() - start_time

    return price, std_error, computation_time


def parallel_var_calculation(
    portfolio_value: float,
    expected_return: float,
    volatility: float,
    confidence_level: float = 0.95,
    time_horizon_days: int = 1,
    num_simulations: int = 100000,
    num_workers: Optional[int] = None
) -> tuple:
    """
    Calculate VaR using parallel Monte Carlo simulation.

    Args:
        portfolio_value: Current portfolio value
        expected_return: Expected daily return
        volatility: Daily volatility
        confidence_level: Confidence level
        time_horizon_days: Time horizon in days
        num_simulations: Number of simulations
        num_workers: Number of parallel workers

    Returns:
        Tuple of (VaR, computation_time)
    """
    start_time = time.time()

    def single_simulation():
        # Simulate portfolio return
        z = np.random.normal(0, 1)
        portfolio_return = (expected_return * time_horizon_days +
                           volatility * np.sqrt(time_horizon_days) * z)
        final_value = portfolio_value * (1 + portfolio_return)
        loss = portfolio_value - final_value
        return loss

    # Run parallel simulations
    simulator = ParallelSimulator(num_workers=num_workers, use_processes=False)
    losses = simulator.run_simulations(single_simulation, num_simulations)

    # Calculate VaR
    losses_array = np.array(losses)
    var = np.percentile(losses_array, (1 - confidence_level) * 100)

    computation_time = time.time() - start_time

    return var, computation_time


def parallel_portfolio_optimization(
    assets_returns: List[np.ndarray],
    num_portfolios: int = 10000,
    num_workers: Optional[int] = None
) -> tuple:
    """
    Perform parallel portfolio optimization using Monte Carlo.

    Args:
        assets_returns: List of return arrays for each asset
        num_portfolios: Number of random portfolios to generate
        num_workers: Number of parallel workers

    Returns:
        Tuple of (best_weights, best_sharpe, all_results, computation_time)
    """
    start_time = time.time()

    num_assets = len(assets_returns)

    # Calculate covariance matrix
    returns_matrix = np.column_stack(assets_returns)
    cov_matrix = np.cov(returns_matrix.T)
    mean_returns = np.mean(returns_matrix, axis=0)

    def single_portfolio():
        # Generate random weights
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)

        # Calculate portfolio metrics
        portfolio_return = np.dot(weights, mean_returns) * 252
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)

        # Sharpe ratio (assuming 4% risk-free rate)
        risk_free_rate = 0.04
        sharpe = (portfolio_return - risk_free_rate) / portfolio_std if portfolio_std > 0 else 0

        return {
            'weights': weights,
            'return': portfolio_return,
            'volatility': portfolio_std,
            'sharpe': sharpe
        }

    # Run parallel simulations
    simulator = ParallelSimulator(num_workers=num_workers, use_processes=False)
    portfolios = simulator.run_simulations(single_portfolio, num_portfolios)

    # Find best portfolio
    best_portfolio = max(portfolios, key=lambda p: p['sharpe'])

    computation_time = time.time() - start_time

    return (
        best_portfolio['weights'],
        best_portfolio['sharpe'],
        portfolios,
        computation_time
    )


class ParallelBacktester:
    """
    Parallel backtesting for strategy optimization.
    """

    def __init__(self, num_workers: Optional[int] = None):
        """
        Initialize parallel backtester.

        Args:
            num_workers: Number of parallel workers
        """
        self.simulator = ParallelSimulator(num_workers=num_workers, use_processes=True)

    def optimize_strategy_parameters(
        self,
        strategy_func: Callable,
        historical_data: Any,
        parameter_grid: dict,
        metric: str = 'sharpe_ratio'
    ) -> dict:
        """
        Optimize strategy parameters using parallel backtesting.

        Args:
            strategy_func: Strategy function to test
            historical_data: Historical data for backtesting
            parameter_grid: Dictionary of parameters to test
            metric: Metric to optimize ('sharpe_ratio', 'total_return', etc.)

        Returns:
            Dictionary with best parameters and results
        """
        # Generate all parameter combinations
        import itertools

        keys = parameter_grid.keys()
        values = parameter_grid.values()
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

        def test_params(params):
            # Run backtest with these parameters
            # (Simplified - full implementation would use BacktestEngine)
            result = {
                'params': params,
                'sharpe_ratio': np.random.normal(1.0, 0.5),  # Placeholder
                'total_return': np.random.normal(0.1, 0.05)  # Placeholder
            }
            return result

        # Run parallel backtests
        results = self.simulator.run_simulations(
            test_params,
            len(combinations),
            # Would pass appropriate args here
        )

        # Find best parameters
        best_result = max(results, key=lambda r: r[metric])

        return best_result


def benchmark_parallel_vs_serial(
    num_simulations: int = 100000,
    num_workers: Optional[int] = None
) -> dict:
    """
    Benchmark parallel vs serial Monte Carlo simulation.

    Args:
        num_simulations: Number of simulations
        num_workers: Number of parallel workers

    Returns:
        Dictionary with benchmark results
    """
    # Test parameters
    S = 100.0
    K = 105.0
    vol = 0.3
    T = 1.0
    r = 0.05

    # Serial execution
    print("Running serial simulation...")
    start = time.time()
    payoffs_serial = []
    for _ in range(num_simulations):
        z = np.random.normal(0, 1)
        ST = S * np.exp((r - 0.5 * vol**2) * T + vol * np.sqrt(T) * z)
        payoffs_serial.append(max(ST - K, 0))
    serial_time = time.time() - start
    serial_price = np.exp(-r * T) * np.mean(payoffs_serial)

    # Parallel execution
    print("Running parallel simulation...")
    parallel_price, std_error, parallel_time = parallel_monte_carlo_option_price(
        S, K, vol, T, r, 'call', num_simulations, num_workers
    )

    # Calculate speedup
    speedup = serial_time / parallel_time if parallel_time > 0 else 0

    return {
        'num_simulations': num_simulations,
        'num_workers': num_workers or cpu_count(),
        'serial_time': serial_time,
        'parallel_time': parallel_time,
        'speedup': speedup,
        'serial_price': serial_price,
        'parallel_price': parallel_price,
        'price_difference': abs(serial_price - parallel_price)
    }


if __name__ == "__main__":
    # Example usage
    print("Running parallel simulation benchmark...")
    results = benchmark_parallel_vs_serial(num_simulations=50000)

    print("\n" + "="*50)
    print("PARALLEL SIMULATION BENCHMARK")
    print("="*50)
    print(f"Number of simulations: {results['num_simulations']:,}")
    print(f"Number of workers:     {results['num_workers']}")
    print(f"\nSerial time:           {results['serial_time']:.4f}s")
    print(f"Parallel time:         {results['parallel_time']:.4f}s")
    print(f"Speedup:               {results['speedup']:.2f}x")
    print(f"\nSerial price:          ${results['serial_price']:.4f}")
    print(f"Parallel price:        ${results['parallel_price']:.4f}")
    print(f"Difference:            ${results['price_difference']:.6f}")
    print("="*50)
