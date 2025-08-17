"""
Simulation and Visualization of Brownian Motion Paths

This script generates and visualizes sample paths of both standard and generalized Brownian motion,
along with their terminal value distributions.

References:
- Glasserman, P. (2003). Monte Carlo Methods in Financial Engineering.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Configuration
plt.style.use('seaborn')
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.size'] = 10

# Parameters
PATHS = 50
POINTS = 1000
TIME_INTERVAL = (0.0, 1.0)
DRIFT = 5.0
VOLATILITY = 2.0
SEED = 42

def simulate_brownian_motion(paths, points, interval, mu=0.0, sigma=1.0):
    """Simulate Brownian motion paths.
    
    Args:
        paths: Number of paths to simulate
        points: Number of points in each path
        interval: Tuple of (start_time, end_time)
        mu: Drift coefficient (default 0 for standard BM)
        sigma: Volatility coefficient (default 1 for standard BM)
        
    Returns:
        Tuple of (time_axis, simulated_paths)
    """
    rng = np.random.default_rng(SEED)
    dt = (interval[1] - interval[0]) / (points - 1)
    t_axis = np.linspace(interval[0], interval[1], points)
    
    # Vectorized implementation for better performance
    Z = rng.normal(0, 1, (paths, points - 1))
    increments = mu * dt + sigma * np.sqrt(dt) * Z
    W = np.cumsum(np.column_stack([np.zeros(paths), increments]), axis=1)
    
    return t_axis, W

def plot_paths(t_axis, paths, title):
    """Plot simulated paths with consistent styling."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t_axis, paths.T, alpha=0.6, linewidth=0.8)
    
    ax.set_title(title, pad=20)
    ax.set_xlabel("Time", labelpad=10)
    ax.set_ylabel("Value", labelpad=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_distribution(values, title):
    """Plot distribution of terminal values."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if isinstance(values, np.ndarray):
        values = pd.DataFrame({'final_values': values})
    
    sns.kdeplot(data=values, x='final_values', fill=True, ax=ax, 
                color='royalblue', alpha=0.6, linewidth=1.5)
    
    ax.set_title(title, pad=20)
    ax.set_xlabel('Terminal Value', labelpad=10)
    ax.set_ylabel('Density', labelpad=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    # 1. Standard Brownian Motion
    t_axis, W = simulate_brownian_motion(PATHS, POINTS, TIME_INTERVAL)
    plot_paths(t_axis, W, "Standard Brownian Motion Sample Paths")
    
    final_values = W[:, -1]
    plot_distribution(final_values, 
                     "Terminal Value Distribution (Standard Brownian Motion)")
    
    print("\nStandard Brownian Motion Statistics:")
    print(f"Mean: {final_values.mean():.4f}")
    print(f"Std Dev: {final_values.std():.4f}")
    print(f"Expected Std Dev: {np.sqrt(TIME_INTERVAL[1]):.4f}")
    
    # 2. Generalized Brownian Motion
    _, X = simulate_brownian_motion(PATHS, POINTS, TIME_INTERVAL, DRIFT, VOLATILITY)
    plot_paths(t_axis, X, f"Generalized Brownian Motion (μ={DRIFT}, σ={VOLATILITY})")
    
    final_values_X = X[:, -1]
    plot_distribution(final_values_X,
                    f"Terminal Value Distribution (μ={DRIFT}, σ={VOLATILITY})")
    
    print("\nGeneralized Brownian Motion Statistics:")
    print(f"Mean: {final_values_X.mean():.4f} (Expected: {DRIFT * TIME_INTERVAL[1]:.4f})")
    print(f"Std Dev: {final_values_X.std():.4f} (Expected: {VOLATILITY * np.sqrt(TIME_INTERVAL[1]):.4f})")

if __name__ == "__main__":
    main()
