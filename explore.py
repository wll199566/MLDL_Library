"""
Explore Data.
"""
import numpy as np

def describe(values:np.ndarray):
    """
    Describe data distribution statistics.
    Args:
        values: 1d numpy array.
    """
    # Remove missing values  
    values = values[~np.isnan(values)]
    
    # Print statistics 
    print(f"min: {np.min(values):.4f} | max: {np.max(values):.4f}")
    print(f"mean: {np.mean(values):.4f} | std: {np.std(values):.4f}")
    print(f"25%: {np.percentile(values, q=25):.4f} | 50%: {np.median(values):.4f} | 75%: {np.percentile(values, q=75):.4f}")
    print(f"5%: {np.percentile(values, q=5):.4f} | 95%: {np.percentile(values, q=95):.4f}")
    print(f"1%: {np.percentile(values, q=1):.4f} | 99%: {np.percentile(values, q=99):.4f}")
