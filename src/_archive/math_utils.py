import numpy as np
import pandas as pd
from typing import Tuple

def calculate_linear_regression_slope(series: pd.Series, period: int = 100) -> float:
    """
    Calculate the slope of a linear regression line for a given period.
    Returns the raw slope (change in y per unit change in x).
    """
    if len(series) < period:
        return 0.0
    
    y = series.tail(period).values
    x = np.arange(period)
    
    # Linear regression formula: slope = (n*sum(xy) - sum(x)*sum(y)) / (n*sum(x^2) - (sum(x))^2)
    n = period
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_xx = np.sum(x**2)
    
    denominator = n * sum_xx - sum_x**2
    if denominator == 0:
        return 0.0
        
    slope = (n * sum_xy - sum_x * sum_y) / denominator
    return slope

def calculate_normalized_slope_degrees(series: pd.Series, period: int = 100) -> float:
    """
    Calculate the normalized slope in degrees (-90 to 90).
    Normalization is done by scaling the price change relative to the average price.
    
    Formula: 
    1. Calculate raw slope.
    2. Normalize: (slope / avg_price) * 100 (percentage change per bar).
    3. Convert to degrees: arctan(normalized_slope) * (180 / pi).
    """
    if len(series) < period:
        return 0.0
        
    y = series.tail(period).values
    avg_price = np.mean(y)
    
    if avg_price == 0:
        return 0.0
        
    raw_slope = calculate_linear_regression_slope(series, period)
    
    # Normalize slope as percentage change per bar
    # We want to know how many % the price changes on average per bar
    norm_slope = (raw_slope / avg_price) * 100
    
    # For institutional "45-degree" logic, we need to map this to a scale where 
    # a violent move is considered > 45 degrees. 
    # We'll use a scaling factor so that a 0.1% change per bar (very violent for 100 bars) 
    # maps to a high angle.
    # If a 100-period move is 10% total, that's 0.1% per bar. 
    # tan(45) = 1. So we scale norm_slope such that 0.1% = 1.0 (45 degrees).
    
    angle_deg = np.degrees(np.arctan(norm_slope * 10)) # 0.1% * 10 = 1.0 -> 45 deg
    
    return angle_deg
