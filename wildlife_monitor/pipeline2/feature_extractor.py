"""
Turns each detection into a fixed-length of numbers
"""

from __future__ import annotations
import math
from datetime import datetime

def extract_hour(timestamp: str) -> float:
    """
    Pulls the hour from timestamp string.
    Returns 12.0 as default
    """
    try:
        dt = datetime.strptime(str(timestamp), "%Y-%m-%d %H:%M:%S")
        return dt.hour + dt.minute / 60.0
    except (ValueError, TypeError):
        return 12.0

def cyclic_hour_encoding(hour: float) -> tuple[float, float]:
    """
    Turn hour-of-day into two numbers on a circle, so 23 and hour 0
    are recognised as close.
    """
    radians = 2 * math.pi * hour / 24.0
    return math.sin(radians), math.cos(radians)

def extract_day_of_year(timestamp: str) -> float:
    """
    Pull the day of year out of timestamp
    return 182.0 as defualt(roughly mid-year)
    """
    try:
        dt = datetime.strptime(str(timestamp), "%Y-%m-%d %H:%M:%S")
        return float(dt.timetuple().tm_yday)
    except(ValueError, TypeError):
        return 182.0

def cyclcic_day_encoding(day_of_year: float) -> tuple[float, float]:
    """
    same circular trick as for hour but now for day of the year
    """
    radians = 2 * math.pi * day_of_year / 365.0
    return math.sin(radians), math.cos(radians)

def build_feature_vector(timestamp: str, instance_count: int,) -> list[float]:
    """
    Convert one detection row into a list of 5 number that a neural network can process
    """
    hour = extract_hour(timestamp)
    day = extract_day_of_year(timestamp)

    sin_hour, cos_hour = cyclic_hour_encoding(hour)
    sin_day, cos_day = cyclcic_day_encoding(day)

    log_count = math.log(instance_count + 1)

    return [sin_hour, cos_hour, sin_day, cos_day, log_count,]