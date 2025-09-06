"""
Utility functions for RTGS AI Analyst
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import re

def detect_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Detect and categorize column types based on content
    
    Returns:
        Dictionary with lists of column names by category
    """
    column_types = {
        'id_columns': [],
        'name_columns': [],
        'location_columns': [],
        'date_columns': [],
        'numeric_columns': [],
        'categorical_columns': [],
        'text_columns': []
    }
    
    for col in df.columns:
        col_lower = col.lower()
        
        # ID columns
        if any(word in col_lower for word in ['id', 'key', 'index', 'code']):
            column_types['id_columns'].append(col)
        
        # Name columns
        elif any(word in col_lower for word in ['name', 'title', 'hospital']):
            column_types['name_columns'].append(col)
        
        # Location columns
        elif any(word in col_lower for word in ['district', 'state', 'city', 'address', 'location', 'pin']):
            column_types['location_columns'].append(col)
        
        # Date columns
        elif any(word in col_lower for word in ['date', 'year', 'month', 'establish', 'start', 'end']):
            column_types['date_columns'].append(col)
        
        # Numeric columns
        elif df[col].dtype in ['int64', 'float64']:
            column_types['numeric_columns'].append(col)
        
        # Categorical vs text columns
        elif df[col].dtype == 'object':
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio < 0.5 and df[col].nunique() <= 50:
                column_types['categorical_columns'].append(col)
            else:
                column_types['text_columns'].append(col)
    
    return column_types

def standardize_categorical_values(series: pd.Series) -> pd.Series:
    """
    Standardize categorical values by fixing case and spacing issues
    """
    if series.dtype != 'object':
        return series
    
    # Create a mapping for standardization
    value_mapping = {}
    unique_values = series.dropna().unique()
    
    for value in unique_values:
        # Clean the value
        cleaned = str(value).strip().title()
        
        # Handle common variations
        if cleaned.lower() in ['govt', 'government', 'public']:
            cleaned = 'Government'
        elif cleaned.lower() in ['pvt', 'private']:
            cleaned = 'Private'
        elif cleaned.lower() in ['yes', 'y', '1', 'true']:
            cleaned = 'Yes'
        elif cleaned.lower() in ['no', 'n', '0', 'false']:
            cleaned = 'No'
        
        value_mapping[value] = cleaned
    
    return series.map(value_mapping)

def calculate_completeness_score(df: pd.DataFrame) -> float:
    """
    Calculate overall data completeness score (0-100)
    """
    total_cells = len(df) * len(df.columns)
    missing_cells = df.isnull().sum().sum()
    completeness = ((total_cells - missing_cells) / total_cells) * 100
    return round(completeness, 2)

def detect_outliers_iqr(series: pd.Series, multiplier: float = 1.5) -> Dict[str, Any]:
    """
    Detect outliers using IQR method
    
    Args:
        series: Numeric series to analyze
        multiplier: IQR multiplier for outlier detection
        
    Returns:
        Dictionary with outlier information
    """
    if series.dtype not in ['int64', 'float64']:
        return {'outliers': [], 'lower_bound': None, 'upper_bound': None, 'count': 0}
    
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    outliers = series[(series < lower_bound) | (series > upper_bound)]
    
    return {
        'outliers': outliers.tolist(),
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'count': len(outliers),
        'percentage': (len(outliers) / len(series)) * 100
    }

def generate_data_profile(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate comprehensive data profile
    """
    profile = {
        'basic_info': {
            'rows': len(df),
            'columns': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'completeness_score': calculate_completeness_score(df)
        },
        'column_types': detect_column_types(df),
        'data_quality': {
            'null_count': df.isnull().sum().sum(),
            'duplicate_rows': df.duplicated().sum(),
            'unique_values_per_column': df.nunique().to_dict()
        }
    }
    
    return profile

def validate_healthcare_data(df: pd.DataFrame) -> List[str]:
    """
    Validate healthcare-specific business rules
    
    Returns:
        List of validation warnings/errors
    """
    warnings = []
    
    # Check bed capacity columns
    bed_columns = [col for col in df.columns if 'bed' in col.lower()]
    for col in bed_columns:
        if col in df.columns and df[col].dtype in ['int64', 'float64']:
            # Check for negative values
            negative_count = (df[col] < 0).sum()
            if negative_count > 0:
                warnings.append(f"Found {negative_count} negative bed capacity values in '{col}'")
            
            # Check for unrealistic values
            max_beds = df[col].max()
            if max_beds > 10000:
                warnings.append(f"Unusually high bed capacity ({max_beds}) found in '{col}'")
    
    # Check establishment dates
    date_columns = [col for col in df.columns if any(word in col.lower() for word in ['establish', 'start', 'date'])]
    current_year = pd.Timestamp.now().year
    
    for col in date_columns:
        if col in df.columns:
            try:
                dates = pd.to_datetime(df[col], errors='coerce')
                future_dates = (dates.dt.year > current_year).sum()
                if future_dates > 0:
                    warnings.append(f"Found {future_dates} future dates in '{col}'")
                    
                very_old = (dates.dt.year < 1800).sum()
                if very_old > 0:
                    warnings.append(f"Found {very_old} unrealistic old dates in '{col}'")
            except:
                pass
    
    # Check hospital types
    type_columns = [col for col in df.columns if 'type' in col.lower()]
    for col in type_columns:
        if col in df.columns and df[col].dtype == 'object':
            unknown_count = df[col].str.lower().str.contains('unknown|other|misc').sum()
            total_count = len(df[col].dropna())
            if unknown_count / total_count > 0.2:
                warnings.append(f"High percentage ({unknown_count/total_count:.1%}) of unknown/other types in '{col}'")
    
    return warnings

def create_ascii_chart(data: Dict[str, int], title: str = "Chart", max_width: int = 50) -> str:
    """
    Create ASCII bar chart for CLI display
    """
    if not data:
        return f"{title}\nNo data to display"
    
    chart = f"\n{title}\n" + "=" * len(title) + "\n"
    
    max_value = max(data.values())
    max_key_length = max(len(str(k)) for k in data.keys())
    
    for key, value in sorted(data.items(), key=lambda x: x[1], reverse=True):
        # Calculate bar length
        bar_length = int((value / max_value) * max_width) if max_value > 0 else 0
        bar = "█" * bar_length
        
        # Format the line
        key_str = str(key).ljust(max_key_length)
        chart += f"{key_str} │{bar} {value}\n"
    
    return chart

def format_bytes(bytes_value: int) -> str:
    """Format bytes into human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} TB"

def clean_column_name(name: str) -> str:
    """
    Clean column name to be more standardized
    """
    # Remove special characters and replace with underscores
    cleaned = re.sub(r'[^\w\s]', '', str(name))
    
    # Replace spaces with underscores
    cleaned = re.sub(r'\s+', '_', cleaned)
    
    # Remove consecutive underscores
    cleaned = re.sub(r'_+', '_', cleaned)
    
    # Remove leading/trailing underscores
    cleaned = cleaned.strip('_')
    
    # Convert to title case for readability
    cleaned = '_'.join(word.capitalize() for word in cleaned.split('_'))
    
    return cleaned

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if division by zero"""
    try:
        return numerator / denominator if denominator != 0 else default
    except (TypeError, ZeroDivisionError):
        return default

def estimate_processing_time(df_size: tuple, operations_count: int) -> str:
    """
    Estimate processing time based on dataset size and operations
    """
    rows, cols = df_size
    total_cells = rows * cols
    
    # Rough estimates based on typical performance
    base_time = total_cells * 0.00001  # seconds per cell
    operation_factor = operations_count * 0.1  # additional time per operation
    
    estimated_seconds = base_time + operation_factor
    
    if estimated_seconds < 60:
        return f"~{estimated_seconds:.0f} seconds"
    elif estimated_seconds < 3600:
        return f"~{estimated_seconds/60:.1f} minutes"
    else:
        return f"~{estimated_seconds/3600:.1f} hours"
