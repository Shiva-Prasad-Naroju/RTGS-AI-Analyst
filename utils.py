"""
Utility functions for RTGS AI Analyst
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_logging(log_file: Optional[str] = None):
    """Setup logging configuration"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    if log_file:
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(level=logging.INFO, format=log_format)

def detect_file_encoding(file_path: str) -> str:
    """Detect file encoding using chardet"""
    try:
        import chardet
        with open(file_path, 'rb') as f:
            raw_data = f.read()
            result = chardet.detect(raw_data)
            return result['encoding'] or 'utf-8'
    except ImportError:
        logger.warning("chardet not available, using utf-8 encoding")
        return 'utf-8'

def load_dataset(file_path: str) -> pd.DataFrame:
    """Load dataset from CSV or Excel file"""
    try:
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.csv':
            encoding = detect_file_encoding(file_path)
            df = pd.read_csv(file_path, encoding=encoding)
        elif file_ext in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        logger.info(f"Successfully loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
        
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise

def get_column_types(df: pd.DataFrame) -> Dict[str, str]:
    """Get detailed column type information"""
    column_types = {}
    
    for col in df.columns:
        dtype = str(df[col].dtype)
        
        # Classify into broader categories
        if dtype.startswith('int') or dtype.startswith('float'):
            column_types[col] = 'numeric'
        elif dtype == 'object':
            # Check if it's actually dates or categorical
            if df[col].dtype == 'object':
                # Try to detect if it's date-like
                try:
                    pd.to_datetime(df[col].dropna().head(10))
                    column_types[col] = 'datetime'
                except:
                    # Check if it's categorical (low cardinality)
                    unique_ratio = df[col].nunique() / len(df)
                    if unique_ratio < 0.5:
                        column_types[col] = 'categorical'
                    else:
                        column_types[col] = 'text'
        elif dtype.startswith('datetime'):
            column_types[col] = 'datetime'
        elif dtype == 'bool':
            column_types[col] = 'boolean'
        else:
            column_types[col] = 'other'
    
    return column_types

def detect_outliers(df: pd.DataFrame, column: str, method: str = 'zscore', threshold: float = 3.0) -> List[int]:
    """Detect outliers in a numeric column"""
    if not pd.api.types.is_numeric_dtype(df[column]):
        return []
    
    if method == 'zscore':
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        return df[z_scores > threshold].index.tolist()
    
    elif method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return df[(df[column] < lower_bound) | (df[column] > upper_bound)].index.tolist()
    
    return []

def calculate_data_quality_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate comprehensive data quality metrics"""
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isnull().sum().sum()
    
    metrics = {
        'total_rows': df.shape[0],
        'total_columns': df.shape[1],
        'total_cells': total_cells,
        'missing_cells': missing_cells,
        'missing_percentage': (missing_cells / total_cells) * 100 if total_cells > 0 else 0,
        'duplicate_rows': df.duplicated().sum(),
        'duplicate_percentage': (df.duplicated().sum() / df.shape[0]) * 100 if df.shape[0] > 0 else 0,
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
        'column_types': get_column_types(df),
        'completeness_by_column': ((df.count() / len(df)) * 100).round(2).to_dict(),
    }
    
    # Add numeric column statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        metrics['numeric_columns'] = len(numeric_cols)
        metrics['skewness'] = df[numeric_cols].skew().to_dict()
        metrics['kurtosis'] = df[numeric_cols].kurtosis().to_dict()
    else:
        metrics['numeric_columns'] = 0
        metrics['skewness'] = {}
        metrics['kurtosis'] = {}
    
    return metrics

def format_number(num: float, decimals: int = 2) -> str:
    """Format number with appropriate precision"""
    if abs(num) >= 1000000:
        return f"{num/1000000:.{decimals}f}M"
    elif abs(num) >= 1000:
        return f"{num/1000:.{decimals}f}K"
    else:
        return f"{num:.{decimals}f}"

def get_timestamp() -> str:
    """Get current timestamp for file naming"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def ask_user_confirmation(message: str) -> bool:
    """Ask user for confirmation (Y/N)"""
    while True:
        response = input(f"{message} (Y/N): ").strip().upper()
        if response in ['Y', 'YES']:
            return True
        elif response in ['N', 'NO']:
            return False
        else:
            print("Please enter Y or N")

def save_dataframe(df: pd.DataFrame, file_path: str, include_index: bool = False):
    """Save DataFrame to CSV with proper encoding"""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=include_index, encoding='utf-8')
        logger.info(f"DataFrame saved to: {file_path}")
    except Exception as e:
        logger.error(f"Error saving DataFrame: {str(e)}")
        raise

def compare_dataframes(df_before: pd.DataFrame, df_after: pd.DataFrame) -> Dict[str, Any]:
    """Compare two dataframes and return differences"""
    comparison = {
        'shape_before': df_before.shape,
        'shape_after': df_after.shape,
        'rows_changed': df_before.shape[0] - df_after.shape[0],
        'columns_changed': df_before.shape[1] - df_after.shape[1],
        'missing_before': df_before.isnull().sum().sum(),
        'missing_after': df_after.isnull().sum().sum(),
        'memory_before_mb': df_before.memory_usage(deep=True).sum() / 1024 / 1024,
        'memory_after_mb': df_after.memory_usage(deep=True).sum() / 1024 / 1024,
    }
    
    # Calculate improvement metrics
    if comparison['missing_before'] > 0:
        comparison['missing_reduction_pct'] = ((comparison['missing_before'] - comparison['missing_after']) / comparison['missing_before']) * 100
    else:
        comparison['missing_reduction_pct'] = 0
    
    return comparison

def generate_summary_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate comprehensive summary statistics"""
    stats = {
        'basic_info': {
            'rows': df.shape[0],
            'columns': df.shape[1],
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
        },
        'missing_data': df.isnull().sum().to_dict(),
        'data_types': df.dtypes.astype(str).to_dict(),
    }
    
    # Numeric statistics
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        stats['numeric_summary'] = numeric_df.describe().to_dict()
    
    # Categorical statistics
    categorical_df = df.select_dtypes(include=['object', 'category'])
    if not categorical_df.empty:
        stats['categorical_summary'] = {}
        for col in categorical_df.columns:
            stats['categorical_summary'][col] = {
                'unique_values': df[col].nunique(),
                'most_frequent': df[col].mode().iloc[0] if not df[col].mode().empty else None,
                'frequency': df[col].value_counts().head(5).to_dict()
            }
    
    return stats