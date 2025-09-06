"""
Ingestion Agent - Handles dataset loading and basic validation
"""
import pandas as pd
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import os

from utils import load_dataset, get_timestamp

logger = logging.getLogger(__name__)

class IngestionAgent:
    """Agent responsible for loading and basic validation of datasets"""
    
    def __init__(self):
        self.name = "Ingestion Agent"
        self.loaded_data = None
        self.metadata = {}
        
    def load_dataset(self, file_path: str) -> Dict[str, Any]:
        """
        Load dataset from file and perform basic validation
        
        Args:
            file_path: Path to the dataset file
            
        Returns:
            Dictionary containing dataset and metadata
        """
        try:
            logger.info(f"{self.name}: Starting dataset ingestion from {file_path}")
            
            # Validate file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Dataset file not found: {file_path}")
            
            # Get file info
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            file_ext = Path(file_path).suffix.lower()
            
            # Load the dataset
            df = load_dataset(file_path)
            
            # Basic validation
            if df.empty:
                raise ValueError("Loaded dataset is empty")
            
            # Store loaded data
            self.loaded_data = df
            
            # Create metadata
            self.metadata = {
                'file_path': file_path,
                'file_name': Path(file_path).name,
                'file_size_mb': round(file_size, 2),
                'file_extension': file_ext,
                'load_timestamp': get_timestamp(),
                'rows': df.shape[0],
                'columns': df.shape[1],
                'column_names': df.columns.tolist(),
                'memory_usage_mb': round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
            }
            
            logger.info(f"{self.name}: Successfully loaded dataset")
            logger.info(f"  - Shape: {df.shape}")
            logger.info(f"  - Size: {file_size:.2f} MB")
            logger.info(f"  - Columns: {len(df.columns)}")
            
            return {
                'status': 'success',
                'data': df,
                'metadata': self.metadata,
                'message': f"Successfully loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns"
            }
            
        except Exception as e:
            error_msg = f"Failed to load dataset: {str(e)}"
            logger.error(f"{self.name}: {error_msg}")
            
            return {
                'status': 'error',
                'data': None,
                'metadata': {},
                'message': error_msg,
                'error': str(e)
            }
    
    def validate_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform additional validation checks on the dataset
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Validation results
        """
        validation_results = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'checks': {}
        }
        
        try:
            # Check 1: Empty dataset
            if df.empty:
                validation_results['errors'].append("Dataset is empty")
                validation_results['is_valid'] = False
            
            # Check 2: Very small dataset
            if len(df) < 5:
                validation_results['warnings'].append(f"Dataset is very small ({len(df)} rows)")
            
            # Check 3: Too many columns
            if len(df.columns) > 100:
                validation_results['warnings'].append(f"Dataset has many columns ({len(df.columns)})")
            
            # Check 4: Column name issues
            duplicate_cols = df.columns[df.columns.duplicated()].tolist()
            if duplicate_cols:
                validation_results['errors'].append(f"Duplicate column names found: {duplicate_cols}")
                validation_results['is_valid'] = False
            
            # Check 5: Unnamed columns
            unnamed_cols = [col for col in df.columns if str(col).startswith('Unnamed:')]
            if unnamed_cols:
                validation_results['warnings'].append(f"Unnamed columns detected: {unnamed_cols}")
            
            # Check 6: Memory usage
            memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
            if memory_mb > 500:  # 500 MB threshold
                validation_results['warnings'].append(f"Large dataset ({memory_mb:.1f} MB)")
            
            # Store check results
            validation_results['checks'] = {
                'row_count': len(df),
                'column_count': len(df.columns),
                'memory_usage_mb': round(memory_mb, 2),
                'has_duplicate_columns': len(duplicate_cols) > 0,
                'has_unnamed_columns': len(unnamed_cols) > 0
            }
            
            logger.info(f"{self.name}: Dataset validation completed")
            if validation_results['warnings']:
                logger.warning(f"Validation warnings: {'; '.join(validation_results['warnings'])}")
            if validation_results['errors']:
                logger.error(f"Validation errors: {'; '.join(validation_results['errors'])}")
            
        except Exception as e:
            validation_results['errors'].append(f"Validation failed: {str(e)}")
            validation_results['is_valid'] = False
            logger.error(f"{self.name}: Validation error: {str(e)}")
        
        return validation_results
    
    def get_basic_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get basic information about the dataset
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Basic dataset information
        """
        try:
            info = {
                'shape': df.shape,
                'column_names': df.columns.tolist(),
                'data_types': df.dtypes.astype(str).to_dict(),
                'missing_values': df.isnull().sum().to_dict(),
                'memory_usage': df.memory_usage(deep=True).to_dict(),
                'total_missing': df.isnull().sum().sum(),
                'missing_percentage': (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
            }
            
            # Add sample data
            info['sample_data'] = {
                'head': df.head(3).to_dict('records'),
                'tail': df.tail(3).to_dict('records')
            }
            
            return info
            
        except Exception as e:
            logger.error(f"{self.name}: Error getting basic info: {str(e)}")
            return {'error': str(e)}
    
    def process(self, file_path: str) -> Dict[str, Any]:
        """
        Main processing method for the Ingestion Agent
        
        Args:
            file_path: Path to the dataset file
            
        Returns:
            Complete processing results
        """
        logger.info(f"{self.name}: Starting ingestion process")
        
        # Load dataset
        load_result = self.load_dataset(file_path)
        
        if load_result['status'] != 'success':
            return load_result
        
        df = load_result['data']
        
        # Validate dataset
        validation_result = self.validate_dataset(df)
        
        # Get basic info
        basic_info = self.get_basic_info(df)
        
        # Compile final result
        result = {
            'agent': self.name,
            'status': 'success' if validation_result['is_valid'] else 'warning',
            'data': df,
            'metadata': load_result['metadata'],
            'validation': validation_result,
            'basic_info': basic_info,
            'message': f"Dataset ingestion completed. {load_result['message']}"
        }
        
        if not validation_result['is_valid']:
            result['message'] += f" Validation errors found: {'; '.join(validation_result['errors'])}"
        
        logger.info(f"{self.name}: Ingestion process completed")
        return result