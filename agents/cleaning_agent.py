"""
Cleaning Agent - Handles data cleaning with human-in-the-loop confirmation
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional
from sklearn.impute import SimpleImputer, KNNImputer

from utils import ask_user_confirmation, compare_dataframes
from config import ANALYSIS_CONFIG

logger = logging.getLogger(__name__)

class CleaningAgent:
    """Agent responsible for data cleaning with human confirmation"""
    
    def __init__(self, interactive: bool = True):
        self.name = "Cleaning Agent"
        self.interactive = interactive
        self.cleaning_actions = []
        self.df_original = None
        
    def clean_missing_values(self, df: pd.DataFrame, missing_info: Dict[str, Any]) -> pd.DataFrame:
        """Clean missing values with user confirmation"""
        df_cleaned = df.copy()
        threshold_pct = ANALYSIS_CONFIG['missing_threshold'] * 100
        
        logger.info(f"{self.name}: Processing missing values...")
        
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count == 0:
                continue
                
            missing_pct = (missing_count / len(df)) * 100
            
            # Suggest action based on missing percentage and data type
            if missing_pct > 50:
                action = f"Drop column '{col}' (>{missing_pct:.1f}% missing)"
                if not self.interactive or ask_user_confirmation(f"Column '{col}' has {missing_pct:.1f}% missing values. Drop this column?"):
                    df_cleaned = df_cleaned.drop(columns=[col])
                    self.cleaning_actions.append(f"Dropped column '{col}' ({missing_pct:.1f}% missing)")
                    logger.info(f"Dropped column '{col}' due to excessive missing values")
                    continue
            
            elif missing_pct > threshold_pct:
                # Determine imputation strategy
                if pd.api.types.is_numeric_dtype(df[col]):
                    strategy = "median"
                    action = f"Impute column '{col}' with median ({missing_count} missing values)"
                    
                    if not self.interactive or ask_user_confirmation(f"Column '{col}' has {missing_count} missing values. Impute with median?"):
                        imputer = SimpleImputer(strategy='median')
                        df_cleaned[col] = imputer.fit_transform(df_cleaned[[col]]).ravel()
                        self.cleaning_actions.append(f"Imputed '{col}' with median ({missing_count} values)")
                        logger.info(f"Imputed column '{col}' with median")
                
                elif df[col].dtype == 'object':
                    mode_value = df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown'
                    action = f"Impute column '{col}' with mode/most frequent value ({missing_count} missing values)"
                    
                    if not self.interactive or ask_user_confirmation(f"Column '{col}' has {missing_count} missing values. Impute with most frequent value ('{mode_value}')?"):
                        df_cleaned[col] = df_cleaned[col].fillna(mode_value)
                        self.cleaning_actions.append(f"Imputed '{col}' with mode value '{mode_value}' ({missing_count} values)")
                        logger.info(f"Imputed column '{col}' with mode value")
                
                else:
                    # For other types, use forward fill or drop
                    action = f"Forward fill column '{col}' ({missing_count} missing values)"
                    
                    if not self.interactive or ask_user_confirmation(f"Column '{col}' has {missing_count} missing values. Use forward fill?"):
                        df_cleaned[col] = df_cleaned[col].fillna(method='ffill')
                        remaining_missing = df_cleaned[col].isnull().sum()
                        if remaining_missing > 0:
                            df_cleaned[col] = df_cleaned[col].fillna(method='bfill')
                        self.cleaning_actions.append(f"Forward/backward filled '{col}' ({missing_count} values)")
                        logger.info(f"Forward/backward filled column '{col}'")
        
        return df_cleaned
    
    def clean_duplicates(self, df: pd.DataFrame, duplicate_info: Dict[str, Any]) -> pd.DataFrame:
        """Remove duplicate rows with user confirmation"""
        df_cleaned = df.copy()
        
        duplicate_count = duplicate_info.get('total_duplicates', 0)
        if duplicate_count == 0:
            return df_cleaned
        
        logger.info(f"{self.name}: Processing {duplicate_count} duplicate rows...")
        
        if not self.interactive or ask_user_confirmation(f"Found {duplicate_count} duplicate rows. Remove duplicates?"):
            df_cleaned = df_cleaned.drop_duplicates()
            self.cleaning_actions.append(f"Removed {duplicate_count} duplicate rows")
            logger.info(f"Removed {duplicate_count} duplicate rows")
        
        return df_cleaned
    
    def clean_data_types(self, df: pd.DataFrame, type_info: Dict[str, Any]) -> pd.DataFrame:
        """Fix data type issues with user confirmation"""
        df_cleaned = df.copy()
        
        type_issues = type_info.get('potential_type_issues', [])
        if not type_issues:
            return df_cleaned
        
        logger.info(f"{self.name}: Processing data type issues...")
        
        for col in df.columns:
            if df[col].dtype == 'object':
                non_null_values = df[col].dropna()
                if len(non_null_values) == 0:
                    continue
                
                # Check if it's numeric data stored as object
                try:
                    pd.to_numeric(non_null_values.astype(str), errors='raise')
                    if not self.interactive or ask_user_confirmation(f"Column '{col}' appears to be numeric but stored as text. Convert to numeric?"):
                        df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
                        self.cleaning_actions.append(f"Converted '{col}' from object to numeric type")
                        logger.info(f"Converted column '{col}' to numeric type")
                except:
                    # Check if it's datetime data
                    try:
                        pd.to_datetime(non_null_values.head(100), errors='raise')
                        if not self.interactive or ask_user_confirmation(f"Column '{col}' appears to contain dates. Convert to datetime?"):
                            df_cleaned[col] = pd.to_datetime(df_cleaned[col], errors='coerce')
                            self.cleaning_actions.append(f"Converted '{col}' from object to datetime type")
                            logger.info(f"Converted column '{col}' to datetime type")
                    except:
                        pass
        
        return df_cleaned
    
    def clean_outliers(self, df: pd.DataFrame, outlier_info: Dict[str, Any]) -> pd.DataFrame:
        """Handle outliers with user confirmation"""
        df_cleaned = df.copy()
        
        outlier_summary = outlier_info.get('outlier_summary', {})
        outlier_indices = outlier_info.get('outlier_indices', {})
        
        logger.info(f"{self.name}: Processing outliers...")
        
        for col, summary in outlier_summary.items():
            if summary['count'] == 0:
                continue
            
            outlier_count = summary['count']
            outlier_pct = summary['percentage']
            
            # Only process if significant outlier percentage
            if outlier_pct > 5:  # More than 5% outliers
                indices = outlier_indices.get(col, [])
                
                if not self.interactive or ask_user_confirmation(f"Column '{col}' has {outlier_count} outliers ({outlier_pct:.1f}%). Remove outlier rows?"):
                    df_cleaned = df_cleaned.drop(indices)
                    self.cleaning_actions.append(f"Removed {outlier_count} outlier rows from column '{col}'")
                    logger.info(f"Removed {outlier_count} outliers from column '{col}'")
                elif not self.interactive or ask_user_confirmation(f"Would you like to cap outliers in '{col}' instead?"):
                    # Cap outliers at 95th and 5th percentiles
                    q95 = df_cleaned[col].quantile(0.95)
                    q05 = df_cleaned[col].quantile(0.05)
                    df_cleaned[col] = df_cleaned[col].clip(lower=q05, upper=q95)
                    self.cleaning_actions.append(f"Capped outliers in column '{col}' at 5th-95th percentiles")
                    logger.info(f"Capped outliers in column '{col}'")
        
        return df_cleaned
    
    def clean_categorical_high_cardinality(self, df: pd.DataFrame, categorical_info: Dict[str, Any]) -> pd.DataFrame:
        """Handle high cardinality categorical variables"""
        df_cleaned = df.copy()
        
        high_card_cols = categorical_info.get('high_cardinality_columns', [])
        if not high_card_cols:
            return df_cleaned
        
        logger.info(f"{self.name}: Processing high cardinality categorical columns...")
        
        for col in high_card_cols:
            unique_count = df[col].nunique()
            
            if not self.interactive or ask_user_confirmation(f"Column '{col}' has {unique_count} unique values (high cardinality). Group rare categories?"):
                # Keep top categories, group others as 'Other'
                value_counts = df_cleaned[col].value_counts()
                top_categories = value_counts.head(20).index  # Keep top 20 categories
                
                df_cleaned[col] = df_cleaned[col].where(df_cleaned[col].isin(top_categories), 'Other')
                
                new_unique_count = df_cleaned[col].nunique()
                self.cleaning_actions.append(f"Grouped rare categories in '{col}' (reduced from {unique_count} to {new_unique_count} categories)")
                logger.info(f"Grouped rare categories in column '{col}'")
        
        return df_cleaned
    
    def validate_cleaning(self, df_original: pd.DataFrame, df_cleaned: pd.DataFrame) -> Dict[str, Any]:
        """Validate the cleaning results"""
        validation = {
            'is_valid': True,
            'issues': [],
            'warnings': [],
            'comparison': compare_dataframes(df_original, df_cleaned)
        }
        
        # Check if we lost too much data
        row_loss_pct = ((df_original.shape[0] - df_cleaned.shape[0]) / df_original.shape[0]) * 100
        if row_loss_pct > 30:
            validation['warnings'].append(f"Lost {row_loss_pct:.1f}% of rows during cleaning")
        
        # Check if we lost important columns
        col_loss = df_original.shape[1] - df_cleaned.shape[1]
        if col_loss > 0:
            validation['warnings'].append(f"Dropped {col_loss} columns during cleaning")
        
        # Check for new missing values (shouldn't happen)
        new_missing = df_cleaned.isnull().sum().sum()
        original_missing = df_original.isnull().sum().sum()
        if new_missing > original_missing:
            validation['issues'].append("Cleaning introduced new missing values")
            validation['is_valid'] = False
        
        # Check data types are reasonable
        for col in df_cleaned.columns:
            if df_cleaned[col].dtype == 'object' and col in df_original.columns:
                if pd.api.types.is_numeric_dtype(df_original[col]):
                    validation['warnings'].append(f"Column '{col}' was converted from numeric to object")
        
        return validation
    
    def process(self, df: pd.DataFrame, inspection_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main processing method for the Cleaning Agent
        
        Args:
            df: DataFrame to clean
            inspection_results: Results from inspection agent
            
        Returns:
            Cleaning results with cleaned DataFrame
        """
        logger.info(f"{self.name}: Starting data cleaning process")
        self.df_original = df.copy()
        self.cleaning_actions = []
        
        try:
            df_cleaned = df.copy()
            
            # Extract inspection information
            missing_info = inspection_results.get('missing_values', {})
            duplicate_info = inspection_results.get('duplicates', {})
            type_info = inspection_results.get('data_types', {})
            outlier_info = inspection_results.get('outliers', {})
            categorical_info = inspection_results.get('categorical', {})
            
            # Perform cleaning steps
            if missing_info.get('total_missing', 0) > 0:
                df_cleaned = self.clean_missing_values(df_cleaned, missing_info)
            
            if duplicate_info.get('total_duplicates', 0) > 0:
                df_cleaned = self.clean_duplicates(df_cleaned, duplicate_info)
            
            if type_info.get('potential_type_issues'):
                df_cleaned = self.clean_data_types(df_cleaned, type_info)
            
            if outlier_info.get('outlier_summary'):
                df_cleaned = self.clean_outliers(df_cleaned, outlier_info)
            
            if categorical_info.get('high_cardinality_columns'):
                df_cleaned = self.clean_categorical_high_cardinality(df_cleaned, categorical_info)
            
            # Validate cleaning results
            validation = self.validate_cleaning(df, df_cleaned)
            
            # Generate comparison metrics
            comparison = compare_dataframes(df, df_cleaned)
            
            result = {
                'agent': self.name,
                'status': 'success',
                'data': df_cleaned,
                'original_data': df,
                'cleaning_actions': self.cleaning_actions,
                'validation': validation,
                'comparison': comparison,
                'message': f"Cleaning completed. Applied {len(self.cleaning_actions)} cleaning actions."
            }
            
            logger.info(f"{self.name}: Cleaning completed successfully")
            logger.info(f"  - Actions applied: {len(self.cleaning_actions)}")
            logger.info(f"  - Shape change: {df.shape} -> {df_cleaned.shape}")
            
            return result
            
        except Exception as e:
            error_msg = f"Cleaning failed: {str(e)}"
            logger.error(f"{self.name}: {error_msg}")
            
            return {
                'agent': self.name,
                'status': 'error',
                'data': df,
                'cleaning_actions': self.cleaning_actions,
                'message': error_msg,
                'error': str(e)
            }