"""
Inspection Agent - Analyzes dataset quality and identifies issues
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Tuple
from scipy import stats

from utils import get_column_types, detect_outliers, calculate_data_quality_metrics
from config import ANALYSIS_CONFIG

logger = logging.getLogger(__name__)

class InspectionAgent:
    """Agent responsible for deep inspection and issue identification"""
    
    def __init__(self):
        self.name = "Inspection Agent"
        self.issues = []
        self.recommendations = []
        
    def inspect_missing_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Inspect missing values patterns"""
        missing_info = {
            'total_missing': df.isnull().sum().sum(),
            'missing_by_column': df.isnull().sum().to_dict(),
            'missing_percentage_by_column': ((df.isnull().sum() / len(df)) * 100).round(2).to_dict(),
            'rows_with_missing': df.isnull().any(axis=1).sum(),
            'columns_with_missing': df.isnull().any().sum(),
            'action_required': []
        }
        
        # Identify columns requiring action
        threshold = ANALYSIS_CONFIG['missing_threshold'] * 100  # Convert to percentage
        
        for col, pct in missing_info['missing_percentage_by_column'].items():
            if pct > threshold:
                action = f"Column '{col}': {pct:.1f}% missing values. Action required: "
                
                if pct > 50:
                    action += "Consider dropping column due to excessive missing data."
                elif df[col].dtype in ['object']:
                    action += "Impute with mode or 'Unknown' category."
                elif pd.api.types.is_numeric_dtype(df[col]):
                    action += "Impute with mean, median, or use advanced imputation."
                else:
                    action += "Review and decide appropriate imputation strategy."
                
                missing_info['action_required'].append(action)
                
        return missing_info
    
    def inspect_duplicates(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Inspect duplicate rows"""
        duplicate_info = {
            'total_duplicates': df.duplicated().sum(),
            'duplicate_percentage': (df.duplicated().sum() / len(df)) * 100,
            'duplicate_indices': df[df.duplicated()].index.tolist(),
            'action_required': []
        }
        
        if duplicate_info['total_duplicates'] > 0:
            action = f"Found {duplicate_info['total_duplicates']} duplicate rows ({duplicate_info['duplicate_percentage']:.2f}%). "
            action += "Action required: Review and remove duplicates."
            duplicate_info['action_required'].append(action)
            
        return duplicate_info
    
    def inspect_data_types(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Inspect data types and type inconsistencies"""
        type_info = {
            'column_types': get_column_types(df),
            'dtype_summary': df.dtypes.value_counts().to_dict(),
            'potential_type_issues': [],
            'action_required': []
        }
        
        # Check for potential type issues
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if numeric data stored as object
                non_null_values = df[col].dropna()
                if len(non_null_values) > 0:
                    # Try to convert to numeric
                    try:
                        pd.to_numeric(non_null_values.astype(str), errors='raise')
                        issue = f"Column '{col}' contains numeric data but stored as object type."
                        type_info['potential_type_issues'].append(issue)
                        type_info['action_required'].append(f"{issue} Action required: Convert to numeric type.")
                    except:
                        # Try to detect dates
                        try:
                            pd.to_datetime(non_null_values.head(100), errors='raise')
                            issue = f"Column '{col}' appears to contain date/time data."
                            type_info['potential_type_issues'].append(issue)
                            type_info['action_required'].append(f"{issue} Action required: Convert to datetime type.")
                        except:
                            pass
        
        return type_info
    
    def inspect_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Inspect outliers in numeric columns"""
        outlier_info = {
            'outlier_summary': {},
            'outlier_indices': {},
            'action_required': []
        }
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df[col].notna().sum() > 0:  # Skip if all null
                outlier_indices = detect_outliers(df, col, threshold=ANALYSIS_CONFIG['outlier_threshold'])
                outlier_count = len(outlier_indices)
                outlier_percentage = (outlier_count / len(df)) * 100
                
                outlier_info['outlier_summary'][col] = {
                    'count': outlier_count,
                    'percentage': round(outlier_percentage, 2)
                }
                outlier_info['outlier_indices'][col] = outlier_indices
                
                if outlier_percentage > 5:  # More than 5% outliers
                    action = f"Column '{col}': {outlier_count} outliers ({outlier_percentage:.1f}%). "
                    action += "Action required: Review outliers and consider removal or transformation."
                    outlier_info['action_required'].append(action)
        
        return outlier_info
    
    def inspect_distributions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Inspect statistical distributions"""
        dist_info = {
            'skewness': {},
            'kurtosis': {},
            'normality_tests': {},
            'action_required': []
        }
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df[col].notna().sum() > 3:  # Need at least 3 values
                data = df[col].dropna()
                
                # Calculate skewness and kurtosis
                skew = stats.skew(data)
                kurt = stats.kurtosis(data)
                
                dist_info['skewness'][col] = round(skew, 3)
                dist_info['kurtosis'][col] = round(kurt, 3)
                
                # Normality test (if sample size is reasonable)
                if len(data) >= 8 and len(data) <= 5000:
                    try:
                        _, p_value = stats.shapiro(data)
                        dist_info['normality_tests'][col] = {
                            'test': 'shapiro',
                            'p_value': round(p_value, 6),
                            'is_normal': p_value > 0.05
                        }
                    except:
                        dist_info['normality_tests'][col] = {
                            'test': 'failed',
                            'p_value': None,
                            'is_normal': False
                        }
                
                # Check for highly skewed data
                if abs(skew) > ANALYSIS_CONFIG['skewness_threshold']:
                    direction = "right" if skew > 0 else "left"
                    action = f"Column '{col}': Highly {direction}-skewed (skewness: {skew:.2f}). "
                    action += "Action required: Consider log transformation or other normalization techniques."
                    dist_info['action_required'].append(action)
        
        return dist_info
    
    def inspect_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Inspect correlations between numeric variables"""
        corr_info = {
            'correlation_matrix': {},
            'high_correlations': [],
            'action_required': []
        }
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            corr_info['correlation_matrix'] = corr_matrix.round(3).to_dict()
            
            # Find high correlations (excluding diagonal)
            threshold = ANALYSIS_CONFIG['correlation_threshold']
            high_corr_pairs = []
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                    corr_value = corr_matrix.iloc[i, j]
                    
                    if abs(corr_value) > threshold and not pd.isna(corr_value):
                        high_corr_pairs.append({
                            'column1': col1,
                            'column2': col2,
                            'correlation': round(corr_value, 3)
                        })
            
            corr_info['high_correlations'] = high_corr_pairs
            
            if high_corr_pairs:
                for pair in high_corr_pairs:
                    action = f"High correlation between '{pair['column1']}' and '{pair['column2']}' ({pair['correlation']:.3f}). "
                    action += "Action required: Consider removing one variable or feature engineering."
                    corr_info['action_required'].append(action)
        
        return corr_info
    
    def inspect_categorical_variables(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Inspect categorical variables"""
        cat_info = {
            'categorical_summary': {},
            'high_cardinality_columns': [],
            'action_required': []
        }
        
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            unique_count = df[col].nunique()
            unique_ratio = unique_count / len(df)
            most_frequent = df[col].mode().iloc[0] if not df[col].mode().empty else None
            most_frequent_count = df[col].value_counts().iloc[0] if len(df[col].value_counts()) > 0 else 0
            
            cat_info['categorical_summary'][col] = {
                'unique_count': unique_count,
                'unique_ratio': round(unique_ratio, 3),
                'most_frequent_value': most_frequent,
                'most_frequent_count': most_frequent_count,
                'most_frequent_percentage': round((most_frequent_count / len(df)) * 100, 2)
            }
            
            # Check for high cardinality
            if unique_ratio > 0.9:
                cat_info['high_cardinality_columns'].append(col)
                action = f"Column '{col}': Very high cardinality ({unique_count} unique values, {unique_ratio:.1%}). "
                action += "Action required: Consider if this should be treated as an identifier rather than categorical."
                cat_info['action_required'].append(action)
            elif unique_ratio > 0.5:
                action = f"Column '{col}': High cardinality ({unique_count} unique values). "
                action += "Action required: Consider grouping rare categories or feature engineering."
                cat_info['action_required'].append(action)
        
        return cat_info
    
    def generate_summary_report(self, inspection_results: Dict[str, Any]) -> str:
        """Generate a human-readable summary report"""
        report = []
        report.append(f"=== {self.name} Summary Report ===\n")
        
        # Count total issues
        total_actions = 0
        for result in inspection_results.values():
            if isinstance(result, dict) and 'action_required' in result:
                total_actions += len(result['action_required'])
        
        report.append(f"Total Issues Identified: {total_actions}\n")
        
        # Missing values summary
        missing = inspection_results.get('missing_values', {})
        if missing.get('total_missing', 0) > 0:
            report.append(f"Missing Values: {missing['total_missing']} total missing values across {missing.get('columns_with_missing', 0)} columns")
        
        # Duplicates summary
        duplicates = inspection_results.get('duplicates', {})
        if duplicates.get('total_duplicates', 0) > 0:
            report.append(f"Duplicates: {duplicates['total_duplicates']} duplicate rows found")
        
        # Data type issues
        dtypes = inspection_results.get('data_types', {})
        if dtypes.get('potential_type_issues'):
            report.append(f"Data Type Issues: {len(dtypes['potential_type_issues'])} potential type conversion opportunities")
        
        # Outliers summary
        outliers = inspection_results.get('outliers', {})
        outlier_cols = [col for col, info in outliers.get('outlier_summary', {}).items() if info['count'] > 0]
        if outlier_cols:
            report.append(f"Outliers: Found in {len(outlier_cols)} numeric columns")
        
        # Distribution issues
        distributions = inspection_results.get('distributions', {})
        skewed_cols = len(distributions.get('action_required', []))
        if skewed_cols > 0:
            report.append(f"Distribution Issues: {skewed_cols} columns with distribution concerns")
        
        # Correlation issues
        correlations = inspection_results.get('correlations', {})
        high_corr_count = len(correlations.get('high_correlations', []))
        if high_corr_count > 0:
            report.append(f"High Correlations: {high_corr_count} highly correlated variable pairs")
        
        # Categorical issues
        categorical = inspection_results.get('categorical', {})
        high_card_count = len(categorical.get('high_cardinality_columns', []))
        if high_card_count > 0:
            report.append(f"Categorical Issues: {high_card_count} high cardinality columns")
        
        report.append("\n" + "="*50)
        
        return "\n".join(report)
    
    def process(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Main processing method for the Inspection Agent
        
        Args:
            df: DataFrame to inspect
            
        Returns:
            Complete inspection results
        """
        logger.info(f"{self.name}: Starting comprehensive dataset inspection")
        
        try:
            inspection_results = {}
            
            # Run all inspection methods
            logger.info(f"{self.name}: Inspecting missing values...")
            inspection_results['missing_values'] = self.inspect_missing_values(df)
            
            logger.info(f"{self.name}: Inspecting duplicates...")
            inspection_results['duplicates'] = self.inspect_duplicates(df)
            
            logger.info(f"{self.name}: Inspecting data types...")
            inspection_results['data_types'] = self.inspect_data_types(df)
            
            logger.info(f"{self.name}: Inspecting outliers...")
            inspection_results['outliers'] = self.inspect_outliers(df)
            
            logger.info(f"{self.name}: Inspecting distributions...")
            inspection_results['distributions'] = self.inspect_distributions(df)
            
            logger.info(f"{self.name}: Inspecting correlations...")
            inspection_results['correlations'] = self.inspect_correlations(df)
            
            logger.info(f"{self.name}: Inspecting categorical variables...")
            inspection_results['categorical'] = self.inspect_categorical_variables(df)
            
            # Generate quality metrics
            logger.info(f"{self.name}: Calculating data quality metrics...")
            inspection_results['quality_metrics'] = calculate_data_quality_metrics(df)
            
            # Generate summary report
            summary_report = self.generate_summary_report(inspection_results)
            
            # Compile all action required items
            all_actions = []
            for result in inspection_results.values():
                if isinstance(result, dict) and 'action_required' in result:
                    all_actions.extend(result['action_required'])
            
            result = {
                'agent': self.name,
                'status': 'success',
                'inspection_results': inspection_results,
                'action_required': all_actions,
                'summary_report': summary_report,
                'total_issues': len(all_actions),
                'message': f"Inspection completed. Found {len(all_actions)} issues requiring attention."
            }
            
            logger.info(f"{self.name}: Inspection completed successfully")
            logger.info(f"  - Total issues found: {len(all_actions)}")
            
            return result
            
        except Exception as e:
            error_msg = f"Inspection failed: {str(e)}"
            logger.error(f"{self.name}: {error_msg}")
            
            return {
                'agent': self.name,
                'status': 'error',
                'inspection_results': {},
                'action_required': [],
                'message': error_msg,
                'error': str(e)
            }