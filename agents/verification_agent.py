"""
Verification Agent - Validates cleaned and transformed dataset quality
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Tuple

from utils import calculate_data_quality_metrics, get_column_types

logger = logging.getLogger(__name__)

class VerificationAgent:
    """Agent responsible for validating final dataset quality"""
    
    def __init__(self):
        self.name = "Verification Agent"
        self.quality_checks = []
        
    def verify_data_completeness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Verify data completeness"""
        completeness_check = {
            'status': 'pass',
            'issues': [],
            'warnings': [],
            'metrics': {}
        }
        
        # Check for missing values
        total_missing = df.isnull().sum().sum()
        missing_percentage = (total_missing / (df.shape[0] * df.shape[1])) * 100
        
        completeness_check['metrics'] = {
            'total_missing_values': total_missing,
            'missing_percentage': round(missing_percentage, 2),
            'rows_with_missing': df.isnull().any(axis=1).sum(),
            'columns_with_missing': df.isnull().any().sum()
        }
        
        # Set status based on missing data
        if missing_percentage > 10:
            completeness_check['status'] = 'fail'
            completeness_check['issues'].append(f"High missing data percentage: {missing_percentage:.2f}%")
        elif missing_percentage > 5:
            completeness_check['status'] = 'warning'
            completeness_check['warnings'].append(f"Moderate missing data: {missing_percentage:.2f}%")
        
        # Check for completely empty columns
        empty_cols = df.columns[df.isnull().all()].tolist()
        if empty_cols:
            completeness_check['status'] = 'fail'
            completeness_check['issues'].append(f"Empty columns found: {empty_cols}")
        
        # Check for columns with very few values
        sparse_cols = []
        for col in df.columns:
            non_null_count = df[col].count()
            if non_null_count < 0.1 * len(df):  # Less than 10% data
                sparse_cols.append(col)
        
        if sparse_cols:
            completeness_check['warnings'].append(f"Sparse columns (< 10% data): {sparse_cols}")
        
        return completeness_check
    
    def verify_data_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Verify data consistency and types"""
        consistency_check = {
            'status': 'pass',
            'issues': [],
            'warnings': [],
            'metrics': {}
        }
        
        # Check for mixed data types in object columns
        mixed_type_issues = []
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].notna().sum() > 0:
                sample_types = df[col].dropna().head(100).apply(type).unique()
                if len(sample_types) > 1:
                    mixed_type_issues.append(col)
        
        if mixed_type_issues:
            consistency_check['warnings'].append(f"Columns with mixed types: {mixed_type_issues}")
        
        # Check for reasonable value ranges in numeric columns
        range_issues = []
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].notna().sum() > 0:
                col_min, col_max = df[col].min(), df[col].max()
                
                # Check for extreme values
                if abs(col_max) > 1e10 or abs(col_min) > 1e10:
                    range_issues.append(f"{col}: extreme values ({col_min:.2e} to {col_max:.2e})")
                
                # Check for infinite values
                if np.isinf(df[col]).any():
                    consistency_check['issues'].append(f"Column '{col}' contains infinite values")
                    consistency_check['status'] = 'fail'
                
                # Check for NaN values in numeric columns
                if np.isnan(df[col]).any():
                    nan_count = np.isnan(df[col]).sum()
                    consistency_check['warnings'].append(f"Column '{col}' has {nan_count} NaN values")
        
        if range_issues:
            consistency_check['warnings'].extend(range_issues)
        
        consistency_check['metrics'] = {
            'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
            'object_columns': len(df.select_dtypes(include=['object']).columns),
            'datetime_columns': len(df.select_dtypes(include=['datetime64']).columns),
            'total_columns': len(df.columns)
        }
        
        return consistency_check
    
    def verify_data_distributions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Verify data distributions are reasonable"""
        distribution_check = {
            'status': 'pass',
            'issues': [],
            'warnings': [],
            'metrics': {}
        }
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            distribution_check['warnings'].append("No numeric columns to verify distributions")
            return distribution_check
        
        skewness_issues = []
        variance_issues = []
        
        for col in numeric_cols:
            if df[col].notna().sum() > 3:
                # Calculate distribution metrics
                try:
                    skewness = df[col].skew()
                    variance = df[col].var()
                    std = df[col].std()
                    
                    # Check for extreme skewness
                    if abs(skewness) > 5:
                        skewness_issues.append(f"{col}: skewness {skewness:.2f}")
                    
                    # Check for zero or very low variance
                    if variance == 0:
                        distribution_check['issues'].append(f"Column '{col}' has zero variance (constant values)")
                        distribution_check['status'] = 'fail'
                    elif std < 1e-10:
                        variance_issues.append(f"{col}: very low variance ({variance:.2e})")
                    
                except Exception as e:
                    distribution_check['warnings'].append(f"Could not analyze distribution for '{col}': {str(e)}")
        
        if skewness_issues:
            distribution_check['warnings'].append(f"Highly skewed columns: {skewness_issues}")
        
        if variance_issues:
            distribution_check['warnings'].append(f"Low variance columns: {variance_issues}")
        
        distribution_check['metrics'] = {
            'columns_analyzed': len(numeric_cols),
            'highly_skewed_count': len(skewness_issues),
            'low_variance_count': len(variance_issues)
        }
        
        return distribution_check
    
    def verify_data_uniqueness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Verify data uniqueness and identify potential duplicates"""
        uniqueness_check = {
            'status': 'pass',
            'issues': [],
            'warnings': [],
            'metrics': {}
        }
        
        # Check for duplicate rows
        duplicate_count = df.duplicated().sum()
        duplicate_percentage = (duplicate_count / len(df)) * 100
        
        uniqueness_check['metrics'] = {
            'total_rows': len(df),
            'duplicate_rows': duplicate_count,
            'duplicate_percentage': round(duplicate_percentage, 2),
            'unique_rows': len(df) - duplicate_count
        }
        
        if duplicate_count > 0:
            if duplicate_percentage > 5:
                uniqueness_check['status'] = 'warning'
                uniqueness_check['warnings'].append(f"Found {duplicate_count} duplicate rows ({duplicate_percentage:.2f}%)")
            else:
                uniqueness_check['warnings'].append(f"Found {duplicate_count} duplicate rows")
        
        # Check for columns that might be identifiers
        potential_id_cols = []
        for col in df.columns:
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio > 0.95:  # More than 95% unique values
                potential_id_cols.append(col)
        
        if potential_id_cols:
            uniqueness_check['warnings'].append(f"Potential identifier columns: {potential_id_cols}")
        
        uniqueness_check['metrics']['potential_id_columns'] = potential_id_cols
        
        return uniqueness_check
    
    def verify_data_size_and_memory(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Verify data size and memory usage are reasonable"""
        size_check = {
            'status': 'pass',
            'issues': [],
            'warnings': [],
            'metrics': {}
        }
        
        # Calculate memory usage
        memory_usage_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        row_count = len(df)
        col_count = len(df.columns)
        
        size_check['metrics'] = {
            'rows': row_count,
            'columns': col_count,
            'memory_usage_mb': round(memory_usage_mb, 2),
            'cells': row_count * col_count
        }
        
        # Check for very large datasets
        if memory_usage_mb > 1000:  # > 1GB
            size_check['warnings'].append(f"Large dataset: {memory_usage_mb:.1f} MB memory usage")
        
        # Check for very small datasets
        if row_count < 10:
            size_check['status'] = 'warning'
            size_check['warnings'].append(f"Very small dataset: only {row_count} rows")
        
        # Check for too many columns relative to rows
        if col_count > row_count:
            size_check['warnings'].append(f"More columns ({col_count}) than rows ({row_count}) - potential overfitting risk")
        
        return size_check
    
    def calculate_overall_quality_score(self, verification_results: Dict[str, Any]) -> float:
        """Calculate an overall quality score based on verification results"""
        total_score = 100.0
        
        for check_name, check_result in verification_results.items():
            if check_name == 'overall_assessment':
                continue
                
            if check_result['status'] == 'fail':
                total_score -= 20  # Major penalty for failures
            elif check_result['status'] == 'warning':
                total_score -= 5   # Minor penalty for warnings
            
            # Additional penalties for specific issues
            issues_count = len(check_result.get('issues', []))
            warnings_count = len(check_result.get('warnings', []))
            
            total_score -= (issues_count * 10)  # 10 points per issue
            total_score -= (warnings_count * 2)  # 2 points per warning
        
        return max(0.0, min(100.0, total_score))  # Clamp between 0 and 100
    
    def generate_verification_summary(self, verification_results: Dict[str, Any]) -> str:
        """Generate a human-readable verification summary"""
        summary = []
        summary.append(f"=== {self.name} Summary ===\n")
        
        overall_score = verification_results.get('overall_assessment', {}).get('quality_score', 0)
        summary.append(f"Overall Data Quality Score: {overall_score:.1f}/100\n")
        
        # Count total issues and warnings
        total_issues = 0
        total_warnings = 0
        
        for check_name, check_result in verification_results.items():
            if check_name == 'overall_assessment':
                continue
            
            issues = len(check_result.get('issues', []))
            warnings = len(check_result.get('warnings', []))
            total_issues += issues
            total_warnings += warnings
            
            status = check_result.get('status', 'unknown')
            summary.append(f"{check_name.replace('_', ' ').title()}: {status.upper()}")
            if issues > 0:
                summary.append(f"  - {issues} critical issues")
            if warnings > 0:
                summary.append(f"  - {warnings} warnings")
        
        summary.append(f"\nTotal Issues: {total_issues}")
        summary.append(f"Total Warnings: {total_warnings}")
        
        # Overall assessment
        if overall_score >= 90:
            summary.append("\n✅ Dataset quality is EXCELLENT")
        elif overall_score >= 75:
            summary.append("\n✅ Dataset quality is GOOD")
        elif overall_score >= 60:
            summary.append("\n⚠️  Dataset quality is ACCEPTABLE with some concerns")
        elif overall_score >= 40:
            summary.append("\n⚠️  Dataset quality is POOR - significant issues found")
        else:
            summary.append("\n❌ Dataset quality is CRITICAL - major issues require attention")
        
        return "\n".join(summary)
    
    def process(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Main processing method for the Verification Agent
        
        Args:
            df: DataFrame to verify
            
        Returns:
            Complete verification results
        """
        logger.info(f"{self.name}: Starting dataset verification")
        
        try:
            verification_results = {}
            
            # Run all verification checks
            logger.info(f"{self.name}: Checking data completeness...")
            verification_results['completeness'] = self.verify_data_completeness(df)
            
            logger.info(f"{self.name}: Checking data consistency...")
            verification_results['consistency'] = self.verify_data_consistency(df)
            
            logger.info(f"{self.name}: Checking data distributions...")
            verification_results['distributions'] = self.verify_data_distributions(df)
            
            logger.info(f"{self.name}: Checking data uniqueness...")
            verification_results['uniqueness'] = self.verify_data_uniqueness(df)
            
            logger.info(f"{self.name}: Checking data size and memory...")
            verification_results['size_and_memory'] = self.verify_data_size_and_memory(df)
            
            # Calculate overall quality score
            quality_score = self.calculate_overall_quality_score(verification_results)
            
            # Generate overall assessment
            total_issues = sum(len(result.get('issues', [])) for result in verification_results.values())
            total_warnings = sum(len(result.get('warnings', [])) for result in verification_results.values())
            
            overall_status = 'pass'
            if total_issues > 0:
                overall_status = 'fail'
            elif total_warnings > 0:
                overall_status = 'warning'
            
            verification_results['overall_assessment'] = {
                'status': overall_status,
                'quality_score': round(quality_score, 1),
                'total_issues': total_issues,
                'total_warnings': total_warnings,
                'data_quality_metrics': calculate_data_quality_metrics(df)
            }
            
            # Generate summary
            summary = self.generate_verification_summary(verification_results)
            
            result = {
                'agent': self.name,
                'status': overall_status,
                'verification_results': verification_results,
                'quality_score': quality_score,
                'summary': summary,
                'message': f"Verification completed. Quality score: {quality_score:.1f}/100, {total_issues} issues, {total_warnings} warnings."
            }
            
            logger.info(f"{self.name}: Verification completed")
            logger.info(f"  - Quality score: {quality_score:.1f}/100")
            logger.info(f"  - Issues: {total_issues}, Warnings: {total_warnings}")
            
            return result
            
        except Exception as e:
            error_msg = f"Verification failed: {str(e)}"
            logger.error(f"{self.name}: {error_msg}")
            
            return {
                'agent': self.name,
                'status': 'error',
                'verification_results': {},
                'message': error_msg,
                'error': str(e)
            }