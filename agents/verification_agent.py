import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import logging

class VerificationAgent:
    """Agent responsible for verifying data quality after cleaning and transformation"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.verification_results = {}
    
    def verify_dataset(self, 
                      original_df: pd.DataFrame, 
                      final_df: pd.DataFrame, 
                      cleaning_log: List[str], 
                      transformation_log: List[str]) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify the final dataset quality
        
        Args:
            original_df: Original raw dataframe
            final_df: Final processed dataframe
            cleaning_log: Log of cleaning operations
            transformation_log: Log of transformation operations
            
        Returns:
            Tuple of (success status, verification results)
        """
        self.verification_results = {
            'overall_success': True,
            'data_quality_checks': {},
            'transformation_verification': {},
            'summary': {},
            'warnings': [],
            'errors': []
        }
        
        self.logger.info("🔍 Starting dataset verification...")
        
        # 1. Basic data integrity checks
        self._verify_data_integrity(original_df, final_df)
        
        # 2. Data quality checks
        self._verify_data_quality(final_df)
        
        # 3. Transformation verification
        self._verify_transformations(final_df, transformation_log)
        
        # 4. Business logic validation
        self._verify_business_logic(final_df)
        
        # 5. Generate verification summary
        self._generate_verification_summary(original_df, final_df, cleaning_log, transformation_log)
        
        success = self.verification_results['overall_success']
        if success:
            self.logger.info("✅ Dataset verification PASSED")
        else:
            self.logger.error("❌ Dataset verification FAILED")
        
        return success, self.verification_results
    
    def _verify_data_integrity(self, original_df: pd.DataFrame, final_df: pd.DataFrame):
        """Verify basic data integrity"""
        
        # Check row count (should not increase significantly)
        original_rows = len(original_df)
        final_rows = len(final_df)
        row_change_pct = ((final_rows - original_rows) / original_rows) * 100
        
        self.verification_results['data_quality_checks']['row_count'] = {
            'original': original_rows,
            'final': final_rows,
            'change_pct': row_change_pct,
            'status': 'PASS' if abs(row_change_pct) < 50 else 'WARN'
        }
        
        if abs(row_change_pct) > 50:
            self.verification_results['warnings'].append(f"Significant row count change: {row_change_pct:.1f}%")
        
        # Check for data corruption
        if final_rows == 0:
            self.verification_results['errors'].append("Final dataset is empty")
            self.verification_results['overall_success'] = False
        
        # Memory usage check
        final_memory = final_df.memory_usage(deep=True).sum()
        original_memory = original_df.memory_usage(deep=True).sum()
        memory_change_pct = ((final_memory - original_memory) / original_memory) * 100
        
        self.verification_results['data_quality_checks']['memory_usage'] = {
            'original_mb': original_memory / 1024 / 1024,
            'final_mb': final_memory / 1024 / 1024,
            'change_pct': memory_change_pct
        }
    
    def _verify_data_quality(self, df: pd.DataFrame):
        """Verify data quality metrics"""
        
        # Check for null values
        null_counts = df.isnull().sum()
        total_nulls = null_counts.sum()
        null_pct = (total_nulls / (len(df) * len(df.columns))) * 100
        
        self.verification_results['data_quality_checks']['null_values'] = {
            'total_nulls': int(total_nulls),
            'null_percentage': null_pct,
            'columns_with_nulls': null_counts[null_counts > 0].to_dict(),
            'status': 'PASS' if null_pct < 10 else 'WARN' if null_pct < 25 else 'FAIL'
        }
        
        if null_pct > 25:
            self.verification_results['errors'].append(f"High null percentage: {null_pct:.1f}%")
            self.verification_results['overall_success'] = False
        elif null_pct > 10:
            self.verification_results['warnings'].append(f"Moderate null percentage: {null_pct:.1f}%")
        
        # Check for duplicate rows
        duplicate_count = df.duplicated().sum()
        duplicate_pct = (duplicate_count / len(df)) * 100
        
        self.verification_results['data_quality_checks']['duplicates'] = {
            'count': int(duplicate_count),
            'percentage': duplicate_pct,
            'status': 'PASS' if duplicate_pct < 5 else 'WARN'
        }
        
        if duplicate_pct > 5:
            self.verification_results['warnings'].append(f"Duplicate rows found: {duplicate_count} ({duplicate_pct:.1f}%)")
        
        # Check data types
        type_issues = []
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if object columns have reasonable length
                max_length = df[col].astype(str).str.len().max()
                if max_length > 1000:
                    type_issues.append(f"Column '{col}' has very long text values (max: {max_length})")
        
        self.verification_results['data_quality_checks']['data_types'] = {
            'issues': type_issues,
            'status': 'PASS' if not type_issues else 'WARN'
        }
        
        # Check for infinite values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        inf_counts = {}
        for col in numeric_cols:
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                inf_counts[col] = int(inf_count)
        
        self.verification_results['data_quality_checks']['infinite_values'] = {
            'columns_with_inf': inf_counts,
            'status': 'PASS' if not inf_counts else 'FAIL'
        }
        
        if inf_counts:
            self.verification_results['errors'].append(f"Infinite values found in: {list(inf_counts.keys())}")
            self.verification_results['overall_success'] = False
    
    def _verify_transformations(self, df: pd.DataFrame, transformation_log: List[str]):
        """Verify that transformations were applied correctly"""
        
        # Check if derived columns exist
        derived_columns = [col for col in df.columns if any(word in col.lower() 
                          for word in ['total', 'avg', 'ratio', 'normalized', 'binary', 'label'])]
        
        self.verification_results['transformation_verification']['derived_columns'] = {
            'count': len(derived_columns),
            'columns': derived_columns,
            'status': 'PASS' if derived_columns else 'INFO'
        }
        
        # Check encoding verification
        encoded_columns = [col for col in df.columns if any(word in col.lower() 
                          for word in ['binary', 'label']) or col.endswith('_0') or col.endswith('_1')]
        
        self.verification_results['transformation_verification']['encoded_columns'] = {
            'count': len(encoded_columns),
            'status': 'PASS'
        }
        
        # Verify numeric transformations
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        transformation_success = True
        
        for col in numeric_cols:
            if df[col].std() == 0 and len(df[col].unique()) > 1:
                self.verification_results['warnings'].append(f"Column '{col}' has zero variance but multiple values")
                transformation_success = False
        
        self.verification_results['transformation_verification']['numeric_transformations'] = {
            'status': 'PASS' if transformation_success else 'WARN'
        }
    
    def _verify_business_logic(self, df: pd.DataFrame):
        """Verify business logic specific to healthcare/hospital data"""
        
        business_checks = {}
        
        # Hospital/healthcare specific checks
        bed_columns = [col for col in df.columns if 'bed' in col.lower()]
        if bed_columns:
            for col in bed_columns:
                if df[col].dtype in ['int64', 'float64']:
                    negative_beds = (df[col] < 0).sum()
                    unrealistic_beds = (df[col] > 10000).sum()  # Assuming >10k beds is unrealistic
                    
                    business_checks[f'{col}_validation'] = {
                        'negative_values': int(negative_beds),
                        'unrealistic_values': int(unrealistic_beds),
                        'status': 'PASS' if negative_beds == 0 and unrealistic_beds == 0 else 'WARN'
                    }
        
        # Check for reasonable date ranges
        date_columns = df.select_dtypes(include=['datetime64']).columns
        current_year = pd.Timestamp.now().year
        
        for col in date_columns:
            if 'establish' in col.lower() or 'start' in col.lower():
                future_dates = (df[col].dt.year > current_year).sum()
                very_old_dates = (df[col].dt.year < 1800).sum()
                
                business_checks[f'{col}_validation'] = {
                    'future_dates': int(future_dates),
                    'unrealistic_old_dates': int(very_old_dates),
                    'status': 'PASS' if future_dates == 0 and very_old_dates == 0 else 'WARN'
                }
        
        self.verification_results['data_quality_checks']['business_logic'] = business_checks
    
    def _generate_verification_summary(self, 
                                     original_df: pd.DataFrame, 
                                     final_df: pd.DataFrame, 
                                     cleaning_log: List[str], 
                                     transformation_log: List[str]):
        """Generate overall verification summary"""
        
        summary = {
            'original_shape': original_df.shape,
            'final_shape': final_df.shape,
            'cleaning_operations': len(cleaning_log),
            'transformation_operations': len(transformation_log),
            'total_operations': len(cleaning_log) + len(transformation_log),
            'warnings_count': len(self.verification_results['warnings']),
            'errors_count': len(self.verification_results['errors']),
            'overall_status': 'PASS' if self.verification_results['overall_success'] else 'FAIL'
        }
        
        # Calculate quality score
        quality_score = 100
        quality_score -= len(self.verification_results['errors']) * 20
        quality_score -= len(self.verification_results['warnings']) * 5
        quality_score = max(0, quality_score)
        
        summary['quality_score'] = quality_score
        summary['quality_grade'] = (
            'A' if quality_score >= 90 else
            'B' if quality_score >= 80 else
            'C' if quality_score >= 70 else
            'D' if quality_score >= 60 else 'F'
        )
        
        self.verification_results['summary'] = summary
