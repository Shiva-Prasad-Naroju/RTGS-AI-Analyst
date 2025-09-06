import pandas as pd
import numpy as np
from typing import Dict, List, Any
import logging
import warnings

class InspectorAgent:
    """Agent responsible for inspecting dataset and generating action plan"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Suppress pandas warnings
        warnings.filterwarnings('ignore', category=UserWarning, module='pandas')
        
    def inspect_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive dataset inspection
        
        Args:
            df: Raw dataframe to inspect
            
        Returns:
            Action plan dictionary
        """
        action_plan = {
            'null_values': {},
            'duplicates': None,
            'column_types': {},
            'categorical_consistency': {},
            'outliers': {},
            'range_checks': {},
            'metadata_issues': {},
            'datetime_validity': {},
            'unique_ids': {},
            'summary': {}
        }
        
        self.logger.info("🔍 Starting dataset inspection...")
        
        # Basic info
        action_plan['summary'] = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'memory_usage': df.memory_usage(deep=True).sum()
        }
        
        # Check null values
        null_counts = df.isnull().sum()
        for col in null_counts[null_counts > 0].index:
            null_pct = (null_counts[col] / len(df)) * 100
            if df[col].dtype in ['int64', 'float64']:
                action_plan['null_values'][col] = f"Impute with median ({null_pct:.1f}% missing)"
            else:
                action_plan['null_values'][col] = f"Impute with mode/UNKNOWN ({null_pct:.1f}% missing)"
        
        # Check duplicates
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            action_plan['duplicates'] = f"Remove {duplicate_count} duplicate rows"
        
        # Check column types and suggest fixes
        for col in df.columns:
            dtype = str(df[col].dtype)
            if dtype == 'object':
                # Check if it should be numeric
                if self._could_be_numeric(df[col]):
                    action_plan['column_types'][col] = "Convert to numeric"
                elif self._could_be_datetime(df[col]):
                    action_plan['column_types'][col] = "Convert to datetime"
                else:
                    # Check categorical consistency
                    inconsistencies = self._check_categorical_consistency(df[col])
                    if inconsistencies:
                        action_plan['categorical_consistency'][col] = f"Standardize: {inconsistencies}"
        
        # Check for outliers in numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            outliers = self._detect_outliers(df[col])
            if outliers['count'] > 0:
                action_plan['outliers'][col] = outliers
                
            # Range checks for specific column patterns
            if 'bed' in col.lower() or 'capacity' in col.lower():
                negative_count = (df[col] < 0).sum()
                if negative_count > 0:
                    action_plan['range_checks'][col] = f"Fix {negative_count} negative values"
        
        # Check column naming issues
        for col in df.columns:
            issues = []
            if ' ' in col:
                issues.append("Contains spaces")
            if col != col.strip():
                issues.append("Has leading/trailing spaces")
            if any(char in col for char in ['(', ')', '[', ']', '/', '\\']):
                issues.append("Contains special characters")
            if issues:
                suggested_name = self._suggest_column_name(col)
                action_plan['metadata_issues'][col] = f"Rename to '{suggested_name}': {', '.join(issues)}"
        
        # Check for potential ID columns
        for col in df.columns:
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio > 0.95:
                action_plan['unique_ids'][col] = f"Potential unique ID ({unique_ratio:.1%} unique values)"
        
        self._log_inspection_summary(action_plan)
        return action_plan
    
    def _could_be_numeric(self, series: pd.Series) -> bool:
        """Check if string column could be converted to numeric"""
        non_null = series.dropna().astype(str)
        if len(non_null) == 0:
            return False
        
        # Remove common non-numeric chars and check
        cleaned = non_null.str.replace(r'[,\s₹$%]', '', regex=True)
        try:
            pd.to_numeric(cleaned, errors='raise')
            return True
        except:
            return False
    
    def _could_be_datetime(self, series: pd.Series) -> bool:
        """Check if column could be datetime - FIXED VERSION"""
        non_null = series.dropna().astype(str)
        if len(non_null) == 0:
            return False
            
        # Sample a few values to test
        sample = non_null.head(min(10, len(non_null)))
        try:
            # Suppress warnings and use simpler datetime detection
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pd.to_datetime(sample, errors='raise')
            return True
        except:
            # Try common date patterns
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # Try specific formats
                    for fmt in ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%Y', '%d-%m-%Y']:
                        try:
                            pd.to_datetime(sample, format=fmt, errors='raise')
                            return True
                        except:
                            continue
            except:
                pass
            return False
    
    def _check_categorical_consistency(self, series: pd.Series) -> List[str]:
        """Check for inconsistent categorical values"""
        if series.dtype != 'object':
            return []
            
        unique_vals = series.dropna().unique()
        if len(unique_vals) > 20:  # Too many unique values
            return []
            
        inconsistencies = []
        val_groups = {}
        
        for val in unique_vals:
            val_lower = str(val).lower().strip()
            if val_lower not in val_groups:
                val_groups[val_lower] = []
            val_groups[val_lower].append(val)
        
        for group_vals in val_groups.values():
            if len(group_vals) > 1:
                inconsistencies.append(f"{group_vals}")
        
        return inconsistencies
    
    def _detect_outliers(self, series: pd.Series) -> Dict[str, Any]:
        """Detect outliers using IQR method"""
        if series.dtype not in ['int64', 'float64']:
            return {'count': 0}
            
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = series[(series < lower_bound) | (series > upper_bound)]
        
        return {
            'count': len(outliers),
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'extreme_values': outliers.tolist()[:5]  # Show first 5
        }
    
    def _suggest_column_name(self, col: str) -> str:
        """Suggest a clean column name"""
        # Remove special chars, replace spaces with underscores
        clean = col.strip()
        clean = ''.join(c if c.isalnum() or c == '_' else '_' for c in clean)
        clean = clean.replace(' ', '_')
        # Remove consecutive underscores
        while '__' in clean:
            clean = clean.replace('__', '_')
        clean = clean.strip('_')
        return clean
    
    def _log_inspection_summary(self, action_plan: Dict[str, Any]):
        """Log summary of inspection results"""
        summary = action_plan['summary']
        self.logger.info(f"📊 Dataset Summary: {summary['total_rows']} rows, {summary['total_columns']} columns")
        
        issues_found = []
        if action_plan['null_values']:
            issues_found.append(f"{len(action_plan['null_values'])} columns with nulls")
        if action_plan['duplicates']:
            issues_found.append("duplicate rows")
        if action_plan['column_types']:
            issues_found.append(f"{len(action_plan['column_types'])} type conversions needed")
        if action_plan['categorical_consistency']:
            issues_found.append(f"{len(action_plan['categorical_consistency'])} categorical inconsistencies")
        if action_plan['outliers']:
            issues_found.append(f"{len(action_plan['outliers'])} columns with outliers")
        
        if issues_found:
            self.logger.info(f"⚠️ Issues found: {', '.join(issues_found)}")
        else:
            self.logger.info("✅ No major issues detected")