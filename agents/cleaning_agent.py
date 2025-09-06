import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import logging
import re

class CleaningAgent:
    """Agent responsible for cleaning the dataset based on action plan"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.cleaning_log = []
    
    def clean_dataset(self, df: pd.DataFrame, action_plan: Dict[str, Any]) -> Tuple[pd.DataFrame, List[str]]:
        """
        Clean dataset according to action plan
        
        Args:
            df: Raw dataframe
            action_plan: Action plan from inspector
            
        Returns:
            Tuple of (cleaned dataframe, cleaning logs)
        """
        df_clean = df.copy()
        self.cleaning_log = []
        
        self.logger.info("🧹 Starting data cleaning process...")
        
        # 1. Handle metadata issues (column renaming)
        df_clean = self._fix_metadata_issues(df_clean, action_plan.get('metadata_issues', {}))
        
        # 2. Handle column type conversions
        df_clean = self._fix_column_types(df_clean, action_plan.get('column_types', {}))
        
        # 3. Handle null values
        df_clean = self._handle_null_values(df_clean, action_plan.get('null_values', {}))
        
        # 4. Remove duplicates
        df_clean = self._remove_duplicates(df_clean, action_plan.get('duplicates'))
        
        # 5. Fix categorical consistency
        df_clean = self._fix_categorical_consistency(df_clean, action_plan.get('categorical_consistency', {}))
        
        # 6. Handle outliers
        df_clean = self._handle_outliers(df_clean, action_plan.get('outliers', {}))
        
        # 7. Apply range checks
        df_clean = self._apply_range_checks(df_clean, action_plan.get('range_checks', {}))
        
        self.logger.info(f"✅ Cleaning completed: {len(self.cleaning_log)} actions performed")
        return df_clean, self.cleaning_log
    
    def _fix_metadata_issues(self, df: pd.DataFrame, metadata_issues: Dict[str, str]) -> pd.DataFrame:
        """Fix column naming issues"""
        if not metadata_issues:
            return df
            
        column_mapping = {}
        for old_col, issue_desc in metadata_issues.items():
            if "Rename to" in issue_desc:
                new_name = issue_desc.split("'")[1]
                column_mapping[old_col] = new_name
                self.cleaning_log.append(f"Renamed column '{old_col}' to '{new_name}'")
        
        if column_mapping:
            df = df.rename(columns=column_mapping)
            self.logger.info(f"📝 Renamed {len(column_mapping)} columns")
        
        return df
    
    def _fix_column_types(self, df: pd.DataFrame, type_issues: Dict[str, str]) -> pd.DataFrame:
        """Fix column type issues"""
        for col, action in type_issues.items():
            if col not in df.columns:
                continue
                
            try:
                if "Convert to numeric" in action:
                    # Clean and convert to numeric
                    df[col] = df[col].astype(str).str.replace(r'[,\s₹$%]', '', regex=True)
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    self.cleaning_log.append(f"Converted '{col}' to numeric")
                    
                elif "Convert to datetime" in action:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    self.cleaning_log.append(f"Converted '{col}' to datetime")
                    
                elif "Convert to int" in action:
                    df[col] = df[col].fillna(0).astype(int)
                    self.cleaning_log.append(f"Converted '{col}' to integer")
                    
            except Exception as e:
                self.logger.warning(f"Failed to convert {col}: {e}")
        
        return df
    
    def _handle_null_values(self, df: pd.DataFrame, null_issues: Dict[str, str]) -> pd.DataFrame:
        """Handle null values"""
        for col, action in null_issues.items():
            if col not in df.columns:
                continue
                
            null_count = df[col].isnull().sum()
            if null_count == 0:
                continue
                
            try:
                if "median" in action.lower():
                    fill_value = df[col].median()
                    df[col] = df[col].fillna(fill_value)
                    self.cleaning_log.append(f"Filled {null_count} null values in '{col}' with median ({fill_value})")
                    
                elif "mode" in action.lower():
                    fill_value = df[col].mode().iloc[0] if not df[col].mode().empty else "UNKNOWN"
                    df[col] = df[col].fillna(fill_value)
                    self.cleaning_log.append(f"Filled {null_count} null values in '{col}' with mode ('{fill_value}')")
                    
                elif "unknown" in action.lower():
                    df[col] = df[col].fillna("UNKNOWN")
                    self.cleaning_log.append(f"Filled {null_count} null values in '{col}' with 'UNKNOWN'")
                    
            except Exception as e:
                self.logger.warning(f"Failed to handle nulls in {col}: {e}")
        
        return df
    
    def _remove_duplicates(self, df: pd.DataFrame, duplicate_action: str) -> pd.DataFrame:
        """Remove duplicate rows"""
        if not duplicate_action:
            return df
            
        initial_rows = len(df)
        df = df.drop_duplicates()
        removed_count = initial_rows - len(df)
        
        if removed_count > 0:
            self.cleaning_log.append(f"Removed {removed_count} duplicate rows")
            self.logger.info(f"🗑️ Removed {removed_count} duplicate rows")
        
        return df
    
    def _fix_categorical_consistency(self, df: pd.DataFrame, consistency_issues: Dict[str, str]) -> pd.DataFrame:
        """Fix categorical value inconsistencies"""
        for col, issue_desc in consistency_issues.items():
            if col not in df.columns:
                continue
                
            try:
                # Extract inconsistent groups from description
                groups = re.findall(r'\[(.*?)\]', issue_desc)
                
                for group in groups:
                    values = [v.strip().strip("'\"") for v in group.split(',')]
                    if len(values) > 1:
                        # Use the most common value as the standard
                        value_counts = df[col].value_counts()
                        standard_value = max(values, key=lambda x: value_counts.get(x, 0))
                        
                        for val in values:
                            if val != standard_value:
                                df[col] = df[col].replace(val, standard_value)
                        
                        self.cleaning_log.append(f"Standardized categorical values in '{col}': {values} → '{standard_value}'")
                        
            except Exception as e:
                self.logger.warning(f"Failed to fix consistency in {col}: {e}")
        
        return df
    
    def _handle_outliers(self, df: pd.DataFrame, outlier_issues: Dict[str, Dict]) -> pd.DataFrame:
        """Handle outliers"""
        for col, outlier_info in outlier_issues.items():
            if col not in df.columns or outlier_info['count'] == 0:
                continue
                
            try:
                # Clip outliers to bounds
                lower_bound = outlier_info['lower_bound']
                upper_bound = outlier_info['upper_bound']
                
                outliers_before = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                df[col] = df[col].clip(lower_bound, upper_bound)
                
                self.cleaning_log.append(f"Clipped {outliers_before} outliers in '{col}' to range [{lower_bound:.2f}, {upper_bound:.2f}]")
                
            except Exception as e:
                self.logger.warning(f"Failed to handle outliers in {col}: {e}")
        
        return df
    
    def _apply_range_checks(self, df: pd.DataFrame, range_issues: Dict[str, str]) -> pd.DataFrame:
        """Apply range checks and fixes"""
        for col, issue_desc in range_issues.items():
            if col not in df.columns:
                continue
                
            try:
                if "negative values" in issue_desc.lower():
                    negative_count = (df[col] < 0).sum()
                    if negative_count > 0:
                        df[col] = df[col].abs()  # Convert to absolute value
                        self.cleaning_log.append(f"Fixed {negative_count} negative values in '{col}' by taking absolute value")
                        
            except Exception as e:
                self.logger.warning(f"Failed to apply range check for {col}: {e}")
        
        return df