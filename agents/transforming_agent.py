import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import logging

class TransformingAgent:
    """Agent responsible for transforming cleaned data for analysis"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.transformation_log = []
    
    def transform_dataset(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Transform cleaned dataset for analysis
        
        Args:
            df: Cleaned dataframe
            
        Returns:
            Tuple of (transformed dataframe, transformation logs)
        """
        df_transformed = df.copy()
        self.transformation_log = []
        
        self.logger.info("⚡ Starting data transformation process...")
        
        # 1. Create derived metrics
        df_transformed = self._create_derived_metrics(df_transformed)
        
        # 2. Handle datetime transformations
        df_transformed = self._transform_datetime_columns(df_transformed)
        
        # 3. Encode categorical variables
        df_transformed = self._encode_categorical_variables(df_transformed)
        
        # 4. Create aggregated features
        df_transformed = self._create_aggregated_features(df_transformed)
        
        # 5. Normalize/standardize if needed
        df_transformed = self._normalize_features(df_transformed)
        
        # 6. Reorder and organize columns
        df_transformed = self._organize_columns(df_transformed)
        
        self.logger.info(f"✅ Transformation completed: {len(self.transformation_log)} transformations applied")
        return df_transformed, self.transformation_log
    
    def _create_derived_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived metrics based on existing columns"""
        
        # Hospital/healthcare specific metrics
        bed_columns = [col for col in df.columns if 'bed' in col.lower()]
        capacity_columns = [col for col in df.columns if 'capacity' in col.lower()]
        
        if len(bed_columns) > 1:
            # Total beds
            df['Total_Beds'] = df[bed_columns].sum(axis=1)
            self.transformation_log.append(f"Created 'Total_Beds' from: {bed_columns}")
            
        # If we have district and hospital count data
        if 'District' in df.columns or 'district' in [col.lower() for col in df.columns]:
            district_col = next((col for col in df.columns if 'district' in col.lower()), None)
            if district_col:
                district_counts = df.groupby(district_col).size()
                df['Hospitals_in_District'] = df[district_col].map(district_counts)
                self.transformation_log.append("Created 'Hospitals_in_District' metric")
        
        # Calculate beds per hospital if we have aggregate data
        if 'Total_Beds' in df.columns and len(df) > 1:
            hospital_count = len(df)
            total_beds = df['Total_Beds'].sum()
            if total_beds > 0:
                df['Avg_Beds_Per_Hospital'] = total_beds / hospital_count
                self.transformation_log.append(f"Created 'Avg_Beds_Per_Hospital' metric ({total_beds/hospital_count:.2f})")
        
        # Healthcare type ratios
        type_columns = [col for col in df.columns if any(word in col.lower() for word in ['type', 'ownership', 'category'])]
        if type_columns:
            type_col = type_columns[0]
            type_counts = df[type_col].value_counts()
            total = len(df)
            for type_val in type_counts.index:
                ratio = type_counts[type_val] / total
                df[f'{type_val}_Ratio'] = ratio
            self.transformation_log.append(f"Created ratio metrics for {type_col}")
        
        return df
    
    def _transform_datetime_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform datetime columns to extract useful features"""
        datetime_cols = df.select_dtypes(include=['datetime64']).columns
        
        for col in datetime_cols:
            try:
                # Extract year, month, quarter
                df[f'{col}_Year'] = df[col].dt.year
                df[f'{col}_Month'] = df[col].dt.month
                df[f'{col}_Quarter'] = df[col].dt.quarter
                df[f'{col}_DayOfWeek'] = df[col].dt.day_name()
                
                # Calculate age/years since
                current_year = pd.Timestamp.now().year
                if 'establish' in col.lower() or 'start' in col.lower():
                    df[f'Years_Since_{col}'] = current_year - df[col].dt.year
                
                self.transformation_log.append(f"Extracted datetime features from '{col}'")
                
            except Exception as e:
                self.logger.warning(f"Failed to transform datetime column {col}: {e}")
        
        return df
    
    def _encode_categorical_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables for analysis"""
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            unique_vals = df[col].nunique()
            
            # Skip if too many unique values (likely text fields)
            if unique_vals > 20:
                continue
                
            try:
                if unique_vals == 2:
                    # Binary encoding
                    unique_list = df[col].unique()
                    df[f'{col}_Binary'] = df[col].map({unique_list[0]: 0, unique_list[1]: 1})
                    self.transformation_log.append(f"Binary encoded '{col}': {unique_list[0]}=0, {unique_list[1]}=1")
                    
                elif unique_vals <= 10:
                    # One-hot encoding for small categories
                    dummies = pd.get_dummies(df[col], prefix=col)
                    df = pd.concat([df, dummies], axis=1)
                    self.transformation_log.append(f"One-hot encoded '{col}' into {len(dummies.columns)} columns")
                else:
                    # Label encoding for medium categories
                    df[f'{col}_Label'] = pd.Categorical(df[col]).codes
                    self.transformation_log.append(f"Label encoded '{col}' ({unique_vals} categories)")
                    
            except Exception as e:
                self.logger.warning(f"Failed to encode categorical column {col}: {e}")
        
        return df
    
    def _create_aggregated_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create aggregated features by groups"""
        
        # Find potential grouping columns
        grouping_cols = []
        for col in df.columns:
            if any(word in col.lower() for word in ['district', 'state', 'region', 'type', 'category']):
                if df[col].nunique() < len(df) * 0.5:  # Not too many unique values
                    grouping_cols.append(col)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for group_col in grouping_cols[:2]:  # Limit to first 2 to avoid too many features
            for num_col in numeric_cols[:3]:  # Limit to first 3 numeric columns
                try:
                    # Group statistics
                    group_stats = df.groupby(group_col)[num_col].agg(['mean', 'std', 'count'])
                    
                    df[f'{num_col}_Mean_by_{group_col}'] = df[group_col].map(group_stats['mean'])
                    df[f'{num_col}_Std_by_{group_col}'] = df[group_col].map(group_stats['std'])
                    
                    self.transformation_log.append(f"Created aggregated features for '{num_col}' by '{group_col}'")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to create aggregated features for {num_col} by {group_col}: {e}")
        
        return df
    
    def _normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize/standardize numeric features if needed"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Only normalize if we have multiple numeric columns with different scales
        if len(numeric_cols) > 1:
            scales = {}
            for col in numeric_cols:
                if df[col].std() > 0:
                    scales[col] = df[col].std()
            
            # If scales differ significantly, normalize
            max_scale = max(scales.values()) if scales else 1
            min_scale = min(scales.values()) if scales else 1
            
            if max_scale / min_scale > 100:  # Significant scale difference
                for col in numeric_cols:
                    if df[col].std() > 0:
                        df[f'{col}_Normalized'] = (df[col] - df[col].mean()) / df[col].std()
                
                self.transformation_log.append(f"Normalized {len(numeric_cols)} numeric columns due to scale differences")
        
        return df
    
    def _organize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Organize columns in logical order"""
        
        # Define column order priorities
        id_cols = [col for col in df.columns if any(word in col.lower() for word in ['id', 'key', 'index'])]
        name_cols = [col for col in df.columns if any(word in col.lower() for word in ['name', 'title'])]
        location_cols = [col for col in df.columns if any(word in col.lower() for word in ['district', 'state', 'city', 'address', 'pin'])]
        type_cols = [col for col in df.columns if any(word in col.lower() for word in ['type', 'category', 'classification'])]
        metric_cols = [col for col in df.columns if any(word in col.lower() for word in ['bed', 'capacity', 'count', 'total'])]
        derived_cols = [col for col in df.columns if any(word in col.lower() for word in ['ratio', 'avg', 'mean', 'normalized'])]
        date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Organize column order
        organized_order = []
        for col_group in [id_cols, name_cols, location_cols, type_cols, metric_cols, date_cols, derived_cols]:
            for col in col_group:
                if col in df.columns and col not in organized_order:
                    organized_order.append(col)
        
        # Add remaining columns
        for col in df.columns:
            if col not in organized_order:
                organized_order.append(col)
        
        df = df[organized_order]
        self.transformation_log.append(f"Organized columns in logical order: {len(organized_order)} columns")
        
        return df
