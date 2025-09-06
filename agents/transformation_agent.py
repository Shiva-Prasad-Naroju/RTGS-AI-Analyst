"""
Transformation Agent - Applies basic transformations for analysis readiness
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer

logger = logging.getLogger(__name__)

class TransformationAgent:
    """Agent responsible for basic data transformations"""
    
    def __init__(self):
        self.name = "Transformation Agent"
        self.transformations = []
        self.encoders = {}
        self.scalers = {}
        
    def encode_categorical_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables for analysis"""
        df_transformed = df.copy()
        
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        logger.info(f"{self.name}: Encoding {len(categorical_cols)} categorical columns...")
        
        for col in categorical_cols:
            unique_count = df[col].nunique()
            
            # Skip if too many unique values (likely identifiers)
            if unique_count > 50:
                logger.warning(f"Skipping encoding for '{col}' - too many unique values ({unique_count})")
                continue
            
            # For binary categorical variables
            if unique_count == 2:
                # Simple binary encoding
                unique_values = df[col].dropna().unique()
                if len(unique_values) == 2:
                    df_transformed[f'{col}_encoded'] = df_transformed[col].map({
                        unique_values[0]: 0,
                        unique_values[1]: 1
                    })
                    self.transformations.append(f"Binary encoded '{col}' -> '{col}_encoded'")
                    logger.info(f"Binary encoded column '{col}'")
            
            # For low cardinality categorical variables (<=10 categories)
            elif unique_count <= 10:
                # One-hot encoding
                dummies = pd.get_dummies(df_transformed[col], prefix=col, dummy_na=True)
                df_transformed = pd.concat([df_transformed, dummies], axis=1)
                self.transformations.append(f"One-hot encoded '{col}' into {len(dummies.columns)} dummy variables")
                logger.info(f"One-hot encoded column '{col}' into {len(dummies.columns)} columns")
            
            # For medium cardinality (11-50 categories)
            else:
                # Label encoding
                le = LabelEncoder()
                non_null_mask = df_transformed[col].notna()
                df_transformed.loc[non_null_mask, f'{col}_encoded'] = le.fit_transform(
                    df_transformed.loc[non_null_mask, col]
                )
                self.encoders[col] = le
                self.transformations.append(f"Label encoded '{col}' -> '{col}_encoded'")
                logger.info(f"Label encoded column '{col}'")
        
        return df_transformed
    
    def scale_numeric_variables(self, df: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
        """Scale numeric variables"""
        df_transformed = df.copy()
        
        # Get numeric columns (excluding encoded columns)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if not col.endswith('_encoded')]
        
        if len(numeric_cols) == 0:
            logger.info(f"{self.name}: No numeric columns to scale")
            return df_transformed
        
        logger.info(f"{self.name}: Scaling {len(numeric_cols)} numeric columns using {method} scaling...")
        
        # Choose scaler
        if method == 'standard':
            scaler = StandardScaler()
            suffix = '_std'
        elif method == 'minmax':
            scaler = MinMaxScaler()
            suffix = '_norm'
        else:
            logger.warning(f"Unknown scaling method: {method}, using standard scaling")
            scaler = StandardScaler()
            suffix = '_std'
        
        # Apply scaling
        for col in numeric_cols:
            if df[col].notna().sum() > 0:  # Only scale if we have non-null values
                non_null_mask = df_transformed[col].notna()
                scaled_values = scaler.fit_transform(df_transformed.loc[non_null_mask, [col]])
                df_transformed.loc[non_null_mask, f'{col}{suffix}'] = scaled_values.ravel()
                
                self.scalers[col] = scaler
                self.transformations.append(f"Applied {method} scaling to '{col}' -> '{col}{suffix}'")
                logger.info(f"Scaled column '{col}' using {method} scaling")
        
        return df_transformed
    
    def handle_skewed_distributions(self, df: pd.DataFrame, skewness_threshold: float = 2.0) -> pd.DataFrame:
        """Apply log transformation to highly skewed numeric variables"""
        df_transformed = df.copy()
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if not col.endswith(('_encoded', '_std', '_norm'))]
        
        logger.info(f"{self.name}: Checking skewness for {len(numeric_cols)} numeric columns...")
        
        for col in numeric_cols:
            if df[col].notna().sum() < 3:  # Need at least 3 values
                continue
                
            # Calculate skewness
            skewness = df[col].skew()
            
            if abs(skewness) > skewness_threshold:
                # Apply log transformation (add 1 to handle zeros)
                if df[col].min() >= 0:  # Only for non-negative values
                    df_transformed[f'{col}_log'] = np.log1p(df_transformed[col])
                    self.transformations.append(f"Applied log transformation to '{col}' (skewness: {skewness:.2f}) -> '{col}_log'")
                    logger.info(f"Applied log transformation to '{col}' (skewness: {skewness:.2f})")
                else:
                    logger.warning(f"Skipped log transformation for '{col}' - contains negative values")
        
        return df_transformed
    
    def create_datetime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from datetime columns"""
        df_transformed = df.copy()
        
        datetime_cols = df.select_dtypes(include=['datetime64']).columns
        
        if len(datetime_cols) == 0:
            return df_transformed
        
        logger.info(f"{self.name}: Creating datetime features for {len(datetime_cols)} columns...")
        
        for col in datetime_cols:
            if df[col].notna().sum() == 0:
                continue
            
            # Extract common datetime features
            df_transformed[f'{col}_year'] = df_transformed[col].dt.year
            df_transformed[f'{col}_month'] = df_transformed[col].dt.month
            df_transformed[f'{col}_day'] = df_transformed[col].dt.day
            df_transformed[f'{col}_dayofweek'] = df_transformed[col].dt.dayofweek
            df_transformed[f'{col}_quarter'] = df_transformed[col].dt.quarter
            
            # Create binary features for common patterns
            df_transformed[f'{col}_is_weekend'] = df_transformed[f'{col}_dayofweek'].isin([5, 6]).astype(int)
            df_transformed[f'{col}_is_month_start'] = df_transformed[col].dt.is_month_start.astype(int)
            df_transformed[f'{col}_is_month_end'] = df_transformed[col].dt.is_month_end.astype(int)
            
            features_created = 8
            self.transformations.append(f"Created {features_created} datetime features from '{col}'")
            logger.info(f"Created {features_created} datetime features from '{col}'")
        
        return df_transformed
    
    def create_interaction_features(self, df: pd.DataFrame, max_interactions: int = 5) -> pd.DataFrame:
        """Create simple interaction features between numeric variables"""
        df_transformed = df.copy()
        
        # Get original numeric columns (not transformed ones)
        numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                       if not col.endswith(('_encoded', '_std', '_norm', '_log'))]
        
        if len(numeric_cols) < 2:
            logger.info(f"{self.name}: Not enough numeric columns for interaction features")
            return df_transformed
        
        logger.info(f"{self.name}: Creating interaction features...")
        
        interactions_created = 0
        for i, col1 in enumerate(numeric_cols):
            if interactions_created >= max_interactions:
                break
            for j, col2 in enumerate(numeric_cols[i+1:], i+1):
                if interactions_created >= max_interactions:
                    break
                
                # Create ratio feature
                col2_nonzero = df_transformed[col2] != 0
                if col2_nonzero.sum() > 0:
                    df_transformed[f'{col1}_{col2}_ratio'] = df_transformed[col1] / df_transformed[col2].where(col2_nonzero, np.nan)
                    interactions_created += 1
                    self.transformations.append(f"Created ratio feature '{col1}_{col2}_ratio'")
                
                if interactions_created >= max_interactions:
                    break
        
        logger.info(f"Created {interactions_created} interaction features")
        return df_transformed
    
    def handle_missing_after_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle any missing values created during transformation"""
        df_transformed = df.copy()
        
        # Check for new missing values in transformed columns
        transformed_cols = [col for col in df_transformed.columns 
                          if col.endswith(('_encoded', '_std', '_norm', '_log')) or 
                             '_ratio' in col or any(dt_feature in col for dt_feature in ['_year', '_month', '_day'])]
        
        for col in transformed_cols:
            missing_count = df_transformed[col].isnull().sum()
            if missing_count > 0:
                if pd.api.types.is_numeric_dtype(df_transformed[col]):
                    # Fill with median for numeric
                    median_val = df_transformed[col].median()
                    df_transformed[col] = df_transformed[col].fillna(median_val)
                    self.transformations.append(f"Filled {missing_count} missing values in '{col}' with median")
                else:
                    # Fill with mode for categorical
                    mode_val = df_transformed[col].mode().iloc[0] if not df_transformed[col].mode().empty else 0
                    df_transformed[col] = df_transformed[col].fillna(mode_val)
                    self.transformations.append(f"Filled {missing_count} missing values in '{col}' with mode")
        
        return df_transformed
    
    def validate_transformations(self, df_original: pd.DataFrame, df_transformed: pd.DataFrame) -> Dict[str, Any]:
        """Validate transformation results"""
        validation = {
            'is_valid': True,
            'issues': [],
            'warnings': [],
            'summary': {}
        }
        
        # Check if we created too many columns
        original_cols = df_original.shape[1]
        transformed_cols = df_transformed.shape[1]
        new_cols = transformed_cols - original_cols
        
        validation['summary'] = {
            'original_columns': original_cols,
            'transformed_columns': transformed_cols,
            'new_columns_created': new_cols,
            'transformations_applied': len(self.transformations)
        }
        
        if new_cols > original_cols * 2:  # More than double the columns
            validation['warnings'].append(f"Created many new columns ({new_cols}), consider feature selection")
        
        # Check for infinite values
        numeric_cols = df_transformed.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if np.isinf(df_transformed[col]).any():
                validation['issues'].append(f"Column '{col}' contains infinite values")
                validation['is_valid'] = False
        
        # Check for extremely large values (potential overflow)
        for col in numeric_cols:
            if df_transformed[col].abs().max() > 1e10:
                validation['warnings'].append(f"Column '{col}' has very large values")
        
        return validation
    
    def process(self, df: pd.DataFrame, apply_scaling: bool = True, 
                handle_skewness: bool = True, create_features: bool = True) -> Dict[str, Any]:
        """
        Main processing method for the Transformation Agent
        
        Args:
            df: DataFrame to transform
            apply_scaling: Whether to apply scaling to numeric variables
            handle_skewness: Whether to handle skewed distributions
            create_features: Whether to create additional features
            
        Returns:
            Transformation results with transformed DataFrame
        """
        logger.info(f"{self.name}: Starting data transformation process")
        self.transformations = []
        
        try:
            df_transformed = df.copy()
            original_shape = df.shape
            
            # Step 1: Encode categorical variables
            logger.info(f"{self.name}: Step 1 - Encoding categorical variables...")
            df_transformed = self.encode_categorical_variables(df_transformed)
            
            # Step 2: Scale numeric variables (if requested)
            if apply_scaling:
                logger.info(f"{self.name}: Step 2 - Scaling numeric variables...")
                df_transformed = self.scale_numeric_variables(df_transformed, method='standard')
            
            # Step 3: Handle skewed distributions (if requested)
            if handle_skewness:
                logger.info(f"{self.name}: Step 3 - Handling skewed distributions...")
                df_transformed = self.handle_skewed_distributions(df_transformed)
            
            # Step 4: Create datetime features (if applicable)
            if create_features:
                logger.info(f"{self.name}: Step 4 - Creating datetime features...")
                df_transformed = self.create_datetime_features(df_transformed)
                
                # Step 5: Create simple interaction features
                logger.info(f"{self.name}: Step 5 - Creating interaction features...")
                df_transformed = self.create_interaction_features(df_transformed)
            
            # Step 6: Handle any missing values created during transformation
            logger.info(f"{self.name}: Step 6 - Handling post-transformation missing values...")
            df_transformed = self.handle_missing_after_transformation(df_transformed)
            
            # Validate transformations
            validation = self.validate_transformations(df, df_transformed)
            
            final_shape = df_transformed.shape
            
            result = {
                'agent': self.name,
                'status': 'success',
                'data': df_transformed,
                'original_data': df,
                'transformations': self.transformations,
                'encoders': self.encoders,
                'scalers': self.scalers,
                'validation': validation,
                'shape_change': {
                    'before': original_shape,
                    'after': final_shape,
                    'columns_added': final_shape[1] - original_shape[1]
                },
                'message': f"Transformation completed. Applied {len(self.transformations)} transformations, shape: {original_shape} -> {final_shape}"
            }
            
            logger.info(f"{self.name}: Transformation completed successfully")
            logger.info(f"  - Transformations applied: {len(self.transformations)}")
            logger.info(f"  - Shape change: {original_shape} -> {final_shape}")
            
            return result
            
        except Exception as e:
            error_msg = f"Transformation failed: {str(e)}"
            logger.error(f"{self.name}: {error_msg}")
            
            return {
                'agent': self.name,
                'status': 'error',
                'data': df,
                'transformations': self.transformations,
                'message': error_msg,
                'error': str(e)
            }