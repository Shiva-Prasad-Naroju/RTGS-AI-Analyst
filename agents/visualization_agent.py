"""
Visualization Agent - Creates comparison charts and exports to PDF
Fixed version that handles different column counts between datasets
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, Any
import os
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime

logger = logging.getLogger(__name__)

class VisualizationAgent:
    """Agent responsible for creating visualizations and charts"""
    
    def __init__(self):
        self.name = "Visualization Agent"
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def process(self, df_original: pd.DataFrame, df_cleaned: pd.DataFrame, 
                analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Main processing method"""
        logger.info(f"{self.name}: Starting visualization creation")
        
        try:
            # Create output path
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = 'outputs/charts'
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"data_quality_report_{timestamp}.pdf")
            
            # Export visualizations
            self.export_to_pdf(output_path, analysis_results, df_original, df_cleaned)
            
            file_size = os.path.getsize(output_path) / (1024 * 1024)
            
            return {
                'agent': self.name,
                'status': 'success',
                'pdf_path': output_path,
                'charts_created': 6,
                'file_size_mb': round(file_size, 2),
                'message': f"Visualization PDF created successfully"
            }
            
        except Exception as e:
            logger.error(f"{self.name}: Visualization failed: {str(e)}")
            return {
                'agent': self.name,
                'status': 'error',
                'message': f"Visualization failed: {str(e)}"
            }
    
    def export_to_pdf(self, output_path: str, analysis_results: Dict[str, Any], 
                     df_original: pd.DataFrame, df_cleaned: pd.DataFrame):
        """Export all visualizations to PDF"""
        
        with PdfPages(output_path) as pdf:
            # Page 1: Summary
            fig1 = self.create_summary_infographic(analysis_results, df_original, df_cleaned)
            pdf.savefig(fig1, bbox_inches='tight')
            plt.close(fig1)
            
            # Page 2: Shape and Quality Comparison
            fig2 = self.create_shape_quality_comparison(df_original, df_cleaned, analysis_results)
            pdf.savefig(fig2, bbox_inches='tight')
            plt.close(fig2)
            
            # Page 3: Missing Data (only common columns)
            fig3 = self.create_missing_data_comparison(df_original, df_cleaned)
            pdf.savefig(fig3, bbox_inches='tight')
            plt.close(fig3)
            
            # Page 4: Numeric Distributions (only common columns)
            fig4 = self.create_distribution_comparison(df_original, df_cleaned)
            pdf.savefig(fig4, bbox_inches='tight')
            plt.close(fig4)
            
            # Page 5: Data Types Comparison
            fig5 = self.create_datatypes_comparison(df_original, df_cleaned)
            pdf.savefig(fig5, bbox_inches='tight')
            plt.close(fig5)
            
            # Page 6: Transformation Summary
            fig6 = self.create_transformation_summary(df_original, df_cleaned, analysis_results)
            pdf.savefig(fig6, bbox_inches='tight')
            plt.close(fig6)
    
    def create_summary_infographic(self, analysis_results: Dict, 
                                  df_original: pd.DataFrame, 
                                  df_cleaned: pd.DataFrame) -> plt.Figure:
        """Create summary infographic"""
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle('RTGS AI Analyst - Data Processing Summary', fontsize=20, fontweight='bold')
        
        # Extract scores safely
        raw_score = analysis_results.get('raw_dataset_analysis', {}).get('quality_score', 0)
        final_score = analysis_results.get('cleaned_dataset_analysis', {}).get('final_quality_score', raw_score)
        improvement = final_score - raw_score
        
        # Create layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Main score
        ax_main = fig.add_subplot(gs[0:2, 0:2])
        ax_main.text(0.5, 0.6, f'{final_score:.1f}', ha='center', va='center', 
                    fontsize=72, fontweight='bold', color='darkblue')
        ax_main.text(0.5, 0.3, 'FINAL QUALITY SCORE', ha='center', va='center', fontsize=16)
        
        improvement_text = f'{"+" if improvement >= 0 else ""}{improvement:.1f} points'
        ax_main.text(0.5, 0.1, improvement_text, ha='center', va='center', 
                    fontsize=12, color='green' if improvement >= 0 else 'red')
        ax_main.axis('off')
        
        # Shape changes
        ax_shape = fig.add_subplot(gs[0, 2])
        ax_shape.text(0.5, 0.8, 'SHAPE CHANGE', ha='center', fontweight='bold')
        ax_shape.text(0.5, 0.5, f'{df_original.shape[0]} × {df_original.shape[1]}', ha='center')
        ax_shape.text(0.5, 0.3, '↓', ha='center', fontsize=20)
        ax_shape.text(0.5, 0.1, f'{df_cleaned.shape[0]} × {df_cleaned.shape[1]}', ha='center')
        ax_shape.axis('off')
        
        # Actions applied
        ax_actions = fig.add_subplot(gs[1, 2])
        actions_count = analysis_results.get('cleaned_dataset_analysis', {}).get(
            'improvement_summary', {}).get('actions_applied', 0)
        ax_actions.text(0.5, 0.5, f'{actions_count}', ha='center', va='center', 
                       fontsize=36, fontweight='bold', color='orange')
        ax_actions.text(0.5, 0.2, 'ACTIONS APPLIED', ha='center', fontsize=10)
        ax_actions.axis('off')
        
        # Quality indicator
        ax_quality = fig.add_subplot(gs[2, :])
        quality_bar = ax_quality.barh(['Original', 'Final'], [raw_score, final_score], 
                                       color=['lightcoral', 'lightgreen'])
        ax_quality.set_xlim(0, 100)
        ax_quality.set_xlabel('Quality Score')
        for i, (label, score) in enumerate(zip(['Original', 'Final'], [raw_score, final_score])):
            ax_quality.text(score + 2, i, f'{score:.1f}', va='center')
        
        return fig
    
    def create_shape_quality_comparison(self, df_original: pd.DataFrame, 
                                       df_cleaned: pd.DataFrame,
                                       analysis_results: Dict) -> plt.Figure:
        """Create shape and quality metrics comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Dataset shape
        labels = ['Original', 'Cleaned']
        rows = [df_original.shape[0], df_cleaned.shape[0]]
        cols = [df_original.shape[1], df_cleaned.shape[1]]
        
        x = np.arange(len(labels))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, rows, width, label='Rows', color='skyblue')
        axes[0, 0].bar(x + width/2, cols, width, label='Columns', color='lightcoral')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Dataset Shape Comparison')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(labels)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Memory usage
        memory_orig = df_original.memory_usage(deep=True).sum() / (1024 * 1024)
        memory_clean = df_cleaned.memory_usage(deep=True).sum() / (1024 * 1024)
        
        axes[0, 1].bar(labels, [memory_orig, memory_clean], color=['orange', 'green'])
        axes[0, 1].set_ylabel('Memory (MB)')
        axes[0, 1].set_title('Memory Usage Comparison')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Missing values total
        missing_orig = df_original.isnull().sum().sum()
        missing_clean = df_cleaned.isnull().sum().sum()
        
        axes[1, 0].bar(labels, [missing_orig, missing_clean], color=['red', 'green'])
        axes[1, 0].set_ylabel('Missing Values')
        axes[1, 0].set_title('Total Missing Values')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Duplicate rows
        dup_orig = df_original.duplicated().sum()
        dup_clean = df_cleaned.duplicated().sum()
        
        axes[1, 1].bar(labels, [dup_orig, dup_clean], color=['red', 'green'])
        axes[1, 1].set_ylabel('Duplicate Rows')
        axes[1, 1].set_title('Duplicate Rows Comparison')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_missing_data_comparison(self, df_original: pd.DataFrame, 
                                      df_cleaned: pd.DataFrame) -> plt.Figure:
        """Compare missing data only for common columns"""
        # Find common columns
        common_cols = list(set(df_original.columns) & set(df_cleaned.columns))
        
        if not common_cols:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No common columns between datasets', 
                   ha='center', va='center', fontsize=14)
            ax.set_title('Missing Data Comparison')
            ax.axis('off')
            return fig
        
        # Sort columns for consistent display
        common_cols = sorted(common_cols)[:15]  # Limit to 15 columns for readability
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Missing counts
        missing_orig = [df_original[col].isnull().sum() for col in common_cols]
        missing_clean = [df_cleaned[col].isnull().sum() for col in common_cols]
        
        x = np.arange(len(common_cols))
        width = 0.35
        
        ax1.bar(x - width/2, missing_orig, width, label='Original', alpha=0.8, color='red')
        ax1.bar(x + width/2, missing_clean, width, label='Cleaned', alpha=0.8, color='green')
        ax1.set_xlabel('Columns')
        ax1.set_ylabel('Missing Count')
        ax1.set_title('Missing Values (Common Columns)')
        ax1.set_xticks(x)
        ax1.set_xticklabels(common_cols, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Missing percentages
        missing_orig_pct = [(m/len(df_original))*100 for m in missing_orig]
        missing_clean_pct = [(m/len(df_cleaned))*100 for m in missing_clean]
        
        ax2.bar(x - width/2, missing_orig_pct, width, label='Original', alpha=0.8, color='red')
        ax2.bar(x + width/2, missing_clean_pct, width, label='Cleaned', alpha=0.8, color='green')
        ax2.set_xlabel('Columns')
        ax2.set_ylabel('Missing %')
        ax2.set_title('Missing Percentage (Common Columns)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(common_cols, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_distribution_comparison(self, df_original: pd.DataFrame, 
                                     df_cleaned: pd.DataFrame) -> plt.Figure:
        """Compare distributions for common numeric columns"""
        # Find common numeric columns
        orig_numeric = set(df_original.select_dtypes(include=[np.number]).columns)
        clean_numeric = set(df_cleaned.select_dtypes(include=[np.number]).columns)
        common_numeric = list(orig_numeric & clean_numeric)[:6]  # Limit to 6
        
        if not common_numeric:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No common numeric columns for distribution comparison', 
                   ha='center', va='center', fontsize=14)
            ax.set_title('Distribution Comparison')
            ax.axis('off')
            return fig
        
        n_cols = len(common_numeric)
        n_rows = (n_cols + 2) // 3
        fig, axes = plt.subplots(n_rows, 3, figsize=(15, n_rows * 4))
        
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, col in enumerate(common_numeric):
            row = idx // 3
            col_idx = idx % 3
            ax = axes[row, col_idx]
            
            # Plot distributions
            ax.hist(df_original[col].dropna(), bins=30, alpha=0.5, label='Original', 
                   color='blue', density=True)
            ax.hist(df_cleaned[col].dropna(), bins=30, alpha=0.5, label='Cleaned', 
                   color='green', density=True)
            
            ax.set_title(f'{col}')
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for idx in range(n_cols, n_rows * 3):
            row = idx // 3
            col_idx = idx % 3
            axes[row, col_idx].axis('off')
        
        plt.tight_layout()
        return fig
    
    def create_datatypes_comparison(self, df_original: pd.DataFrame, 
                                   df_cleaned: pd.DataFrame) -> plt.Figure:
        """Compare data types between datasets"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Original data types
        orig_types = df_original.dtypes.value_counts()
        ax1.pie(orig_types.values, labels=orig_types.index, autopct='%1.1f%%', startangle=90)
        ax1.set_title(f'Original Data Types ({df_original.shape[1]} columns)')
        
        # Cleaned data types
        clean_types = df_cleaned.dtypes.value_counts()
        ax2.pie(clean_types.values, labels=clean_types.index, autopct='%1.1f%%', startangle=90)
        ax2.set_title(f'Cleaned Data Types ({df_cleaned.shape[1]} columns)')
        
        plt.tight_layout()
        return fig
    
    def create_transformation_summary(self, df_original: pd.DataFrame, 
                                    df_cleaned: pd.DataFrame,
                                    analysis_results: Dict) -> plt.Figure:
        """Create transformation summary"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Column changes
        orig_cols = set(df_original.columns)
        clean_cols = set(df_cleaned.columns)
        
        new_cols = clean_cols - orig_cols
        removed_cols = orig_cols - clean_cols
        common_cols = orig_cols & clean_cols
        
        labels = ['Common', 'New', 'Removed']
        sizes = [len(common_cols), len(new_cols), len(removed_cols)]
        colors = ['lightblue', 'lightgreen', 'lightcoral']
        
        # Filter out zero values for pie chart
        non_zero_indices = [i for i, s in enumerate(sizes) if s > 0]
        if non_zero_indices:
            filtered_labels = [labels[i] for i in non_zero_indices]
            filtered_sizes = [sizes[i] for i in non_zero_indices]
            filtered_colors = [colors[i] for i in non_zero_indices]
            
            axes[0, 0].pie(filtered_sizes, labels=filtered_labels, colors=filtered_colors,
                          autopct='%1.0f', startangle=90)
            axes[0, 0].set_title('Column Changes')
        else:
            axes[0, 0].text(0.5, 0.5, 'No column changes', ha='center', va='center')
            axes[0, 0].set_title('Column Changes')
            axes[0, 0].axis('off')
        
        # Transformation types
        transformations = analysis_results.get('cleaned_dataset_analysis', {}).get(
            'changes_made', {}).get('transformations', [])
        
        if transformations:
            # Count transformation types
            trans_types = {}
            for trans in transformations:
                if isinstance(trans, str):
                    trans_type = trans.split(':')[0] if ':' in trans else 'Other'
                    trans_types[trans_type] = trans_types.get(trans_type, 0) + 1
            
            if trans_types:
                axes[0, 1].bar(trans_types.keys(), trans_types.values(), color='orange')
                axes[0, 1].set_title('Transformation Types Applied')
                axes[0, 1].set_xlabel('Type')
                axes[0, 1].set_ylabel('Count')
                axes[0, 1].tick_params(axis='x', rotation=45)
            else:
                axes[0, 1].text(0.5, 0.5, 'No transformations', ha='center', va='center')
                axes[0, 1].axis('off')
        else:
            axes[0, 1].text(0.5, 0.5, 'No transformations applied', ha='center', va='center')
            axes[0, 1].axis('off')
        
        # New columns created (if any)
        if new_cols:
            new_cols_list = list(new_cols)[:10]  # Show first 10
            axes[1, 0].barh(range(len(new_cols_list)), [1]*len(new_cols_list), color='green')
            axes[1, 0].set_yticks(range(len(new_cols_list)))
            axes[1, 0].set_yticklabels(new_cols_list)
            axes[1, 0].set_title(f'New Columns Added ({len(new_cols)} total)')
            axes[1, 0].set_xlabel('Created')
        else:
            axes[1, 0].text(0.5, 0.5, 'No new columns added', ha='center', va='center')
            axes[1, 0].axis('off')
        
        # Summary statistics
        summary_data = {
            'Original Columns': df_original.shape[1],
            'Final Columns': df_cleaned.shape[1],
            'Rows Unchanged': df_cleaned.shape[0] == df_original.shape[0],
            'New Features': len(new_cols)
        }
        
        axes[1, 1].axis('tight')
        axes[1, 1].axis('off')
        table_data = [[k, v] for k, v in summary_data.items()]
        table = axes[1, 1].table(cellText=table_data, colLabels=['Metric', 'Value'],
                                 cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        axes[1, 1].set_title('Transformation Summary')
        
        plt.tight_layout()
        return fig