"""
Visualization Agent - Creates comparison charts and exports to PDF
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, List, Any, Optional, Tuple
import os
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime

from config import VIZ_CONFIG, PATHS
from utils import get_timestamp

logger = logging.getLogger(__name__)

class VisualizationAgent:
    """Agent responsible for creating visualizations and charts"""
    
    def __init__(self):
        self.name = "Visualization Agent"
        self.figures = []
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette(VIZ_CONFIG['color_palette'])
        
    def create_missing_data_comparison(self, df_original: pd.DataFrame, 
                                     df_cleaned: pd.DataFrame) -> plt.Figure:
        """Create missing data comparison chart"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Original dataset missing data
        missing_orig = df_original.isnull().sum()
        missing_orig_pct = (missing_orig / len(df_original)) * 100
        
        # Cleaned dataset missing data
        missing_clean = df_cleaned.isnull().sum()
        missing_clean_pct = (missing_clean / len(df_cleaned)) * 100
        
        # Plot 1: Missing data counts
        x_pos = np.arange(len(missing_orig))
        width = 0.35
        
        ax1.bar(x_pos - width/2, missing_orig.values, width, 
                label='Original', alpha=0.8, color=VIZ_CONFIG['color_palette'][0])
        ax1.bar(x_pos + width/2, missing_clean.values, width, 
                label='Cleaned', alpha=0.8, color=VIZ_CONFIG['color_palette'][1])
        
        ax1.set_xlabel('Columns')
        ax1.set_ylabel('Missing Values Count')
        ax1.set_title('Missing Values Comparison (Count)')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(missing_orig.index, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Missing data percentages
        ax2.bar(x_pos - width/2, missing_orig_pct.values, width, 
                label='Original', alpha=0.8, color=VIZ_CONFIG['color_palette'][0])
        ax2.bar(x_pos + width/2, missing_clean_pct.values, width, 
                label='Cleaned', alpha=0.8, color=VIZ_CONFIG['color_palette'][1])
        
        ax2.set_xlabel('Columns')
        ax2.set_ylabel('Missing Values (%)')
        ax2.set_title('Missing Values Comparison (Percentage)')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(missing_orig_pct.index, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_data_distribution_comparison(self, df_original: pd.DataFrame, 
                                          df_cleaned: pd.DataFrame) -> plt.Figure:
        """Create data distribution comparison for numeric columns"""
        numeric_cols = df_original.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            # Create empty plot
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No numeric columns available for distribution comparison', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title('Data Distribution Comparison')
            return fig
        
        # Limit to first 6 columns for visibility
        cols_to_plot = numeric_cols[:6]
        n_cols = len(cols_to_plot)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten() if n_cols > 1 else [axes]
        
        for i, col in enumerate(cols_to_plot):
            if i >= 6:  # Safety check
                break
                
            ax = axes[i] if i < len(axes) else axes[-1]
            
            # Original data
            orig_data = df_original[col].dropna()
            clean_data = df_cleaned[col].dropna() if col in df_cleaned.columns else pd.Series()
            
            if len(orig_data) > 0:
                ax.hist(orig_data, bins=30, alpha=0.6, label='Original', 
                       color=VIZ_CONFIG['color_palette'][0], density=True)
            
            if len(clean_data) > 0:
                ax.hist(clean_data, bins=30, alpha=0.6, label='Cleaned', 
                       color=VIZ_CONFIG['color_palette'][1], density=True)
            
            ax.set_title(f'Distribution: {col}')
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(n_cols, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def create_data_quality_summary(self, df_original: pd.DataFrame, 
                                   df_cleaned: pd.DataFrame,
                                   analysis_results: Dict[str, Any]) -> plt.Figure:
        """Create data quality summary dashboard"""
        fig = plt.figure(figsize=(16, 12))
        
        # Create a grid layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Quality Score Comparison
        ax1 = fig.add_subplot(gs[0, 0])
        raw_score = analysis_results.get('raw_dataset_analysis', {}).get('quality_score', 0)
        final_score = analysis_results.get('cleaned_dataset_analysis', {}).get('final_quality_score', 0)
        
        scores = [raw_score, final_score]
        labels = ['Original', 'Cleaned']
        colors = [VIZ_CONFIG['color_palette'][3] if raw_score < 70 else VIZ_CONFIG['color_palette'][2], 
                 VIZ_CONFIG['color_palette'][2]]
        
        bars = ax1.bar(labels, scores, color=colors, alpha=0.8)
        ax1.set_ylim(0, 100)
        ax1.set_ylabel('Quality Score')
        ax1.set_title('Data Quality Score')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{score:.1f}', ha='center', va='bottom')
        
        # 2. Dataset Shape Comparison
        ax2 = fig.add_subplot(gs[0, 1])
        shape_data = {
            'Rows': [df_original.shape[0], df_cleaned.shape[0]],
            'Columns': [df_original.shape[1], df_cleaned.shape[1]]
        }
        
        x = np.arange(len(labels))
        width = 0.35
        
        ax2.bar(x - width/2, shape_data['Rows'], width, label='Rows', 
               color=VIZ_CONFIG['color_palette'][0], alpha=0.8)
        ax2.bar(x + width/2, shape_data['Columns'], width, label='Columns', 
               color=VIZ_CONFIG['color_palette'][1], alpha=0.8)
        
        ax2.set_xlabel('Dataset')
        ax2.set_ylabel('Count')
        ax2.set_title('Dataset Shape Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Missing Data Summary
        ax3 = fig.add_subplot(gs[0, 2])
        missing_orig = df_original.isnull().sum().sum()
        missing_clean = df_cleaned.isnull().sum().sum()
        
        missing_data = [missing_orig, missing_clean]
        ax3.bar(labels, missing_data, color=[VIZ_CONFIG['color_palette'][3], VIZ_CONFIG['color_palette'][2]], alpha=0.8)
        ax3.set_ylabel('Total Missing Values')
        ax3.set_title('Missing Data Reduction')
        ax3.grid(True, alpha=0.3)
        
        # 4. Data Types Distribution (Original)
        ax4 = fig.add_subplot(gs[1, 0])
        orig_types = df_original.dtypes.value_counts()
        ax4.pie(orig_types.values, labels=orig_types.index, autopct='%1.1f%%', startangle=90)
        ax4.set_title('Original Data Types')
        
        # 5. Data Types Distribution (Cleaned)
        ax5 = fig.add_subplot(gs[1, 1])
        clean_types = df_cleaned.dtypes.value_counts()
        ax5.pie(clean_types.values, labels=clean_types.index, autopct='%1.1f%%', startangle=90)
        ax5.set_title('Cleaned Data Types')
        
        # 6. Memory Usage Comparison
        ax6 = fig.add_subplot(gs[1, 2])
        memory_orig = df_original.memory_usage(deep=True).sum() / (1024 * 1024)
        memory_clean = df_cleaned.memory_usage(deep=True).sum() / (1024 * 1024)
        
        memory_data = [memory_orig, memory_clean]
        ax6.bar(labels, memory_data, color=[VIZ_CONFIG['color_palette'][0], VIZ_CONFIG['color_palette'][1]], alpha=0.8)
        ax6.set_ylabel('Memory Usage (MB)')
        ax6.set_title('Memory Usage Comparison')
        ax6.grid(True, alpha=0.3)
        
        # 7. Actions Applied Summary (Bottom row, spanning all columns)
        ax7 = fig.add_subplot(gs[2, :])
        
        # Get actions from analysis results
        cleaning_actions = analysis_results.get('cleaned_dataset_analysis', {}).get('changes_made', {}).get('cleaning_actions', [])
        transformations = analysis_results.get('cleaned_dataset_analysis', {}).get('changes_made', {}).get('transformations', [])
        
        action_types = ['Cleaning Actions', 'Transformations']
        action_counts = [len(cleaning_actions), len(transformations)]
        
        bars = ax7.barh(action_types, action_counts, color=VIZ_CONFIG['color_palette'][:2], alpha=0.8)
        ax7.set_xlabel('Number of Actions')
        ax7.set_title('Data Processing Actions Applied')
        ax7.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, count in zip(bars, action_counts):
            width = bar.get_width()
            ax7.text(width + 0.1, bar.get_y() + bar.get_height()/2,
                    f'{count}', ha='left', va='center')
        
        return fig
    
    def create_correlation_comparison(self, df_original: pd.DataFrame, 
                                    df_cleaned: pd.DataFrame) -> plt.Figure:
        """Create correlation matrix comparison"""
        numeric_cols_orig = df_original.select_dtypes(include=[np.number]).columns
        numeric_cols_clean = df_cleaned.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols_orig) < 2 and len(numeric_cols_clean) < 2:
            # Create empty plot
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'Insufficient numeric columns for correlation analysis', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title('Correlation Matrix Comparison')
            return fig
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Original correlation matrix
        if len(numeric_cols_orig) >= 2:
            corr_orig = df_original[numeric_cols_orig].corr()
            sns.heatmap(corr_orig, annot=True, cmap='coolwarm', center=0,
                       square=True, ax=ax1, cbar_kws={'shrink': 0.8})
            ax1.set_title('Original Dataset Correlations')
        else:
            ax1.text(0.5, 0.5, 'Insufficient numeric columns\nin original dataset', 
                    ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Original Dataset Correlations')
        
        # Cleaned correlation matrix
        if len(numeric_cols_clean) >= 2:
            corr_clean = df_cleaned[numeric_cols_clean].corr()
            sns.heatmap(corr_clean, annot=True, cmap='coolwarm', center=0,
                       square=True, ax=ax2, cbar_kws={'shrink': 0.8})
            ax2.set_title('Cleaned Dataset Correlations')
        else:
            ax2.text(0.5, 0.5, 'Insufficient numeric columns\nin cleaned dataset', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Cleaned Dataset Correlations')
        
        plt.tight_layout()
        return fig
    
    def create_categorical_analysis(self, df_original: pd.DataFrame, 
                                  df_cleaned: pd.DataFrame) -> plt.Figure:
        """Create categorical variables analysis"""
        cat_cols_orig = df_original.select_dtypes(include=['object', 'category']).columns
        cat_cols_clean = df_cleaned.select_dtypes(include=['object', 'category']).columns
        
        if len(cat_cols_orig) == 0 and len(cat_cols_clean) == 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No categorical columns available for analysis', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title('Categorical Variables Analysis')
            return fig
        
        # Create subplots for unique value counts
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Unique value counts in original dataset
        if len(cat_cols_orig) > 0:
            unique_counts_orig = [df_original[col].nunique() for col in cat_cols_orig[:10]]
            col_names_orig = cat_cols_orig[:10]
            
            axes[0, 0].barh(col_names_orig, unique_counts_orig, color=VIZ_CONFIG['color_palette'][0], alpha=0.8)
            axes[0, 0].set_xlabel('Unique Values Count')
            axes[0, 0].set_title('Original Dataset: Categorical Cardinality')
            axes[0, 0].grid(True, alpha=0.3)
        else:
            axes[0, 0].text(0.5, 0.5, 'No categorical columns\nin original dataset', 
                           ha='center', va='center', transform=axes[0, 0].transAxes)
            axes[0, 0].set_title('Original Dataset: Categorical Cardinality')
        
        # Plot 2: Unique value counts in cleaned dataset
        if len(cat_cols_clean) > 0:
            unique_counts_clean = [df_cleaned[col].nunique() for col in cat_cols_clean[:10]]
            col_names_clean = cat_cols_clean[:10]
            
            axes[0, 1].barh(col_names_clean, unique_counts_clean, color=VIZ_CONFIG['color_palette'][1], alpha=0.8)
            axes[0, 1].set_xlabel('Unique Values Count')
            axes[0, 1].set_title('Cleaned Dataset: Categorical Cardinality')
            axes[0, 1].grid(True, alpha=0.3)
        else:
            axes[0, 1].text(0.5, 0.5, 'No categorical columns\nin cleaned dataset', 
                           ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Cleaned Dataset: Categorical Cardinality')
        
        # Plot 3: Most frequent values in original (if available)
        if len(cat_cols_orig) > 0:
            sample_col = cat_cols_orig[0]
            value_counts = df_original[sample_col].value_counts().head(10)
            
            axes[1, 0].bar(range(len(value_counts)), value_counts.values, 
                          color=VIZ_CONFIG['color_palette'][2], alpha=0.8)
            axes[1, 0].set_xlabel('Categories')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title(f'Original: Top Categories in {sample_col}')
            axes[1, 0].set_xticks(range(len(value_counts)))
            axes[1, 0].set_xticklabels(value_counts.index, rotation=45, ha='right')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'No categorical data available', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Original: Category Frequencies')
        
        # Plot 4: Most frequent values in cleaned (if available)
        if len(cat_cols_clean) > 0:
            sample_col = cat_cols_clean[0]
            value_counts = df_cleaned[sample_col].value_counts().head(10)
            
            axes[1, 1].bar(range(len(value_counts)), value_counts.values, 
                          color=VIZ_CONFIG['color_palette'][3], alpha=0.8)
            axes[1, 1].set_xlabel('Categories')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title(f'Cleaned: Top Categories in {sample_col}')
            axes[1, 1].set_xticks(range(len(value_counts)))
            axes[1, 1].set_xticklabels(value_counts.index, rotation=45, ha='right')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'No categorical data available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Cleaned: Category Frequencies')
        
        plt.tight_layout()
        return fig
    
    def create_summary_infographic(self, analysis_results: Dict[str, Any]) -> plt.Figure:
        """Create a summary infographic of the entire process"""
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle('RTGS AI Analyst - Data Processing Summary', fontsize=20, fontweight='bold')
        
        # Create custom layout
        gs = fig.add_gridspec(4, 4, hspace=0.4, wspace=0.3)
        
        # Key metrics
        raw_analysis = analysis_results.get('raw_dataset_analysis', {})
        cleaned_analysis = analysis_results.get('cleaned_dataset_analysis', {})
        
        raw_score = raw_analysis.get('quality_score', 0)
        final_score = cleaned_analysis.get('final_quality_score', 0)
        improvement = final_score - raw_score
        
        # 1. Main Quality Score (top center, large)
        ax_main = fig.add_subplot(gs[0:2, 1:3])
        ax_main.text(0.5, 0.7, f'{final_score:.1f}', ha='center', va='center', 
                    fontsize=72, fontweight='bold', color=VIZ_CONFIG['color_palette'][2])
        ax_main.text(0.5, 0.4, 'FINAL QUALITY SCORE', ha='center', va='center', 
                    fontsize=16, fontweight='bold')
        ax_main.text(0.5, 0.2, f'(+{improvement:.1f} improvement)', ha='center', va='center', 
                    fontsize=12, color=VIZ_CONFIG['color_palette'][2])
        ax_main.set_xlim(0, 1)
        ax_main.set_ylim(0, 1)
        ax_main.axis('off')
        
        # Add quality indicator
        if final_score >= 90:
            quality_text = "EXCELLENT"
            quality_color = 'green'
        elif final_score >= 75:
            quality_text = "GOOD"
            quality_color = 'orange'
        elif final_score >= 60:
            quality_text = "ACCEPTABLE"
            quality_color = 'gold'
        else:
            quality_text = "NEEDS WORK"
            quality_color = 'red'
        
        ax_main.text(0.5, 0.05, quality_text, ha='center', va='center', 
                    fontsize=14, fontweight='bold', color=quality_color)
        
        # 2. Original vs Final comparison (left side)
        ax_comparison = fig.add_subplot(gs[0, 0])
        scores = [raw_score, final_score]
        labels = ['Original', 'Final']
        colors = ['lightcoral', VIZ_CONFIG['color_palette'][2]]
        
        bars = ax_comparison.bar(labels, scores, color=colors, alpha=0.8)
        ax_comparison.set_ylim(0, 100)
        ax_comparison.set_title('Quality Improvement', fontweight='bold')
        ax_comparison.grid(True, alpha=0.3)
        
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax_comparison.text(bar.get_x() + bar.get_width()/2., height + 2,
                              f'{score:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Actions Applied (right side)
        ax_actions = fig.add_subplot(gs[0, 3])
        actions_count = cleaned_analysis.get('improvement_summary', {}).get('actions_applied', 0)
        
        ax_actions.text(0.5, 0.6, f'{actions_count}', ha='center', va='center', 
                       fontsize=36, fontweight='bold', color=VIZ_CONFIG['color_palette'][1])
        ax_actions.text(0.5, 0.3, 'ACTIONS\nAPPLIED', ha='center', va='center', 
                       fontsize=12, fontweight='bold')
        ax_actions.set_xlim(0, 1)
        ax_actions.set_ylim(0, 1)
        ax_actions.axis('off')
        
        # 4. Key improvements (bottom left)
        ax_improvements = fig.add_subplot(gs[2:, :2])
        improvements = cleaned_analysis.get('improvements', [])[:5]
        
        ax_improvements.text(0.5, 0.9, 'KEY IMPROVEMENTS', ha='center', va='top', 
                            fontsize=14, fontweight='bold', transform=ax_improvements.transAxes)
        
        for i, improvement in enumerate(improvements):
            y_pos = 0.8 - (i * 0.15)
            ax_improvements.text(0.05, y_pos, f'• {improvement}', ha='left', va='top', 
                                fontsize=10, transform=ax_improvements.transAxes)
        
        ax_improvements.set_xlim(0, 1)
        ax_improvements.set_ylim(0, 1)
        ax_improvements.axis('off')
        
        # 5. Recommendations (bottom right)
        ax_recommendations = fig.add_subplot(gs[2:, 2:])
        recommendations = cleaned_analysis.get('future_recommendations', [])[:4]
        
        ax_recommendations.text(0.5, 0.9, 'RECOMMENDATIONS', ha='center', va='top', 
                               fontsize=14, fontweight='bold', transform=ax_recommendations.transAxes)
        
        for i, rec in enumerate(recommendations):
            y_pos = 0.8 - (i * 0.15)
            ax_recommendations.text(0.05, y_pos, f'• {rec}', ha='left', va='top', 
                                   fontsize=10, transform=ax_recommendations.transAxes)
        
        ax_recommendations.set_xlim(0, 1)
        ax_recommendations.set_ylim(0, 1)
        ax_recommendations.axis('off')
        
        # 6. Readiness status (top right)
        ax_readiness = fig.add_subplot(gs[1, 3])
        readiness = cleaned_analysis.get('readiness_assessment', {})
        is_ready = readiness.get('overall_ready', False)
        
        status_text = "READY FOR\nANALYSIS" if is_ready else "NEEDS MORE\nWORK"
        status_color = VIZ_CONFIG['color_palette'][2] if is_ready else 'orange'
        
        ax_readiness.text(0.5, 0.5, status_text, ha='center', va='center', 
                         fontsize=12, fontweight='bold', color=status_color)
        ax_readiness.set_xlim(0, 1)
        ax_readiness.set_ylim(0, 1)
        ax_readiness.axis('off')
        
        return fig
    
    def export_to_pdf(self, output_path: str, analysis_results: Dict[str, Any], 
                     df_original: pd.DataFrame, df_cleaned: pd.DataFrame) -> str:
        """Export all visualizations to a PDF file"""
        
        logger.info(f"{self.name}: Creating visualization PDF...")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with PdfPages(output_path) as pdf:
            # Page 1: Summary Infographic
            logger.info(f"{self.name}: Creating summary infographic...")
            fig1 = self.create_summary_infographic(analysis_results)
            pdf.savefig(fig1, bbox_inches='tight', dpi=300)
            plt.close(fig1)
            
            # Page 2: Data Quality Summary
            logger.info(f"{self.name}: Creating data quality summary...")
            fig2 = self.create_data_quality_summary(df_original, df_cleaned, analysis_results)
            pdf.savefig(fig2, bbox_inches='tight', dpi=300)
            plt.close(fig2)
            
            # Page 3: Missing Data Comparison
            logger.info(f"{self.name}: Creating missing data comparison...")
            fig3 = self.create_missing_data_comparison(df_original, df_cleaned)
            pdf.savefig(fig3, bbox_inches='tight', dpi=300)
            plt.close(fig3)
            
            # Page 4: Distribution Comparison
            logger.info(f"{self.name}: Creating distribution comparison...")
            fig4 = self.create_data_distribution_comparison(df_original, df_cleaned)
            pdf.savefig(fig4, bbox_inches='tight', dpi=300)
            plt.close(fig4)
            
            # Page 5: Correlation Comparison
            logger.info(f"{self.name}: Creating correlation comparison...")
            fig5 = self.create_correlation_comparison(df_original, df_cleaned)
            pdf.savefig(fig5, bbox_inches='tight', dpi=300)
            plt.close(fig5)
            
            # Page 6: Categorical Analysis
            logger.info(f"{self.name}: Creating categorical analysis...")
            fig6 = self.create_categorical_analysis(df_original, df_cleaned)
            pdf.savefig(fig6, bbox_inches='tight', dpi=300)
            plt.close(fig6)
            
            # Set PDF metadata
            pdf_info = pdf.infodict()
            pdf_info['Title'] = 'RTGS AI Analyst - Data Quality Report'
            pdf_info['Author'] = 'RTGS AI Analyst'
            pdf_info['Subject'] = 'Data Quality Analysis and Comparison'
            pdf_info['Keywords'] = 'Data Quality, Data Cleaning, Analysis, Visualization'
            pdf_info['Creator'] = 'RTGS AI Analyst System'
            pdf_info['Producer'] = 'matplotlib/seaborn'
            pdf_info['CreationDate'] = datetime.now()
        
        logger.info(f"{self.name}: PDF exported successfully to {output_path}")
        return output_path
    
    def process(self, df_original: pd.DataFrame, df_cleaned: pd.DataFrame, 
                analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main processing method for the Visualization Agent
        
        Args:
            df_original: Original raw dataset
            df_cleaned: Cleaned dataset
            analysis_results: Results from analysis agent
            
        Returns:
            Visualization results with PDF path
        """
        logger.info(f"{self.name}: Starting visualization creation")
        
        try:
            # Create output filename
            timestamp = get_timestamp()
            filename = f"data_quality_report_{timestamp}.pdf"
            output_path = os.path.join(PATHS.outputs_charts, filename)
            
            # Export visualizations to PDF
            pdf_path = self.export_to_pdf(output_path, analysis_results, df_original, df_cleaned)
            
            # Calculate some summary statistics
            charts_created = 6  # Number of chart pages
            file_size = os.path.getsize(pdf_path) / (1024 * 1024)  # MB
            
            result = {
                'agent': self.name,
                'status': 'success',
                'pdf_path': pdf_path,
                'charts_created': charts_created,
                'file_size_mb': round(file_size, 2),
                'message': f"Visualization PDF created successfully with {charts_created} charts. File size: {file_size:.2f} MB"
            }
            
            logger.info(f"{self.name}: Visualization completed successfully")
            logger.info(f"  - PDF created: {pdf_path}")
            logger.info(f"  - Charts: {charts_created}")
            logger.info(f"  - File size: {file_size:.2f} MB")
            
            return result
            
        except Exception as e:
            error_msg = f"Visualization failed: {str(e)}"
            logger.error(f"{self.name}: {error_msg}")
            
            return {
                'agent': self.name,
                'status': 'error',
                'message': error_msg,
                'error': str(e)
            }