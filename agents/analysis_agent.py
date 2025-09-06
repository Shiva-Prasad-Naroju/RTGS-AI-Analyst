import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from typing import Dict, List, Any, Tuple
import logging
from pathlib import Path

class AnalysisAgent:
    """Agent responsible for generating insights and analysis from the processed dataset"""
    
    def __init__(self, output_dir: str = "outputs"):
        self.logger = logging.getLogger(__name__)
        self.console = Console()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "charts").mkdir(exist_ok=True)
        
    def generate_analysis(self, 
                         df: pd.DataFrame, 
                         cleaning_log: List[str], 
                         transformation_log: List[str],
                         verification_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive analysis and insights
        
        Args:
            df: Final processed dataframe
            cleaning_log: Log of cleaning operations
            transformation_log: Log of transformation operations
            verification_results: Results from verification
            
        Returns:
            Analysis results dictionary
        """
        self.logger.info("📊 Starting comprehensive analysis...")
        
        analysis_results = {
            'basic_statistics': {},
            'insights': {},
            'policy_recommendations': [],
            'visualizations': [],
            'data_summary': {}
        }
        
        # 1. Basic statistical analysis
        analysis_results['basic_statistics'] = self._generate_basic_statistics(df)
        
        # 2. Domain-specific insights
        analysis_results['insights'] = self._generate_domain_insights(df)
        
        # 3. Policy recommendations
        analysis_results['policy_recommendations'] = self._generate_policy_recommendations(df)
        
        # 4. Create visualizations
        analysis_results['visualizations'] = self._create_visualizations(df)
        
        # 5. Generate data processing summary
        analysis_results['data_summary'] = self._generate_processing_summary(
            df, cleaning_log, transformation_log, verification_results
        )
        
        # 6. Display results in CLI
        self._display_cli_results(analysis_results)
        
        # 7. Save reports
        self._save_reports(analysis_results, df)
        
        self.logger.info("✅ Analysis completed successfully")
        return analysis_results
    
    def _generate_basic_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate basic statistical summary"""
        
        stats = {
            'dataset_shape': df.shape,
            'numeric_summary': {},
            'categorical_summary': {},
            'missing_data': {},
            'column_types': df.dtypes.value_counts().to_dict()
        }
        
        # Numeric columns analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            stats['numeric_summary'] = df[numeric_cols].describe().to_dict()
        
        # Categorical columns analysis
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].nunique() <= 20:  # Only for manageable number of categories
                stats['categorical_summary'][col] = df[col].value_counts().head(10).to_dict()
        
        # Missing data analysis
        missing_counts = df.isnull().sum()
        stats['missing_data'] = missing_counts[missing_counts > 0].to_dict()
        
        return stats
    
    def _generate_domain_insights(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate domain-specific insights for healthcare/hospital data"""
        
        insights = {
            'key_metrics': {},
            'distributions': {},
            'correlations': {},
            'trends': {},
            'outliers': {}
        }
        
        # Healthcare-specific metrics
        bed_columns = [col for col in df.columns if 'bed' in col.lower()]
        if bed_columns:
            total_beds = df[bed_columns].sum().sum() if len(bed_columns) > 1 else df[bed_columns[0]].sum()
            avg_beds = df[bed_columns[0]].mean() if bed_columns else 0
            
            insights['key_metrics']['total_beds'] = int(total_beds) if not pd.isna(total_beds) else 0
            insights['key_metrics']['average_beds_per_hospital'] = round(avg_beds, 2) if not pd.isna(avg_beds) else 0
            insights['key_metrics']['hospitals_with_zero_beds'] = int((df[bed_columns[0]] == 0).sum()) if bed_columns else 0
        
        # Geographic distribution
        location_cols = [col for col in df.columns if any(word in col.lower() for word in ['district', 'state', 'city'])]
        if location_cols:
            location_col = location_cols[0]
            location_dist = df[location_col].value_counts()
            insights['distributions']['geographic'] = {
                'total_locations': len(location_dist),
                'top_locations': location_dist.head(5).to_dict(),
                'underserved_locations': location_dist.tail(5).to_dict()
            }
        
        # Hospital type distribution
        type_cols = [col for col in df.columns if any(word in col.lower() for word in ['type', 'ownership', 'category'])]
        if type_cols:
            type_col = type_cols[0]
            type_dist = df[type_col].value_counts()
            insights['distributions']['hospital_types'] = type_dist.to_dict()
        
        # Correlations
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            correlation_matrix = df[numeric_cols].corr()
            # Find strong correlations (> 0.7 or < -0.7)
            strong_correlations = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_val = correlation_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7 and not pd.isna(corr_val):
                        strong_correlations.append({
                            'col1': correlation_matrix.columns[i],
                            'col2': correlation_matrix.columns[j],
                            'correlation': round(corr_val, 3)
                        })
            insights['correlations']['strong_correlations'] = strong_correlations
        
        return insights
    
    def _generate_policy_recommendations(self, df: pd.DataFrame) -> List[str]:
        """Generate policy recommendations based on analysis"""
        
        recommendations = []
        
        # Geographic equity analysis
        location_cols = [col for col in df.columns if any(word in col.lower() for word in ['district', 'state'])]
        if location_cols and len(df) > 1:
            location_col = location_cols[0]
            location_counts = df[location_col].value_counts()
            
            if len(location_counts) > 1:
                min_hospitals = location_counts.min()
                max_hospitals = location_counts.max()
                
                if max_hospitals / min_hospitals > 3:
                    underserved = location_counts[location_counts <= min_hospitals * 2].index.tolist()
                    # Convert all items to strings to avoid join error
                    underserved_str = [str(item) for item in underserved[:3]]
                    recommendations.append(
                        f"Address geographic inequality: {len(underserved)} locations have significantly fewer hospitals. "
                        f"Priority areas: {', '.join(underserved_str)}"
                    )
        
        # Capacity analysis
        bed_columns = [col for col in df.columns if 'bed' in col.lower()]
        if bed_columns:
            bed_col = bed_columns[0]
            if (df[bed_col] == 0).any():
                zero_bed_count = (df[bed_col] == 0).sum()
                recommendations.append(
                    f"Infrastructure development needed: {zero_bed_count} facilities have zero bed capacity"
                )
            
            if df[bed_col].std() > df[bed_col].mean():
                recommendations.append(
                    "Standardize hospital capacity: High variation in bed capacity suggests need for capacity planning"
                )
        
        # Type distribution analysis
        type_cols = [col for col in df.columns if any(word in col.lower() for word in ['type', 'ownership'])]
        if type_cols:
            type_col = type_cols[0]
            type_dist = df[type_col].value_counts(normalize=True)
            
            if len(type_dist) > 1:
                private_ratio = type_dist.get('Private', 0) + type_dist.get('private', 0)
                govt_ratio = type_dist.get('Government', 0) + type_dist.get('Govt', 0) + type_dist.get('Public', 0)
                
                if private_ratio > 0.7:
                    recommendations.append(
                        "Increase public healthcare infrastructure: High dependence on private healthcare facilities"
                    )
                elif govt_ratio > 0.8:
                    recommendations.append(
                        "Encourage public-private partnerships: Consider involving private sector for efficiency"
                    )
        
        # Data quality recommendations
        null_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        if null_pct > 10:
            recommendations.append(
                f"Improve data collection: {null_pct:.1f}% missing data affects analysis reliability"
            )
        
        return recommendations
    
    def _create_visualizations(self, df: pd.DataFrame) -> List[str]:
        """Create and save visualizations"""
        
        visualizations = []
        plt.style.use('default')  # Use default style
        
        try:
            # 1. Numeric distributions
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                axes = axes.flatten()
                
                for i, col in enumerate(numeric_cols[:4]):
                    if i < 4:
                        df[col].hist(bins=20, ax=axes[i], alpha=0.7)
                        axes[i].set_title(f'Distribution of {col}')
                        axes[i].set_xlabel(col)
                        axes[i].set_ylabel('Frequency')
                
                # Hide empty subplots
                for j in range(len(numeric_cols), 4):
                    axes[j].set_visible(False)
                
                plt.tight_layout()
                chart_path = self.output_dir / "charts" / "numeric_distributions.png"
                plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                plt.close()
                visualizations.append(str(chart_path))
            
            # 2. Categorical distributions
            categorical_cols = df.select_dtypes(include=['object']).columns
            manageable_cats = [col for col in categorical_cols if df[col].nunique() <= 10]
            
            if manageable_cats:
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                axes = axes.flatten()
                
                for i, col in enumerate(manageable_cats[:4]):
                    if i < 4:
                        value_counts = df[col].value_counts()
                        value_counts.plot(kind='bar', ax=axes[i], alpha=0.7)
                        axes[i].set_title(f'Distribution of {col}')
                        axes[i].set_xlabel(col)
                        axes[i].set_ylabel('Count')
                        axes[i].tick_params(axis='x', rotation=45)
                
                # Hide empty subplots
                for j in range(len(manageable_cats), 4):
                    axes[j].set_visible(False)
                
                plt.tight_layout()
                chart_path = self.output_dir / "charts" / "categorical_distributions.png"
                plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                plt.close()
                visualizations.append(str(chart_path))
            
            # 3. Correlation heatmap
            if len(numeric_cols) > 1:
                plt.figure(figsize=(10, 8))
                correlation_matrix = df[numeric_cols].corr()
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                           square=True, linewidths=0.5)
                plt.title('Correlation Heatmap')
                plt.tight_layout()
                chart_path = self.output_dir / "charts" / "correlation_heatmap.png"
                plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                plt.close()
                visualizations.append(str(chart_path))
            
        except Exception as e:
            self.logger.warning(f"Failed to create some visualizations: {e}")
        
        return visualizations
    
    def _generate_processing_summary(self, 
                                   df: pd.DataFrame, 
                                   cleaning_log: List[str], 
                                   transformation_log: List[str],
                                   verification_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of all processing steps"""
        
        return {
            'final_dataset_shape': df.shape,
            'cleaning_operations_count': len(cleaning_log),
            'transformation_operations_count': len(transformation_log),
            'quality_score': verification_results.get('summary', {}).get('quality_score', 0),
            'quality_grade': verification_results.get('summary', {}).get('quality_grade', 'N/A'),
            'total_operations': len(cleaning_log) + len(transformation_log),
            'warnings': len(verification_results.get('warnings', [])),
            'errors': len(verification_results.get('errors', []))
        }
    
    def _display_cli_results(self, analysis_results: Dict[str, Any]):
        """Display results in CLI using Rich"""
        
        # Dataset Summary
        summary = analysis_results['data_summary']
        summary_table = Table(title="🔍 Dataset Processing Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")
        
        summary_table.add_row("Final Shape", f"{summary['final_dataset_shape'][0]} rows × {summary['final_dataset_shape'][1]} columns")
        summary_table.add_row("Cleaning Operations", str(summary['cleaning_operations_count']))
        summary_table.add_row("Transformation Operations", str(summary['transformation_operations_count']))
        summary_table.add_row("Quality Score", f"{summary['quality_score']}/100 (Grade: {summary['quality_grade']})")
        summary_table.add_row("Warnings", str(summary['warnings']))
        summary_table.add_row("Errors", str(summary['errors']))
        
        self.console.print(summary_table)
        self.console.print()
        
        # Key Metrics
        if analysis_results['insights'].get('key_metrics'):
            metrics = analysis_results['insights']['key_metrics']
            metrics_table = Table(title="📊 Key Healthcare Metrics")
            metrics_table.add_column("Metric", style="cyan")
            metrics_table.add_column("Value", style="yellow")
            
            for metric, value in metrics.items():
                metrics_table.add_row(metric.replace('_', ' ').title(), str(value))
            
            self.console.print(metrics_table)
            self.console.print()
        
        # Geographic Distribution
        if analysis_results['insights'].get('distributions', {}).get('geographic'):
            geo_dist = analysis_results['insights']['distributions']['geographic']
            geo_table = Table(title="🗺️ Geographic Distribution")
            geo_table.add_column("Location", style="cyan")
            geo_table.add_column("Count", style="green")
            
            for location, count in geo_dist['top_locations'].items():
                geo_table.add_row(str(location), str(count))
            
            self.console.print(geo_table)
            self.console.print()
        
        # Policy Recommendations
        if analysis_results['policy_recommendations']:
            self.console.print(Panel.fit("🎯 Policy Recommendations", style="bold blue"))
            for i, rec in enumerate(analysis_results['policy_recommendations'], 1):
                self.console.print(f"{i}. {rec}")
            self.console.print()
        
        # Visualizations
        if analysis_results['visualizations']:
            viz_panel = Panel.fit(f"📈 Generated {len(analysis_results['visualizations'])} visualizations", style="bold green")
            self.console.print(viz_panel)
            for viz_path in analysis_results['visualizations']:
                self.console.print(f"  • {viz_path}")
            self.console.print()
    
    def _save_reports(self, analysis_results: Dict[str, Any], df: pd.DataFrame):
        """Save detailed reports to files with proper encoding"""
        
        # Save insights report with UTF-8 encoding
        insights_path = self.output_dir / "insights.md"
        try:
            with open(insights_path, 'w', encoding='utf-8') as f:
                f.write("# RTGS AI Analyst - Healthcare Data Analysis Report\n\n")
                
                # Dataset Summary
                summary = analysis_results['data_summary']
                f.write("## Dataset Summary\n\n")
                f.write(f"- **Shape**: {summary['final_dataset_shape'][0]} rows × {summary['final_dataset_shape'][1]} columns\n")
                f.write(f"- **Data Quality Score**: {summary['quality_score']}/100 (Grade: {summary['quality_grade']})\n")
                f.write(f"- **Processing Operations**: {summary['total_operations']} total\n")
                f.write(f"  - Cleaning: {summary['cleaning_operations_count']}\n")
                f.write(f"  - Transformations: {summary['transformation_operations_count']}\n\n")
                
                # Key Metrics
                if analysis_results['insights'].get('key_metrics'):
                    f.write("## Key Healthcare Metrics\n\n")
                    for metric, value in analysis_results['insights']['key_metrics'].items():
                        f.write(f"- **{metric.replace('_', ' ').title()}**: {value}\n")
                    f.write("\n")
                
                # Geographic Distribution
                if analysis_results['insights'].get('distributions', {}).get('geographic'):
                    geo_dist = analysis_results['insights']['distributions']['geographic']
                    f.write("## Geographic Distribution\n\n")
                    f.write(f"**Total Locations**: {geo_dist['total_locations']}\n\n")
                    
                    f.write("### Top Locations by Hospital Count\n")
                    for location, count in geo_dist['top_locations'].items():
                        f.write(f"- {location}: {count} hospitals\n")
                    f.write("\n")
                    
                    if geo_dist.get('underserved_locations'):
                        f.write("### Potentially Underserved Locations\n")
                        for location, count in geo_dist['underserved_locations'].items():
                            f.write(f"- {location}: {count} hospitals\n")
                        f.write("\n")
                
                # Hospital Types
                if analysis_results['insights'].get('distributions', {}).get('hospital_types'):
                    f.write("## Hospital Type Distribution\n\n")
                    type_dist = analysis_results['insights']['distributions']['hospital_types']
                    for hospital_type, count in type_dist.items():
                        f.write(f"- **{hospital_type}**: {count} hospitals\n")
                    f.write("\n")
                
                # Policy Recommendations (without emojis for file)
                f.write("## Policy Recommendations\n\n")
                for i, rec in enumerate(analysis_results['policy_recommendations'], 1):
                    # Remove emojis for file compatibility
                    clean_rec = rec.encode('ascii', 'ignore').decode('ascii')
                    f.write(f"{i}. {clean_rec}\n")
                f.write("\n")
                
                # Technical Details
                f.write("## Technical Analysis Details\n\n")
                if analysis_results['insights'].get('correlations', {}).get('strong_correlations'):
                    f.write("### Strong Correlations Found\n")
                    for corr in analysis_results['insights']['correlations']['strong_correlations']:
                        f.write(f"- {corr['col1']} <-> {corr['col2']}: {corr['correlation']}\n")
                    f.write("\n")
                
                # Processing Summary
                f.write("## Processing Summary\n\n")
                f.write(f"- **Total cleaning operations**: {summary['cleaning_operations_count']}\n")
                f.write(f"- **Total transformations**: {summary['transformation_operations_count']}\n")
                f.write(f"- **Final quality score**: {summary['quality_score']}/100\n")
                f.write(f"- **Warnings**: {summary['warnings']}\n")
                f.write(f"- **Errors**: {summary['errors']}\n\n")
                
        except Exception as e:
            self.logger.warning(f"Failed to save insights report with UTF-8, trying ASCII: {e}")
            # Fallback to ASCII if UTF-8 fails
            with open(insights_path, 'w', encoding='ascii', errors='ignore') as f:
                f.write("# RTGS AI Analyst - Healthcare Data Analysis Report\n\n")
                f.write("Report generated successfully but some special characters may be missing.\n\n")
        
        # Save processed dataset
        try:
            processed_data_path = self.output_dir.parent / "data" / "processed" / "final_dataset.csv"
            processed_data_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(processed_data_path, index=False)
        except Exception as e:
            self.logger.warning(f"Failed to save processed dataset: {e}")
        
        self.logger.info(f"Reports saved to: {self.output_dir}")