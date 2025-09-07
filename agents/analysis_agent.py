"""
Analysis Agent - Provides intelligent analysis and insights using LLM
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional
import os

try:
    from langchain_groq import ChatGroq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

from utils import generate_summary_stats, compare_dataframes
from config import MODEL_CONFIG

logger = logging.getLogger(__name__)

class AnalysisAgent:
    """Agent responsible for intelligent analysis and insights generation"""
    
    def __init__(self):
        self.name = "Analysis Agent"
        self.llm = self._initialize_llm()
        
    def _initialize_llm(self):
        """Initialize the language model"""
        try:
            # Try Groq first
            if GROQ_AVAILABLE and MODEL_CONFIG.groq_api_key:
                logger.info(f"{self.name}: Using ChatGroq Llama-3.1-8B-Instant")
                return ChatGroq(
                    groq_api_key=MODEL_CONFIG.groq_api_key,
                    model_name=MODEL_CONFIG.groq_model,
                    temperature=MODEL_CONFIG.groq_temperature
                )
            # Fallback to OpenAI
            # elif OPENAI_AVAILABLE and MODEL_CONFIG.openai_api_key:
            #     logger.info(f"{self.name}: Using OpenAI GPT-3.5-Turbo")
            #     return ChatOpenAI(
            #         openai_api_key=MODEL_CONFIG.openai_api_key,
            #         model_name=MODEL_CONFIG.openai_model,
            #         temperature=MODEL_CONFIG.openai_temperature
            #     )
            else:
                logger.warning(f"{self.name}: No LLM available, using rule-based analysis")
                return None
        except Exception as e:
            logger.warning(f"{self.name}: Failed to initialize LLM: {str(e)}")
            return None
    
    def analyze_raw_dataset(self, df: pd.DataFrame, inspection_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the raw dataset and identify vulnerabilities using LLM"""
        basic_stats = generate_summary_stats(df)
        quality_metrics = inspection_results.get('inspection_results', {}).get('quality_metrics', {})
        prompt = f"""
        You are a senior data analyst. Given the following dataset summary and inspection results, 
        identify vulnerabilities, provide recommendations, and summarize data quality.

        Dataset Overview:
        - Shape: {df.shape[0]} rows, {df.shape[1]} columns
        - Memory usage: {basic_stats['basic_info']['memory_usage_mb']:.1f} MB
        - Missing data: {quality_metrics.get('missing_percentage', 0):.1f}%
        - Data types: {quality_metrics.get('column_types', {})}

        Inspection Results:
        {inspection_results}

        Please respond with:
        1. Key vulnerabilities (bullet points)
        2. Recommendations (bullet points)
        3. Data quality score (0-100)
        4. Brief professional assessment (max 200 words)
        """
        response = self.llm.invoke(prompt)
        # Parse response (assuming markdown or structured text)
        analysis_result = {
            'dataset_overview': {
                'shape': df.shape,
                'memory_usage_mb': basic_stats['basic_info']['memory_usage_mb'],
                'data_types_summary': quality_metrics.get('column_types', {}),
                'missing_data_summary': f"{quality_metrics.get('missing_percentage', 0):.1f}% missing overall"
            },
            'llm_analysis': response.content
        }
        return analysis_result

    def analyze_cleaned_dataset(self, df_original: pd.DataFrame, df_cleaned: pd.DataFrame, 
                              cleaning_results: Dict[str, Any], transformation_results: Dict[str, Any],
                              verification_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the cleaned dataset and changes made using LLM"""
        comparison = compare_dataframes(df_original, df_cleaned)
        prompt = f"""
        You are a senior data analyst. Given the following cleaning and transformation results, 
        summarize improvements, remaining issues, and readiness for analysis.

        Original Dataset: {df_original.shape[0]} rows, {df_original.shape[1]} columns
        Cleaned Dataset: {df_cleaned.shape[0]} rows, {df_cleaned.shape[1]} columns
        Cleaning Actions: {cleaning_results.get('cleaning_actions', [])}
        Transformations: {transformation_results.get('transformations', [])}
        Verification Results: {verification_results}
        Data Comparison: {comparison}

        Please respond with:
        1. Improvements made (bullet points)
        2. Remaining issues (bullet points)
        3. Readiness assessment (bullet points)
        4. Final quality score (0-100)
        5. Recommendations for further improvement
        """
        response = self.llm.invoke(prompt)
        analysis_result = {
            'llm_analysis': response.content
        }
        return analysis_result

    def analyze_raw_dataset(self, df: pd.DataFrame, inspection_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the raw dataset and identify vulnerabilities"""
        
        # Generate basic statistics
        basic_stats = generate_summary_stats(df)
        
        # Extract key issues from inspection
        action_items = inspection_results.get('action_required', [])
        quality_metrics = inspection_results.get('inspection_results', {}).get('quality_metrics', {})
        
        # Rule-based analysis
        vulnerabilities = []
        recommendations = []
        
        # Missing data vulnerabilities
        missing_pct = quality_metrics.get('missing_percentage', 0)
        if missing_pct > 10:
            vulnerabilities.append(f"High missing data rate ({missing_pct:.1f}%) affects data reliability")
            recommendations.append("Implement robust missing data handling strategy before analysis")
        
        # Data type issues
        type_issues = inspection_results.get('inspection_results', {}).get('data_types', {}).get('potential_type_issues', [])
        if type_issues:
            vulnerabilities.append(f"Data type inconsistencies found in {len(type_issues)} columns")
            recommendations.append("Standardize data types to ensure accurate analysis")
        
        # Duplicate data
        duplicates = inspection_results.get('inspection_results', {}).get('duplicates', {}).get('total_duplicates', 0)
        if duplicates > 0:
            vulnerabilities.append(f"Contains {duplicates} duplicate records affecting data integrity")
            recommendations.append("Remove duplicates to prevent skewed analysis results")
        
        # Outliers
        outlier_summary = inspection_results.get('inspection_results', {}).get('outliers', {}).get('outlier_summary', {})
        outlier_cols = [col for col, info in outlier_summary.items() if info.get('count', 0) > 0]
        if outlier_cols:
            vulnerabilities.append(f"Outliers detected in {len(outlier_cols)} numeric columns")
            recommendations.append("Review and handle outliers that may skew statistical analysis")
        
        # High cardinality categorical variables
        high_card_cols = inspection_results.get('inspection_results', {}).get('categorical', {}).get('high_cardinality_columns', [])
        if high_card_cols:
            vulnerabilities.append(f"High cardinality categorical variables: {len(high_card_cols)} columns")
            recommendations.append("Consider grouping or encoding high cardinality categorical variables")
        
        analysis_result = {
            'dataset_overview': {
                'shape': df.shape,
                'memory_usage_mb': basic_stats['basic_info']['memory_usage_mb'],
                'data_types_summary': quality_metrics.get('column_types', {}),
                'missing_data_summary': f"{missing_pct:.1f}% missing overall"
            },
            'vulnerabilities': vulnerabilities,
            'recommendations': recommendations,
            'quality_score': self._calculate_raw_quality_score(inspection_results),
            'action_items': action_items[:10],  # Top 10 most critical
            'basic_statistics': basic_stats
        }
        
        # Add LLM-powered insights if available
        if self.llm:
            try:
                llm_insights = self._get_llm_insights_raw(df, analysis_result)
                analysis_result['llm_insights'] = llm_insights
            except Exception as e:
                logger.warning(f"{self.name}: LLM analysis failed: {str(e)}")
                analysis_result['llm_insights'] = "LLM analysis unavailable"
        
        return analysis_result
    
    def analyze_cleaned_dataset(self, df_original: pd.DataFrame, df_cleaned: pd.DataFrame, 
                              cleaning_results: Dict[str, Any], transformation_results: Dict[str, Any],
                              verification_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the cleaned dataset and changes made"""
        
        # Compare datasets
        comparison = compare_dataframes(df_original, df_cleaned)
        
        # Get cleaning and transformation actions
        cleaning_actions = cleaning_results.get('cleaning_actions', [])
        transformations = transformation_results.get('transformations', [])
        
        # Generate quality assessment
        quality_score = verification_results.get('quality_score', 0)
        verification_summary = verification_results.get('verification_results', {})
        
        # Rule-based analysis of improvements
        improvements = []
        remaining_issues = []
        
        # Assess missing data improvement
        missing_reduction = comparison.get('missing_reduction_pct', 0)
        if missing_reduction > 0:
            improvements.append(f"Reduced missing data by {missing_reduction:.1f}%")
        
        # Assess duplicate removal
        if comparison.get('rows_changed', 0) < 0:
            rows_removed = abs(comparison['rows_changed'])
            improvements.append(f"Removed {rows_removed} problematic rows (duplicates/outliers)")
        
        # Assess transformations
        if transformations:
            improvements.append(f"Applied {len(transformations)} data transformations for analysis readiness")
        
        # Check remaining issues
        total_issues = verification_results.get('verification_results', {}).get('overall_assessment', {}).get('total_issues', 0)
        total_warnings = verification_results.get('verification_results', {}).get('overall_assessment', {}).get('total_warnings', 0)
        
        if total_issues > 0:
            remaining_issues.append(f"{total_issues} critical issues still present")
        if total_warnings > 0:
            remaining_issues.append(f"{total_warnings} warnings require attention")
        
        # Generate recommendations for further improvement
        future_recommendations = []
        
        if quality_score < 80:
            future_recommendations.append("Consider additional data quality improvements before analysis")
        
        # Check for potential analysis opportunities
        numeric_cols = len(df_cleaned.select_dtypes(include=[np.number]).columns)
        categorical_cols = len(df_cleaned.select_dtypes(include=['object', 'category']).columns)
        
        if numeric_cols >= 2:
            future_recommendations.append("Dataset ready for correlation analysis and regression modeling")
        if categorical_cols > 0 and numeric_cols > 0:
            future_recommendations.append("Good mix of categorical and numeric variables for comprehensive analysis")
        if df_cleaned.shape[0] > 1000:
            future_recommendations.append("Dataset size suitable for machine learning applications")
        
        analysis_result = {
            'improvement_summary': {
                'original_shape': comparison['shape_before'],
                'final_shape': comparison['shape_after'],
                'quality_improvement': f"Score improved to {quality_score:.1f}/100",
                'actions_applied': len(cleaning_actions) + len(transformations)
            },
            'changes_made': {
                'cleaning_actions': cleaning_actions,
                'transformations': transformations,
                'data_comparison': comparison
            },
            'improvements': improvements,
            'remaining_issues': remaining_issues,
            'future_recommendations': future_recommendations,
            'readiness_assessment': self._assess_analysis_readiness(df_cleaned, quality_score),
            'final_quality_score': quality_score
        }
        
        # Add LLM-powered insights if available
        if self.llm:
            try:
                llm_insights = self._get_llm_insights_cleaned(df_original, df_cleaned, analysis_result)
                analysis_result['llm_insights'] = llm_insights
            except Exception as e:
                logger.warning(f"{self.name}: LLM analysis failed: {str(e)}")
                analysis_result['llm_insights'] = "LLM analysis unavailable"
        
        return analysis_result
    
    
    def _calculate_raw_quality_score(self, inspection_results: Dict[str, Any]) -> float:
        """Calculate quality score for raw dataset"""
        score = 100.0
        
        # Penalize based on issues found
        action_items = len(inspection_results.get('action_required', []))
        score -= min(50, action_items * 5)  # Max 50 points penalty
        
        # Additional penalties for specific issues
        quality_metrics = inspection_results.get('inspection_results', {}).get('quality_metrics', {})
        missing_pct = quality_metrics.get('missing_percentage', 0)
        score -= min(30, missing_pct)  # Max 30 points for missing data
        
        return max(0, score)
    
    def _assess_analysis_readiness(self, df: pd.DataFrame, quality_score: float) -> Dict[str, Any]:
        """Assess if dataset is ready for analysis"""
        readiness = {
            'overall_ready': quality_score >= 70,
            'quality_score': quality_score,
            'readiness_factors': []
        }
        
        # Check various readiness factors
        if df.isnull().sum().sum() == 0:
            readiness['readiness_factors'].append("✅ No missing values")
        else:
            readiness['readiness_factors'].append("⚠️ Some missing values remain")
        
        if len(df.select_dtypes(include=[np.number]).columns) > 0:
            readiness['readiness_factors'].append("✅ Numeric variables available for analysis")
        
        if len(df.select_dtypes(include=['object', 'category']).columns) > 0:
            readiness['readiness_factors'].append("✅ Categorical variables available for segmentation")
        
        if df.shape[0] >= 100:
            readiness['readiness_factors'].append("✅ Sufficient sample size for analysis")
        else:
            readiness['readiness_factors'].append("⚠️ Small sample size may limit analysis")
        
        if quality_score >= 90:
            readiness['recommendation'] = "Dataset is excellent quality and ready for advanced analytics"
        elif quality_score >= 70:
            readiness['recommendation'] = "Dataset is good quality and ready for most analyses"
        elif quality_score >= 50:
            readiness['recommendation'] = "Dataset needs some improvements but can be used for basic analysis"
        else:
            readiness['recommendation'] = "Dataset requires significant improvements before analysis"
        
        return readiness
    
    def _get_llm_insights_raw(self, df: pd.DataFrame, analysis_result: Dict[str, Any]) -> str:
        """Get LLM-powered insights for raw dataset"""
        
        prompt = f"""
        As a senior data analyst, analyze this raw dataset and provide professional insights:

        Dataset Overview:
        - Shape: {df.shape[0]} rows, {df.shape[1]} columns
        - Missing data: {analysis_result['dataset_overview']['missing_data_summary']}
        - Memory usage: {analysis_result['dataset_overview']['memory_usage_mb']:.1f} MB

        Key Issues Found:
        {chr(10).join(['- ' + vuln for vuln in analysis_result['vulnerabilities']])}

        Column Information:
        {chr(10).join([f"- {col}: {dtype}" for col, dtype in list(df.dtypes.astype(str).items())[:10]])}

        Provide a brief professional assessment (max 200 words) covering:
        1. Overall data quality assessment
        2. Main risks for analysis
        3. Priority actions needed
        4. Potential analytical opportunities

        Write as a data analyst would for stakeholders.
        """
        
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            logger.error(f"LLM analysis failed: {str(e)}")
            return f"LLM analysis failed: {str(e)}"
    
    def _get_llm_insights_cleaned(self, df_original: pd.DataFrame, df_cleaned: pd.DataFrame, 
                                 analysis_result: Dict[str, Any]) -> str:
        """Get LLM-powered insights for cleaned dataset"""
        
        prompt = f"""
        As a senior data analyst, analyze the data cleaning results and provide professional insights:

        Dataset Transformation:
        - Original: {df_original.shape[0]} rows, {df_original.shape[1]} columns
        - Cleaned: {df_cleaned.shape[0]} rows, {df_cleaned.shape[1]} columns
        - Final quality score: {analysis_result['final_quality_score']:.1f}/100

        Actions Performed:
        {chr(10).join(['- ' + action for action in analysis_result['changes_made']['cleaning_actions'][:5]])}

        Improvements Made:
        {chr(10).join(['- ' + improvement for improvement in analysis_result['improvements']])}

        Remaining Issues:
        {chr(10).join(['- ' + issue for issue in analysis_result['remaining_issues']])}

        Provide a brief professional assessment (max 200 words) covering:
        1. Quality of the cleaning process
        2. Dataset readiness for analysis
        3. Recommended next steps
        4. Analysis opportunities with cleaned data

        Write as a data analyst would for stakeholders.
        """
        
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            logger.error(f"LLM analysis failed: {str(e)}")
            return f"LLM analysis failed: {str(e)}"
    
    def generate_executive_summary(self, raw_analysis: Dict[str, Any], 
                                 cleaned_analysis: Dict[str, Any]) -> str:
        """Generate executive summary of the entire analysis"""
        
        summary_lines = []
        summary_lines.append("=== EXECUTIVE SUMMARY ===\n")
        
        # Raw dataset assessment
        raw_score = raw_analysis.get('quality_score', 0)
        final_score = cleaned_analysis.get('final_quality_score', 0)
        
        summary_lines.append(f"📊 DATASET QUALITY TRANSFORMATION")
        summary_lines.append(f"   • Initial Quality Score: {raw_score:.1f}/100")
        summary_lines.append(f"   • Final Quality Score: {final_score:.1f}/100")
        summary_lines.append(f"   • Improvement: +{final_score - raw_score:.1f} points\n")
        
        # Key vulnerabilities addressed
        vulnerabilities = len(raw_analysis.get('vulnerabilities', []))
        improvements = len(cleaned_analysis.get('improvements', []))
        
        summary_lines.append(f"🔧 ACTIONS TAKEN")
        summary_lines.append(f"   • {vulnerabilities} vulnerabilities identified")
        summary_lines.append(f"   • {improvements} improvements implemented")
        
        actions_count = cleaned_analysis.get('improvement_summary', {}).get('actions_applied', 0)
        summary_lines.append(f"   • {actions_count} total cleaning/transformation actions\n")
        
        # Readiness assessment
        readiness = cleaned_analysis.get('readiness_assessment', {})
        is_ready = readiness.get('overall_ready', False)
        
        summary_lines.append(f"✅ ANALYSIS READINESS")
        status = "READY" if is_ready else "NEEDS IMPROVEMENT"
        summary_lines.append(f"   • Status: {status}")
        summary_lines.append(f"   • {readiness.get('recommendation', 'No recommendation available')}\n")
        
        # Top recommendations
        future_recs = cleaned_analysis.get('future_recommendations', [])[:3]
        if future_recs:
            summary_lines.append(f"🎯 KEY RECOMMENDATIONS")
            for i, rec in enumerate(future_recs, 1):
                summary_lines.append(f"   {i}. {rec}")
        
        return "\n".join(summary_lines)
    
    def process(self, df_original: pd.DataFrame, df_cleaned: pd.DataFrame,
                inspection_results: Dict[str, Any], cleaning_results: Dict[str, Any],
                transformation_results: Dict[str, Any], verification_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main processing method for the Analysis Agent
        
        Args:
            df_original: Original raw dataset
            df_cleaned: Cleaned and transformed dataset
            inspection_results: Results from inspection agent
            cleaning_results: Results from cleaning agent
            transformation_results: Results from transformation agent
            verification_results: Results from verification agent
            
        Returns:
            Complete analysis results
        """
        logger.info(f"{self.name}: Starting comprehensive dataset analysis")
        
        try:
            # Analyze raw dataset
            logger.info(f"{self.name}: Analyzing raw dataset vulnerabilities...")
            raw_analysis = self.analyze_raw_dataset(df_original, inspection_results)
            
            # Analyze cleaned dataset
            logger.info(f"{self.name}: Analyzing cleaned dataset improvements...")
            cleaned_analysis = self.analyze_cleaned_dataset(
                df_original, df_cleaned, cleaning_results, 
                transformation_results, verification_results
            )
            
            # Generate executive summary
            logger.info(f"{self.name}: Generating executive summary...")
            executive_summary = self.generate_executive_summary(raw_analysis, cleaned_analysis)
            
            result = {
                'agent': self.name,
                'status': 'success',
                'raw_dataset_analysis': raw_analysis,
                'cleaned_dataset_analysis': cleaned_analysis,
                'executive_summary': executive_summary,
                'llm_available': self.llm is not None,
                'message': f"Analysis completed. Dataset quality improved from {raw_analysis.get('quality_score', 0):.1f} to {cleaned_analysis.get('final_quality_score', 0):.1f}"
            }
            
            logger.info(f"{self.name}: Analysis completed successfully")
            
            return result
            
        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            logger.error(f"{self.name}: {error_msg}")
            
            return {
                'agent': self.name,
                'status': 'error',
                'message': error_msg,
                'error': str(e)
            }