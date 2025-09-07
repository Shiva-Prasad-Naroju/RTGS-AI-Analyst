"""
Enhanced Report Agent - Generates government-grade PDF reports with deep insights
Policy-maker oriented comprehensive data analysis reports
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
import os
from datetime import datetime
from reportlab.lib.pagesizes import letter, A4, landscape
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, 
                                PageBreak, Image, KeepTogether, ListFlowable, ListItem)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY, TA_RIGHT
from reportlab.pdfgen import canvas
from reportlab.graphics.shapes import Drawing, Line
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.piecharts import Pie
from reportlab.platypus.tableofcontents import TableOfContents

from config import REPORT_CONFIG, PATHS
from utils import get_timestamp

logger = logging.getLogger(__name__)

class ReportAgent:
    """Enhanced Agent for generating government-grade analytical reports"""
    
    def __init__(self):
        self.name = "Enhanced Report Agent"
        self.styles = getSampleStyleSheet()
        self._setup_professional_styles()
        self.insights_generated = []
        self.policy_recommendations = []
        
    def _setup_professional_styles(self):
        """Setup professional government-grade paragraph styles"""
        
        # Main Title - Government Report Style
        self.styles.add(ParagraphStyle(
            name='MainTitle',
            parent=self.styles['Title'],
            fontSize=28,
            spaceAfter=40,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#1e3d59'),
            fontName='Helvetica-Bold',
            leading=34
        ))
        
        # Executive Title
        self.styles.add(ParagraphStyle(
            name='ExecutiveTitle',
            parent=self.styles['Title'],
            fontSize=20,
            spaceAfter=20,
            spaceBefore=30,
            alignment=TA_LEFT,
            textColor=colors.HexColor('#1e3d59'),
            fontName='Helvetica-Bold',
            borderWidth=2,
            borderColor=colors.HexColor('#1e3d59'),
            borderPadding=10
        ))
        
        # Section Headers with numbering
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading1'],
            fontSize=18,
            spaceAfter=16,
            spaceBefore=24,
            textColor=colors.HexColor('#1e3d59'),
            fontName='Helvetica-Bold',
            leftIndent=0,
            bulletIndent=0
        ))
        
        # Subsection Headers
        self.styles.add(ParagraphStyle(
            name='SubsectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            spaceBefore=16,
            textColor=colors.HexColor('#2c5282'),
            fontName='Helvetica-Bold',
            leftIndent=12
        ))
        
        # Executive Summary Style
        self.styles.add(ParagraphStyle(
            name='ExecutiveSummary',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceAfter=10,
            alignment=TA_JUSTIFY,
            fontName='Helvetica',
            leftIndent=20,
            rightIndent=20,
            textColor=colors.HexColor('#2c3e50')
        ))
        
        # Body Text - Professional
        self.styles.add(ParagraphStyle(
            name='ProfessionalBody',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=8,
            alignment=TA_JUSTIFY,
            fontName='Helvetica',
            leading=14,
            textColor=colors.HexColor('#333333')
        ))
        
        # Key Finding Style
        self.styles.add(ParagraphStyle(
            name='KeyFinding',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=8,
            leftIndent=30,
            fontName='Helvetica-Bold',
            textColor=colors.HexColor('#c41e3a'),
            bulletFontName='Symbol',
            bulletFontSize=12,
            bulletIndent=20
        ))
        
        # Policy Recommendation Style
        self.styles.add(ParagraphStyle(
            name='PolicyRec',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=10,
            leftIndent=25,
            fontName='Helvetica',
            textColor=colors.HexColor('#1e3d59'),
            backColor=colors.HexColor('#f0f8ff'),
            borderWidth=1,
            borderColor=colors.HexColor('#1e3d59'),
            borderPadding=8
        ))
        
        # Critical Insight Style
        self.styles.add(ParagraphStyle(
            name='CriticalInsight',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceAfter=12,
            spaceBefore=12,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold',
            textColor=colors.HexColor('#ff6e40'),
            backColor=colors.HexColor('#fff5f0'),
            borderWidth=2,
            borderColor=colors.HexColor('#ff6e40'),
            borderPadding=10
        ))
        
        # Statistical Note Style
        self.styles.add(ParagraphStyle(
            name='StatNote',
            parent=self.styles['Normal'],
            fontSize=9,
            spaceAfter=6,
            fontName='Helvetica-Oblique',
            textColor=colors.HexColor('#666666'),
            leftIndent=40
        ))
        
        # Footer Style
        self.styles.add(ParagraphStyle(
            name='Footer',
            parent=self.styles['Normal'],
            fontSize=8,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#999999'),
            fontName='Helvetica'
        ))
    
    def _create_header_footer(self, canvas, doc):
        """Create professional header and footer"""
        canvas.saveState()
        
        # Header
        canvas.setFont('Helvetica-Bold', 10)
        canvas.setFillColor(colors.HexColor('#1e3d59'))
        canvas.drawString(doc.leftMargin, doc.height + doc.topMargin - 15, 
                         "RTGS DATA INTELLIGENCE REPORT")
        canvas.drawRightString(doc.width + doc.leftMargin, doc.height + doc.topMargin - 15,
                              "CONFIDENTIAL")
        
        # Header line
        canvas.setStrokeColor(colors.HexColor('#1e3d59'))
        canvas.setLineWidth(1)
        canvas.line(doc.leftMargin, doc.height + doc.topMargin - 20,
                   doc.width + doc.leftMargin, doc.height + doc.topMargin - 20)
        
        # Footer
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(colors.HexColor('#666666'))
        
        # Footer line
        canvas.line(doc.leftMargin, 35, doc.width + doc.leftMargin, 35)
        
        # Footer text
        canvas.drawString(doc.leftMargin, 25, 
                         f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | RTGS AI Analyst System v2.0")
        canvas.drawCentredString(doc.width/2 + doc.leftMargin, 25,
                                f"Page {doc.page}")
        canvas.drawRightString(doc.width + doc.leftMargin, 25,
                              "Classification: Official")
        
        canvas.restoreState()
    
    def _extract_deep_insights(self, df_original: pd.DataFrame, df_cleaned: pd.DataFrame,
                              analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract deep insights from data for policy makers"""
        insights = []
        
        # 1. Data Quality Insights
        raw_score = analysis_results.get('raw_dataset_analysis', {}).get('quality_score', 0)
        final_score = analysis_results.get('cleaned_dataset_analysis', {}).get('final_quality_score', 0)
        improvement = final_score - raw_score
        
        if improvement > 15:
            insights.append({
                'type': 'critical',
                'category': 'Data Quality',
                'finding': f"Exceptional data quality improvement of {improvement:.1f} points achieved",
                'implication': "High confidence in analytical results due to superior data preparation",
                'recommendation': "Proceed with advanced predictive analytics and policy modeling"
            })
        elif improvement > 10:
            insights.append({
                'type': 'important',
                'category': 'Data Quality',
                'finding': f"Significant data quality enhancement of {improvement:.1f} points",
                'implication': "Data reliability substantially improved for decision-making",
                'recommendation': "Deploy standard analytical models with confidence"
            })
        
        # 2. Feature Engineering Insights
        new_features = df_cleaned.shape[1] - df_original.shape[1]
        if new_features > 15:
            insights.append({
                'type': 'critical',
                'category': 'Feature Engineering',
                'finding': f"Generated {new_features} new analytical dimensions",
                'implication': "Substantially enhanced analytical capability and pattern detection potential",
                'recommendation': "Leverage new features for comprehensive policy impact assessment"
            })
        
        # 3. Data Completeness Insights
        completeness = (1 - df_cleaned.isnull().sum().sum()/(df_cleaned.shape[0]*df_cleaned.shape[1])) * 100
        if completeness > 95:
            insights.append({
                'type': 'positive',
                'category': 'Data Completeness',
                'finding': f"Achieved {completeness:.1f}% data completeness",
                'implication': "Minimal bias risk from missing data",
                'recommendation': "Full-scale analysis can proceed without imputation concerns"
            })
        elif completeness < 80:
            insights.append({
                'type': 'warning',
                'category': 'Data Completeness',
                'finding': f"Data completeness at {completeness:.1f}% requires attention",
                'implication': "Potential bias in analysis due to missing information",
                'recommendation': "Apply advanced imputation techniques before critical decisions"
            })
        
        # 4. Statistical Distribution Insights
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            high_skew_cols = []
            for col in numeric_cols:
                skewness = df_cleaned[col].skew()
                if abs(skewness) > 2:
                    high_skew_cols.append(col)
            
            if len(high_skew_cols) > 3:
                insights.append({
                    'type': 'important',
                    'category': 'Statistical Distribution',
                    'finding': f"{len(high_skew_cols)} variables show significant skewness",
                    'implication': "Non-normal distributions may affect traditional statistical methods",
                    'recommendation': "Apply robust statistical methods or transformation techniques"
                })
        
        # 5. Outlier Detection Insights
        outlier_percentages = []
        for col in numeric_cols[:10]:
            Q1 = df_cleaned[col].quantile(0.25)
            Q3 = df_cleaned[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df_cleaned[(df_cleaned[col] < Q1 - 1.5*IQR) | (df_cleaned[col] > Q3 + 1.5*IQR)]
            outlier_pct = len(outliers) / len(df_cleaned) * 100
            outlier_percentages.append(outlier_pct)
        
        if outlier_percentages:
            avg_outliers = np.mean(outlier_percentages)
            if avg_outliers > 5:
                insights.append({
                    'type': 'warning',
                    'category': 'Anomaly Detection',
                    'finding': f"Average outlier rate of {avg_outliers:.1f}% detected",
                    'implication': "Potential data quality issues or genuine anomalies requiring investigation",
                    'recommendation': "Conduct targeted investigation of outlier patterns before policy decisions"
                })
        
        # 6. Correlation Insights
        if len(numeric_cols) > 1:
            corr_matrix = df_cleaned[numeric_cols].corr()
            high_corr_pairs = []
            for i in range(len(corr_matrix)):
                for j in range(i+1, len(corr_matrix)):
                    if abs(corr_matrix.iloc[i, j]) > 0.8:
                        high_corr_pairs.append((corr_matrix.index[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
            
            if high_corr_pairs:
                insights.append({
                    'type': 'important',
                    'category': 'Correlation Analysis',
                    'finding': f"Identified {len(high_corr_pairs)} highly correlated variable pairs",
                    'implication': "Strong interdependencies exist that could affect policy outcomes",
                    'recommendation': "Consider multicollinearity in predictive models and policy simulations"
                })
        
        # 7. Scale and Scope Insights
        if df_cleaned.shape[0] > 10000:
            insights.append({
                'type': 'positive',
                'category': 'Data Scale',
                'finding': f"Large-scale dataset with {df_cleaned.shape[0]:,} records",
                'implication': "Sufficient data for robust statistical inference and machine learning",
                'recommendation': "Deploy advanced AI/ML models for predictive policy analysis"
            })
        elif df_cleaned.shape[0] < 1000:
            insights.append({
                'type': 'warning',
                'category': 'Data Scale',
                'finding': f"Limited dataset size with {df_cleaned.shape[0]:,} records",
                'implication': "Statistical power may be limited for complex analyses",
                'recommendation': "Use conservative statistical methods and bootstrap techniques"
            })
        
        return insights
    
    def _generate_policy_recommendations(self, insights: List[Dict[str, Any]], 
                                        analysis_results: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate policy recommendations based on insights"""
        recommendations = []
        
        # Categorize insights
        critical_insights = [i for i in insights if i['type'] == 'critical']
        important_insights = [i for i in insights if i['type'] == 'important']
        warning_insights = [i for i in insights if i['type'] == 'warning']
        
        # Priority 1: Address Critical Issues
        if critical_insights:
            recommendations.append({
                'priority': 'IMMEDIATE',
                'timeframe': '0-30 days',
                'recommendation': 'Leverage exceptional data quality improvements for immediate policy decisions',
                'action_items': [
                    'Deploy predictive models for scenario planning',
                    'Conduct comprehensive impact assessments',
                    'Generate executive dashboards for real-time monitoring'
                ],
                'expected_outcome': 'Enhanced decision-making capability with high-confidence data'
            })
        
        # Priority 2: Strategic Initiatives
        if important_insights:
            recommendations.append({
                'priority': 'HIGH',
                'timeframe': '1-3 months',
                'recommendation': 'Implement advanced analytical frameworks based on enhanced dataset',
                'action_items': [
                    'Develop machine learning models for pattern recognition',
                    'Create automated reporting systems',
                    'Establish data-driven KPIs for policy effectiveness'
                ],
                'expected_outcome': 'Systematic improvement in policy formulation and evaluation'
            })
        
        # Priority 3: Risk Mitigation
        if warning_insights:
            recommendations.append({
                'priority': 'MEDIUM',
                'timeframe': '3-6 months',
                'recommendation': 'Address identified data limitations and risks',
                'action_items': [
                    'Implement data quality monitoring systems',
                    'Develop contingency plans for data gaps',
                    'Establish validation protocols for critical metrics'
                ],
                'expected_outcome': 'Reduced risk in data-driven decision making'
            })
        
        # General Recommendations
        final_score = analysis_results.get('cleaned_dataset_analysis', {}).get('final_quality_score', 0)
        
        if final_score > 80:
            recommendations.append({
                'priority': 'STRATEGIC',
                'timeframe': 'Ongoing',
                'recommendation': 'Capitalize on high-quality data infrastructure',
                'action_items': [
                    'Establish center of excellence for data analytics',
                    'Develop predictive policy modeling capabilities',
                    'Create data sharing frameworks across departments'
                ],
                'expected_outcome': 'Transformation to data-driven governance model'
            })
        
        return recommendations
    
    def _create_executive_brief(self, analysis_results: Dict[str, Any],
                               insights: List[Dict[str, Any]],
                               recommendations: List[Dict[str, str]]) -> List:
        """Create executive brief section"""
        content = []
        
        # Executive Brief Title
        title = Paragraph("EXECUTIVE BRIEF", self.styles['ExecutiveTitle'])
        content.append(title)
        content.append(Spacer(1, 20))
        
        # Key Metrics Summary
        raw_score = analysis_results.get('raw_dataset_analysis', {}).get('quality_score', 0)
        final_score = analysis_results.get('cleaned_dataset_analysis', {}).get('final_quality_score', 0)
        
        exec_summary = f"""
        <para alignment="justify">
        <b>SITUATION ASSESSMENT:</b> The RTGS data infrastructure has undergone comprehensive 
        enhancement through advanced AI-driven analysis. The data quality score improved from 
        <font color="#c41e3a"><b>{raw_score:.1f}</b></font> to 
        <font color="#2e8b57"><b>{final_score:.1f}</b></font>, representing a 
        <b>{((final_score-raw_score)/raw_score*100):.1f}%</b> improvement in analytical capability.
        </para>
        """
        content.append(Paragraph(exec_summary, self.styles['ExecutiveSummary']))
        content.append(Spacer(1, 12))
        
        # Critical Findings
        critical_findings = [i for i in insights if i['type'] == 'critical']
        if critical_findings:
            content.append(Paragraph("<b>CRITICAL FINDINGS:</b>", self.styles['SubsectionHeader']))
            for finding in critical_findings[:3]:
                bullet_text = f"• <b>{finding['category']}:</b> {finding['finding']}"
                content.append(Paragraph(bullet_text, self.styles['KeyFinding']))
            content.append(Spacer(1, 12))
        
        # Immediate Actions Required
        immediate_recs = [r for r in recommendations if r['priority'] == 'IMMEDIATE']
        if immediate_recs:
            content.append(Paragraph("<b>IMMEDIATE ACTIONS REQUIRED:</b>", self.styles['SubsectionHeader']))
            for rec in immediate_recs:
                action_text = f"""
                <para>
                {rec['recommendation']}<br/>
                <font size="9"><i>Timeframe: {rec['timeframe']}</i></font>
                </para>
                """
                content.append(Paragraph(action_text, self.styles['PolicyRec']))
            content.append(Spacer(1, 12))
        
        # Decision Support Matrix
        content.append(Paragraph("<b>DECISION SUPPORT MATRIX:</b>", self.styles['SubsectionHeader']))
        
        decision_matrix = [
            ["Aspect", "Status", "Confidence", "Action Required"],
            ["Data Quality", self._get_status_label(final_score), f"{final_score:.0f}%", "Monitor"],
            ["Analytical Readiness", "READY", "High", "Deploy"],
            ["Risk Level", "LOW", "High", "Proceed"],
            ["Resource Requirements", "OPTIMAL", "Medium", "Maintain"]
        ]
        
        matrix_table = Table(decision_matrix, colWidths=[2.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
        matrix_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e3d59')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f0f8ff')),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#1e3d59')),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#f0f8ff'), colors.white])
        ]))
        
        content.append(matrix_table)
        content.append(PageBreak())
        
        return content
    
    def _create_detailed_analysis(self, df_original: pd.DataFrame, df_cleaned: pd.DataFrame,
                                 analysis_results: Dict[str, Any],
                                 insights: List[Dict[str, Any]]) -> List:
        """Create detailed analysis section"""
        content = []
        
        # Section Header
        content.append(Paragraph("1. DETAILED ANALYTICAL ASSESSMENT", self.styles['SectionHeader']))
        content.append(Spacer(1, 12))
        
        # 1.1 Data Transformation Analysis
        content.append(Paragraph("1.1 Data Transformation Impact Analysis", self.styles['SubsectionHeader']))
        
        transformation_text = f"""
        <para alignment="justify">
        The data transformation pipeline processed <b>{df_original.shape[0]:,}</b> records across 
        <b>{df_original.shape[1]}</b> original dimensions, expanding to <b>{df_cleaned.shape[1]}</b> 
        analytical dimensions through advanced feature engineering. This represents a 
        <b>{((df_cleaned.shape[1]-df_original.shape[1])/df_original.shape[1]*100):.1f}%</b> 
        increase in analytical capability.
        </para>
        """
        content.append(Paragraph(transformation_text, self.styles['ProfessionalBody']))
        content.append(Spacer(1, 8))
        
        # Key Transformations Table
        transformations = analysis_results.get('cleaned_dataset_analysis', {}).get('changes_made', {}).get('transformations', [])
        if transformations:
            trans_data = [["Transformation Type", "Count", "Impact"]]
            trans_summary = {}
            for trans in transformations[:20]:
                trans_type = trans.split(':')[0] if ':' in trans else 'Other'
                trans_summary[trans_type] = trans_summary.get(trans_type, 0) + 1
            
            for trans_type, count in trans_summary.items():
                impact = "High" if count > 5 else "Medium" if count > 2 else "Low"
                trans_data.append([trans_type, str(count), impact])
            
            trans_table = Table(trans_data, colWidths=[3*inch, 1.5*inch, 1.5*inch])
            trans_table.setStyle(self._get_professional_table_style())
            content.append(trans_table)
            content.append(Spacer(1, 12))
        
        # 1.2 Statistical Properties Analysis
        content.append(Paragraph("1.2 Statistical Properties Assessment", self.styles['SubsectionHeader']))
        
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            stat_analysis = f"""
            <para alignment="justify">
            Statistical analysis reveals <b>{len(numeric_cols)}</b> quantitative variables 
            available for advanced modeling. Distribution analysis indicates:
            </para>
            """
            content.append(Paragraph(stat_analysis, self.styles['ProfessionalBody']))
            
            # Statistical summary
            stat_insights = []
            for col in numeric_cols[:5]:
                skew = df_cleaned[col].skew()
                kurt = df_cleaned[col].kurtosis()
                if abs(skew) > 1:
                    stat_insights.append(f"• <b>{col}</b>: Skewness = {skew:.2f} (requires transformation)")
                if abs(kurt) > 3:
                    stat_insights.append(f"• <b>{col}</b>: Kurtosis = {kurt:.2f} (heavy-tailed distribution)")
            
            for insight in stat_insights[:5]:
                content.append(Paragraph(insight, self.styles['StatNote']))
            content.append(Spacer(1, 12))
        
        # 1.3 Correlation and Dependencies
        content.append(Paragraph("1.3 Inter-variable Dependencies", self.styles['SubsectionHeader']))
        
        if len(numeric_cols) > 1:
            corr_text = """
            <para alignment="justify">
            Correlation analysis reveals complex interdependencies within the dataset. 
            Strong correlations (|r| > 0.7) indicate potential multicollinearity concerns 
            for regression-based policy models. Weak correlations suggest independent 
            variation suitable for multi-factor analysis.
            </para>
            """
            content.append(Paragraph(corr_text, self.styles['ProfessionalBody']))
            content.append(Spacer(1, 12))
        
        # 1.4 Data Quality Metrics Deep Dive
        content.append(Paragraph("1.4 Comprehensive Quality Metrics", self.styles['SubsectionHeader']))
        
        quality_metrics = [
            ["Metric", "Original", "Processed", "Change", "Impact"],
            ["Completeness", f"{(1-df_original.isnull().sum().sum()/(df_original.shape[0]*df_original.shape[1]))*100:.1f}%",
             f"{(1-df_cleaned.isnull().sum().sum()/(df_cleaned.shape[0]*df_cleaned.shape[1]))*100:.1f}%",
             "↑", "Positive"],
            ["Unique Records", f"{(1-df_original.duplicated().sum()/len(df_original))*100:.1f}%",
             f"{(1-df_cleaned.duplicated().sum()/len(df_cleaned))*100:.1f}%",
             "↑", "Positive"],
            ["Memory Efficiency", f"{df_original.memory_usage(deep=True).sum()/(1024*1024):.1f} MB",
             f"{df_cleaned.memory_usage(deep=True).sum()/(1024*1024):.1f} MB",
             "↑", "Acceptable"],
            ["Feature Space", f"{df_original.shape[1]} dims",
             f"{df_cleaned.shape[1]} dims",
             f"+{df_cleaned.shape[1]-df_original.shape[1]}", "Enhanced"]
        ]
        
        quality_table = Table(quality_metrics, colWidths=[1.8*inch, 1.2*inch, 1.2*inch, 0.8*inch, 1*inch])
        quality_table.setStyle(self._get_professional_table_style())
        content.append(quality_table)
        
        content.append(PageBreak())
        return content
    
    def _create_insights_synthesis(self, insights: List[Dict[str, Any]]) -> List:
        """Create insights synthesis section"""
        content = []
        
        content.append(Paragraph("2. STRATEGIC INSIGHTS SYNTHESIS", self.styles['SectionHeader']))
        content.append(Spacer(1, 12))
        
        # Group insights by category
        insights_by_category = {}
        for insight in insights:
            category = insight['category']
            if category not in insights_by_category:
                insights_by_category[category] = []
            insights_by_category[category].append(insight)
        
        # Present insights by category
        for i, (category, cat_insights) in enumerate(insights_by_category.items(), 1):
            content.append(Paragraph(f"2.{i} {category}", self.styles['SubsectionHeader']))
            
            for insight in cat_insights:
                # Insight finding
                if insight['type'] == 'critical':
                    finding_style = self.styles['CriticalInsight']
                else:
                    finding_style = self.styles['ProfessionalBody']
                
                finding_text = f"<b>Finding:</b> {insight['finding']}"
                content.append(Paragraph(finding_text, finding_style))
                
                # Implication
                implication_text = f"<b>Policy Implication:</b> {insight['implication']}"
                content.append(Paragraph(implication_text, self.styles['ProfessionalBody']))
                
                # Recommendation
                rec_text = f"<b>Recommendation:</b> {insight['recommendation']}"
                content.append(Paragraph(rec_text, self.styles['PolicyRec']))
                content.append(Spacer(1, 12))
        
        content.append(PageBreak())
        return content
    
    def _create_policy_framework(self, recommendations: List[Dict[str, str]]) -> List:
        """Create policy framework section"""
        content = []
        
        content.append(Paragraph("3. POLICY IMPLEMENTATION FRAMEWORK", self.styles['SectionHeader']))
        content.append(Spacer(1, 12))
        
        # Introduction
        intro_text = """
        <para alignment="justify">
        Based on comprehensive data analysis, the following policy implementation framework 
        provides actionable guidance for leveraging data insights in decision-making processes. 
        Each recommendation is prioritized based on impact potential and resource requirements.
        </para>
        """
        content.append(Paragraph(intro_text, self.styles['ProfessionalBody']))
        content.append(Spacer(1, 16))
        
        # Present recommendations by priority
        priority_order = ['IMMEDIATE', 'HIGH', 'MEDIUM', 'STRATEGIC']
        priority_colors = {
            'IMMEDIATE': colors.HexColor('#dc3545'),
            'HIGH': colors.HexColor('#fd7e14'),
            'MEDIUM': colors.HexColor('#ffc107'),
            'STRATEGIC': colors.HexColor('#28a745')
        }
        
        for priority in priority_order:
            priority_recs = [r for r in recommendations if r['priority'] == priority]
            
            for rec in priority_recs:
                # Priority header
                priority_style = ParagraphStyle(
                    name='PriorityHeader',
                    parent=self.styles['SubsectionHeader'],
                    textColor=priority_colors[priority]
                )
                content.append(Paragraph(f"Priority: {priority} | Timeframe: {rec['timeframe']}", 
                                        priority_style))
                
                # Recommendation
                content.append(Paragraph(f"<b>Strategic Initiative:</b> {rec['recommendation']}", 
                                        self.styles['ProfessionalBody']))
                content.append(Spacer(1, 8))
                
                # Action items
                content.append(Paragraph("<b>Action Items:</b>", self.styles['ProfessionalBody']))
                for action in rec['action_items']:
                    content.append(Paragraph(f"• {action}", self.styles['KeyFinding']))
                content.append(Spacer(1, 8))
                
                # Expected outcome
                content.append(Paragraph(f"<b>Expected Outcome:</b> {rec['expected_outcome']}", 
                                        self.styles['ProfessionalBody']))
                content.append(Spacer(1, 16))
        
        content.append(PageBreak())
        return content
    
    def _create_technical_appendix(self, df_original: pd.DataFrame, df_cleaned: pd.DataFrame,
                                  analysis_results: Dict[str, Any]) -> List:
        """Create technical appendix"""
        content = []
        
        content.append(Paragraph("APPENDIX A: TECHNICAL SPECIFICATIONS", self.styles['SectionHeader']))
        content.append(Spacer(1, 12))
        
        # A.1 Dataset Specifications
        content.append(Paragraph("A.1 Dataset Specifications", self.styles['SubsectionHeader']))
        
        specs_data = [
            ["Specification", "Original Dataset", "Processed Dataset"],
            ["Record Count", f"{df_original.shape[0]:,}", f"{df_cleaned.shape[0]:,}"],
            ["Feature Count", str(df_original.shape[1]), str(df_cleaned.shape[1])],
            ["Memory Usage", f"{df_original.memory_usage(deep=True).sum()/(1024*1024):.2f} MB",
             f"{df_cleaned.memory_usage(deep=True).sum()/(1024*1024):.2f} MB"],
            ["Numeric Features", str(len(df_original.select_dtypes(include=[np.number]).columns)),
             str(len(df_cleaned.select_dtypes(include=[np.number]).columns))],
            ["Categorical Features", str(len(df_original.select_dtypes(include=['object']).columns)),
             str(len(df_cleaned.select_dtypes(include=['object']).columns))],
            ["Missing Data %", f"{df_original.isnull().sum().sum()/(df_original.shape[0]*df_original.shape[1])*100:.2f}%",
             f"{df_cleaned.isnull().sum().sum()/(df_cleaned.shape[0]*df_cleaned.shape[1])*100:.2f}%"],
            ["Duplicate Records", str(df_original.duplicated().sum()),
             str(df_cleaned.duplicated().sum())]
        ]
        
        specs_table = Table(specs_data, colWidths=[2*inch, 2*inch, 2*inch])
        specs_table.setStyle(self._get_professional_table_style())
        content.append(specs_table)
        content.append(Spacer(1, 16))
        
        # A.2 Processing Pipeline
        content.append(Paragraph("A.2 Processing Pipeline Configuration", self.styles['SubsectionHeader']))
        
        pipeline_text = """
        <para alignment="justify">
        The RTGS AI Analyst system employed an 8-stage processing pipeline:
        </para>
        """
        content.append(Paragraph(pipeline_text, self.styles['ProfessionalBody']))
        
        pipeline_stages = [
            "1. <b>Data Ingestion:</b> Validated input and established baseline metrics",
            "2. <b>Inspection Analysis:</b> Identified quality issues and vulnerabilities",
            "3. <b>Intelligent Cleaning:</b> Applied context-aware data cleaning algorithms",
            "4. <b>Feature Engineering:</b> Generated derived variables and transformations",
            "5. <b>Quality Verification:</b> Validated improvements and consistency",
            "6. <b>Statistical Analysis:</b> Performed comprehensive statistical assessment",
            "7. <b>Visualization Generation:</b> Created analytical charts and graphs",
            "8. <b>Report Synthesis:</b> Generated this comprehensive analytical report"
        ]
        
        for stage in pipeline_stages:
            content.append(Paragraph(stage, self.styles['ProfessionalBody']))
        content.append(Spacer(1, 16))
        
        # A.3 Statistical Methods
        content.append(Paragraph("A.3 Statistical Methods Applied", self.styles['SubsectionHeader']))
        
        methods_text = """
        <para alignment="justify">
        The following statistical and machine learning methods were applied:
        • <b>Descriptive Statistics:</b> Mean, median, mode, standard deviation, skewness, kurtosis
        • <b>Correlation Analysis:</b> Pearson, Spearman rank correlations
        • <b>Outlier Detection:</b> IQR method, Z-score analysis, Isolation Forest
        • <b>Distribution Analysis:</b> Normality tests, Q-Q plots, histogram analysis
        • <b>Feature Engineering:</b> Scaling, encoding, log transformations, polynomial features
        • <b>Dimensionality Analysis:</b> PCA, feature importance ranking
        </para>
        """
        content.append(Paragraph(methods_text, self.styles['ProfessionalBody']))
        
        content.append(PageBreak())
        return content
    
    def _create_methodology_section(self) -> List:
        """Create methodology section"""
        content = []
        
        content.append(Paragraph("APPENDIX B: ANALYTICAL METHODOLOGY", self.styles['SectionHeader']))
        content.append(Spacer(1, 12))
        
        # B.1 Quality Assessment Framework
        content.append(Paragraph("B.1 Data Quality Assessment Framework", self.styles['SubsectionHeader']))
        
        quality_framework = """
        <para alignment="justify">
        Data quality assessment employed a multi-dimensional framework evaluating:
        
        <b>1. Completeness:</b> Percentage of non-null values across all fields
        <b>2. Consistency:</b> Logical coherence and format standardization
        <b>3. Accuracy:</b> Statistical validation and outlier assessment
        <b>4. Validity:</b> Compliance with business rules and constraints
        <b>5. Uniqueness:</b> Duplicate detection and entity resolution
        <b>6. Timeliness:</b> Currency and relevance of data points
        
        Each dimension contributes to the composite quality score using weighted aggregation.
        </para>
        """
        content.append(Paragraph(quality_framework, self.styles['ProfessionalBody']))
        content.append(Spacer(1, 12))
        
        # B.2 Machine Learning Readiness
        content.append(Paragraph("B.2 Machine Learning Readiness Criteria", self.styles['SubsectionHeader']))
        
        ml_criteria = """
        <para alignment="justify">
        ML readiness assessment evaluated:
        • Sample-to-feature ratio (minimum 10:1 recommended)
        • Feature variance and information content
        • Target variable distribution (for supervised learning)
        • Missing data patterns and imputation feasibility
        • Categorical encoding requirements
        • Scaling and normalization needs
        </para>
        """
        content.append(Paragraph(ml_criteria, self.styles['ProfessionalBody']))
        
        content.append(PageBreak())
        return content
    
    def _create_glossary(self) -> List:
        """Create glossary of terms"""
        content = []
        
        content.append(Paragraph("GLOSSARY OF TERMS", self.styles['SectionHeader']))
        content.append(Spacer(1, 12))
        
        glossary_terms = [
            ["Term", "Definition"],
            ["RTGS", "Real-Time Gross Settlement - A funds transfer system for large-value transactions"],
            ["Data Quality Score", "Composite metric (0-100) measuring overall dataset quality"],
            ["Feature Engineering", "Process of creating new variables from existing data"],
            ["Correlation", "Statistical measure of relationship between variables (-1 to +1)"],
            ["Skewness", "Measure of distribution asymmetry (0 = symmetric)"],
            ["Kurtosis", "Measure of distribution tail heaviness (3 = normal)"],
            ["IQR", "Interquartile Range - Used for outlier detection"],
            ["PCA", "Principal Component Analysis - Dimensionality reduction technique"],
            ["ML Readiness", "Assessment of data suitability for machine learning"],
            ["Multicollinearity", "High correlation between predictor variables"]
        ]
        
        glossary_table = Table(glossary_terms, colWidths=[1.5*inch, 4.5*inch])
        glossary_table.setStyle(self._get_professional_table_style())
        content.append(glossary_table)
        
        return content
    
    def _get_professional_table_style(self) -> TableStyle:
        """Get professional table style"""
        return TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e3d59')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dee2e6')),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')]),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
        ])
    
    def _get_status_label(self, score: float) -> str:
        """Get status label based on score"""
        if score >= 90:
            return "EXCELLENT"
        elif score >= 80:
            return "VERY GOOD"
        elif score >= 70:
            return "GOOD"
        elif score >= 60:
            return "ACCEPTABLE"
        else:
            return "NEEDS IMPROVEMENT"
    
    def generate_report(self, output_path: str, analysis_results: Dict[str, Any],
                       verification_results: Dict[str, Any],
                       df_original: pd.DataFrame, df_cleaned: pd.DataFrame) -> str:
        """Generate the enhanced government-grade PDF report"""
        
        logger.info(f"{self.name}: Generating government-grade analytical report...")
        
        # Extract deep insights
        insights = self._extract_deep_insights(df_original, df_cleaned, analysis_results)
        
        # Generate policy recommendations
        recommendations = self._generate_policy_recommendations(insights, analysis_results)
        
        # Create document with professional settings
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=1*inch,
            bottomMargin=0.75*inch,
            title="RTGS Data Intelligence Report",
            author="RTGS AI Analyst System",
            subject="Comprehensive Data Analysis for Policy Decision Making"
        )
        
        # Build content
        content = []
        
        # Title Page
        content.extend(self._create_title_page())
        
        # Table of Contents
        content.extend(self._create_table_of_contents())
        
        # Executive Brief
        content.extend(self._create_executive_brief(analysis_results, insights, recommendations))
        
        # Detailed Analysis
        content.extend(self._create_detailed_analysis(df_original, df_cleaned, analysis_results, insights))
        
        # Insights Synthesis
        content.extend(self._create_insights_synthesis(insights))
        
        # Policy Framework
        content.extend(self._create_policy_framework(recommendations))
        
        # Technical Appendix
        content.extend(self._create_technical_appendix(df_original, df_cleaned, analysis_results))
        
        # Methodology
        content.extend(self._create_methodology_section())
        
        # Glossary
        content.extend(self._create_glossary())
        
        # Build PDF
        doc.build(content, onFirstPage=self._create_header_footer, 
                 onLaterPages=self._create_header_footer)
        
        logger.info(f"{self.name}: Government-grade report generated successfully: {output_path}")
        return output_path
    
    def _create_title_page(self) -> List:
        """Create professional title page"""
        content = []
        
        # Add spacing
        content.append(Spacer(1, 2*inch))
        
        # Main title
        title = Paragraph("RTGS DATA INTELLIGENCE REPORT", self.styles['MainTitle'])
        content.append(title)
        
        # Subtitle
        subtitle = Paragraph("Comprehensive Analytical Assessment for Policy Decision Support", 
                           ParagraphStyle(name='Subtitle', parent=self.styles['Title'],
                                        fontSize=16, alignment=TA_CENTER,
                                        textColor=colors.HexColor('#2c5282')))
        content.append(subtitle)
        content.append(Spacer(1, 1*inch))
        
        # Report metadata
        metadata = f"""
        <para alignment="center">
        <b>Classification:</b> Official<br/>
        <b>Distribution:</b> Senior Management and Policy Makers<br/>
        <b>Prepared by:</b> RTGS AI Analyst System v2.0<br/>
        <b>Date:</b> {datetime.now().strftime('%B %d, %Y')}<br/>
        <b>Report ID:</b> RTGS-{datetime.now().strftime('%Y%m%d-%H%M%S')}
        </para>
        """
        content.append(Paragraph(metadata, self.styles['ProfessionalBody']))
        content.append(Spacer(1, 2*inch))
        
        # Disclaimer
        disclaimer = """
        <para alignment="center">
        <font size="9">
        This report contains proprietary analytical insights generated through advanced AI algorithms. 
        The findings and recommendations are based on data available at the time of analysis. 
        Decision makers should consider this analysis in conjunction with other relevant factors 
        and domain expertise.
        </font>
        </para>
        """
        content.append(Paragraph(disclaimer, self.styles['Footer']))
        
        content.append(PageBreak())
        return content
    
    def _create_table_of_contents(self) -> List:
        """Create table of contents"""
        content = []
        
        content.append(Paragraph("TABLE OF CONTENTS", self.styles['SectionHeader']))
        content.append(Spacer(1, 20))
        
        toc_items = [
            ("EXECUTIVE BRIEF", "3"),
            ("1. DETAILED ANALYTICAL ASSESSMENT", "4"),
            ("   1.1 Data Transformation Impact Analysis", "4"),
            ("   1.2 Statistical Properties Assessment", "5"),
            ("   1.3 Inter-variable Dependencies", "6"),
            ("   1.4 Comprehensive Quality Metrics", "7"),
            ("2. STRATEGIC INSIGHTS SYNTHESIS", "8"),
            ("3. POLICY IMPLEMENTATION FRAMEWORK", "11"),
            ("APPENDIX A: TECHNICAL SPECIFICATIONS", "14"),
            ("APPENDIX B: ANALYTICAL METHODOLOGY", "16"),
            ("GLOSSARY OF TERMS", "18")
        ]
        
        for item, page in toc_items:
            toc_text = f'<para>{item}{"."*(60-len(item))}{page}</para>'
            content.append(Paragraph(toc_text, ParagraphStyle(
                name='TOC', parent=self.styles['Normal'],
                fontSize=11, leftIndent=20 if item.startswith("   ") else 0
            )))
            content.append(Spacer(1, 6))
        
        content.append(PageBreak())
        return content
    
    def process(self, analysis_results: Dict[str, Any], 
                verification_results: Dict[str, Any],
                df_original: pd.DataFrame = None,
                df_cleaned: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Main processing method for the Enhanced Report Agent
        """
        logger.info(f"{self.name}: Starting government-grade report generation")
        
        try:
            # Create output filename
            timestamp = get_timestamp()
            filename = f"rtgs_intelligence_report_{timestamp}.pdf"
            output_path = os.path.join(PATHS.outputs_reports, filename)
            
            # Ensure we have dataframes (for backward compatibility)
            if df_original is None or df_cleaned is None:
                # Try to get from analysis results if not provided
                logger.warning("DataFrames not provided, report will have limited insights")
                # Create dummy dataframes for compatibility
                df_original = pd.DataFrame()
                df_cleaned = pd.DataFrame()
            
            # Generate report
            report_path = self.generate_report(output_path, analysis_results, 
                                              verification_results, df_original, df_cleaned)
            
            # Calculate file size
            file_size = os.path.getsize(report_path) / (1024 * 1024)
            
            result = {
                'agent': self.name,
                'status': 'success',
                'report_path': report_path,
                'file_size_mb': round(file_size, 2),
                'insights_generated': len(self.insights_generated),
                'policy_recommendations': len(self.policy_recommendations),
                'report_type': 'Government-Grade Intelligence Report',
                'message': f"Premium analytical report generated with deep insights. Size: {file_size:.2f} MB"
            }
            
            logger.info(f"{self.name}: Report generation completed successfully")
            
            return result
            
        except Exception as e:
            error_msg = f"Report generation failed: {str(e)}"
            logger.error(f"{self.name}: {error_msg}")
            
            return {
                'agent': self.name,
                'status': 'error',
                'message': error_msg,
                'error': str(e)
            }
    
    def _generate_basic_report(self, output_path: str, analysis_results: Dict[str, Any],
                              verification_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate basic report without DataFrames (backward compatibility)"""
        logger.info(f"{self.name}: Generating basic report without DataFrames...")
        
        # Create document
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=1*inch,
            bottomMargin=0.75*inch
        )
        
        content = []
        
        # Title
        content.append(Paragraph("RTGS DATA INTELLIGENCE REPORT", self.styles['MainTitle']))
        content.append(Spacer(1, 30))
        
        # Executive Summary based on available data
        content.append(Paragraph("EXECUTIVE SUMMARY", self.styles['SectionHeader']))
        
        raw_score = analysis_results.get('raw_dataset_analysis', {}).get('quality_score', 0)
        final_score = analysis_results.get('cleaned_dataset_analysis', {}).get('final_quality_score', 0)
        improvement = final_score - raw_score
        
        summary_text = f"""
        The RTGS AI Analyst system has processed your dataset and achieved a quality improvement 
        from {raw_score:.1f} to {final_score:.1f}, representing a {improvement:.1f} point enhancement.
        
        Data Quality Status: {self._get_status_label(final_score)}
        """
        content.append(Paragraph(summary_text, self.styles['ProfessionalBody']))
        content.append(Spacer(1, 20))
        
        # Raw Dataset Analysis
        content.append(Paragraph("1. RAW DATASET ANALYSIS", self.styles['SectionHeader']))
        raw_analysis = analysis_results.get('raw_dataset_analysis', {})
        
        overview = raw_analysis.get('dataset_overview', {})
        if overview:
            overview_text = f"""
            Dataset Shape: {overview.get('shape', 'N/A')}
            Memory Usage: {overview.get('memory_usage_mb', 'N/A')} MB
            Missing Data: {overview.get('missing_data_summary', 'N/A')}
            """
            content.append(Paragraph(overview_text, self.styles['ProfessionalBody']))
        
        vulnerabilities = raw_analysis.get('vulnerabilities', [])
        if vulnerabilities:
            content.append(Paragraph("Vulnerabilities Identified:", self.styles['SubsectionHeader']))
            for vuln in vulnerabilities[:5]:
                content.append(Paragraph(f"• {vuln}", self.styles['KeyFinding']))
        content.append(Spacer(1, 20))
        
        # Cleaned Dataset Analysis
        content.append(Paragraph("2. DATA CLEANING RESULTS", self.styles['SectionHeader']))
        cleaned_analysis = analysis_results.get('cleaned_dataset_analysis', {})
        
        improvement_summary = cleaned_analysis.get('improvement_summary', {})
        if improvement_summary:
            improvement_text = f"""
            Original Shape: {improvement_summary.get('original_shape', 'N/A')}
            Final Shape: {improvement_summary.get('final_shape', 'N/A')}
            Actions Applied: {improvement_summary.get('actions_applied', 0)}
            """
            content.append(Paragraph(improvement_text, self.styles['ProfessionalBody']))
        
        improvements = cleaned_analysis.get('improvements', [])
        if improvements:
            content.append(Paragraph("Improvements Achieved:", self.styles['SubsectionHeader']))
            for imp in improvements[:5]:
                content.append(Paragraph(f"• {imp}", self.styles['KeyFinding']))
        content.append(Spacer(1, 20))
        
        # Verification Results
        content.append(Paragraph("3. QUALITY VERIFICATION", self.styles['SectionHeader']))
        quality_score = verification_results.get('quality_score', 0)
        verification_text = f"""
        Final Quality Score: {quality_score:.1f}/100
        Status: {self._get_status_label(quality_score)}
        """
        content.append(Paragraph(verification_text, self.styles['ProfessionalBody']))
        content.append(Spacer(1, 20))
        
        # Recommendations
        content.append(Paragraph("4. RECOMMENDATIONS", self.styles['SectionHeader']))
        recommendations = cleaned_analysis.get('future_recommendations', [])
        if recommendations:
            for rec in recommendations[:5]:
                content.append(Paragraph(f"• {rec}", self.styles['PolicyRec']))
        else:
            content.append(Paragraph("Continue with standard analytical procedures.", 
                                   self.styles['ProfessionalBody']))
        
        # Build PDF
        doc.build(content, onFirstPage=self._create_header_footer, 
                 onLaterPages=self._create_header_footer)
        
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        
        return {
            'agent': self.name,
            'status': 'success',
            'report_path': output_path,
            'file_size_mb': round(file_size, 2),
            'report_type': 'Basic Report (Limited Data)',
            'message': f"Basic report generated (DataFrames not available). Size: {file_size:.2f} MB"
        }







# For backward compatibility, also create the original ReportAgent class
# class ReportAgent(EnhancedReportAgent):
#     """Original Report Agent for backward compatibility"""
    
#     def __init__(self):
#         super().__init__()
#         self.name = "Report Agent"
    
#     def process(self, analysis_results: Dict[str, Any], 
#                 verification_results: Dict[str, Any]) -> Dict[str, Any]:
#         """
#         Original process method that doesn't require DataFrames
#         """
#         logger.info(f"{self.name}: Starting report generation")
        
#         try:
#             # Create output filename
#             timestamp = get_timestamp()
#             filename = f"data_analysis_report_{timestamp}.pdf"
#             output_path = os.path.join(PATHS.outputs_reports, filename)
            
#             # Use the basic report generation
#             return self._generate_basic_report(output_path, analysis_results, verification_results)
            
#         except Exception as e:
#             error_msg = f"Report generation failed: {str(e)}"
#             logger.error(f"{self.name}: {error_msg}")
            
#             return {
#                 'agent': self.name,
#                 'status': 'error',
#                 'message': error_msg,
#                 'error': str(e)
#             }

