"""
RTGS AI Analyst - Main Supervisor Agent
Orchestrates the multi-agent data analysis pipeline
"""
import pandas as pd
import logging
from typing import Dict, Any, Optional
import os
from datetime import datetime

# Import all agents
from agents import (
    IngestionAgent, InspectionAgent, CleaningAgent, TransformationAgent,
    VerificationAgent, AnalysisAgent, VisualizationAgent, ReportAgent
)

from config import PATHS, MODEL_CONFIG
from utils import setup_logging, save_dataframe, get_timestamp

# Setup logging
logger = logging.getLogger(__name__)

class SupervisorAgent:
    """
    Supervisor Agent that orchestrates the entire data analysis pipeline
    """
    
    def __init__(self, interactive: bool = True):
        self.name = "Supervisor Agent"
        self.interactive = interactive
        self.execution_log = []
        self.results = {}
        
        # Initialize all agents
        self.ingestion_agent = IngestionAgent()
        self.inspection_agent = InspectionAgent()
        self.cleaning_agent = CleaningAgent(interactive=interactive)
        self.transformation_agent = TransformationAgent()
        self.verification_agent = VerificationAgent()
        self.analysis_agent = AnalysisAgent()
        self.visualization_agent = VisualizationAgent()
        self.report_agent = ReportAgent()
        
        # Create necessary directories
        PATHS.create_directories()
        
        logger.info(f"{self.name}: Initialized with {'interactive' if interactive else 'automated'} mode")
    
    def _log_step(self, step: str, agent: str, status: str, message: str = "", duration: float = 0):
        """Log execution step"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'step': step,
            'agent': agent,
            'status': status,
            'message': message,
            'duration_seconds': duration
        }
        self.execution_log.append(log_entry)
        
        # Also log to console
        logger.info(f"Step {step}: {agent} - {status}")
        if message:
            logger.info(f"  Message: {message}")
    
    def _display_progress(self, current_step: int, total_steps: int, step_name: str):
        """Display progress to user"""
        progress = (current_step / total_steps) * 100
        print(f"\n{'='*60}")
        print(f"Progress: {progress:.1f}% ({current_step}/{total_steps})")
        print(f"Current Step: {step_name}")
        print(f"{'='*60}")
    
    def run_pipeline(self, file_path: str, apply_transformations: bool = True,
                    create_visualizations: bool = True, generate_report: bool = True) -> Dict[str, Any]:
        """
        Run the complete data analysis pipeline
        
        Args:
            file_path: Path to the dataset file
            apply_transformations: Whether to apply data transformations
            create_visualizations: Whether to create visualization PDF
            generate_report: Whether to generate analysis report
            
        Returns:
            Complete pipeline results
        """
        
        start_time = datetime.now()
        logger.info(f"{self.name}: Starting complete data analysis pipeline")
        logger.info(f"Input file: {file_path}")
        
        total_steps = 8 if generate_report else 7 if create_visualizations else 6
        current_step = 0
        
        try:
            # Step 1: Data Ingestion
            current_step += 1
            self._display_progress(current_step, total_steps, "Data Ingestion")
            
            step_start = datetime.now()
            ingestion_result = self.ingestion_agent.process(file_path)
            step_duration = (datetime.now() - step_start).total_seconds()
            
            if ingestion_result['status'] != 'success':
                self._log_step(current_step, "Ingestion Agent", "FAILED", 
                             ingestion_result['message'], step_duration)
                return self._create_failure_result("Data ingestion failed", ingestion_result)
            
            df_original = ingestion_result['data']
            self.results['ingestion'] = ingestion_result
            self._log_step(current_step, "Ingestion Agent", "SUCCESS", 
                         ingestion_result['message'], step_duration)
            
            # Step 2: Data Inspection
            current_step += 1
            self._display_progress(current_step, total_steps, "Data Quality Inspection")
            
            step_start = datetime.now()
            inspection_result = self.inspection_agent.process(df_original)
            step_duration = (datetime.now() - step_start).total_seconds()
            
            if inspection_result['status'] == 'error':
                self._log_step(current_step, "Inspection Agent", "FAILED", 
                             inspection_result['message'], step_duration)
                return self._create_failure_result("Data inspection failed", inspection_result)
            
            self.results['inspection'] = inspection_result
            self._log_step(current_step, "Inspection Agent", "SUCCESS", 
                         f"Found {inspection_result['total_issues']} issues", step_duration)
            
            # Step 3: Data Cleaning
            current_step += 1
            self._display_progress(current_step, total_steps, "Data Cleaning")
            
            step_start = datetime.now()
            cleaning_result = self.cleaning_agent.process(df_original, inspection_result)
            step_duration = (datetime.now() - step_start).total_seconds()
            
            if cleaning_result['status'] == 'error':
                self._log_step(current_step, "Cleaning Agent", "FAILED", 
                             cleaning_result['message'], step_duration)
                return self._create_failure_result("Data cleaning failed", cleaning_result)
            
            df_cleaned = cleaning_result['data']
            self.results['cleaning'] = cleaning_result
            self._log_step(current_step, "Cleaning Agent", "SUCCESS", 
                         f"Applied {len(cleaning_result['cleaning_actions'])} actions", step_duration)
            
            # Step 4: Data Transformation (optional)
            current_step += 1
            if apply_transformations:
                self._display_progress(current_step, total_steps, "Data Transformation")
                
                step_start = datetime.now()
                transformation_result = self.transformation_agent.process(df_cleaned)
                step_duration = (datetime.now() - step_start).total_seconds()
                
                if transformation_result['status'] == 'error':
                    self._log_step(current_step, "Transformation Agent", "FAILED", 
                                 transformation_result['message'], step_duration)
                    return self._create_failure_result("Data transformation failed", transformation_result)
                
                df_final = transformation_result['data']
                self.results['transformation'] = transformation_result
                self._log_step(current_step, "Transformation Agent", "SUCCESS", 
                             f"Applied {len(transformation_result['transformations'])} transformations", step_duration)
            else:
                df_final = df_cleaned
                self.results['transformation'] = {'status': 'skipped', 'transformations': []}
                self._log_step(current_step, "Transformation Agent", "SKIPPED", 
                             "Transformations disabled", 0)
            
            # Step 5: Data Verification
            current_step += 1
            self._display_progress(current_step, total_steps, "Data Quality Verification")
            
            step_start = datetime.now()
            verification_result = self.verification_agent.process(df_final)
            step_duration = (datetime.now() - step_start).total_seconds()
            
            if verification_result['status'] == 'error':
                self._log_step(current_step, "Verification Agent", "FAILED", 
                             verification_result['message'], step_duration)
                return self._create_failure_result("Data verification failed", verification_result)
            
            self.results['verification'] = verification_result
            self._log_step(current_step, "Verification Agent", "SUCCESS", 
                         f"Quality score: {verification_result['quality_score']:.1f}/100", step_duration)
            
            # Step 6: Analysis
            current_step += 1
            self._display_progress(current_step, total_steps, "AI-Powered Analysis")
            
            step_start = datetime.now()
            analysis_result = self.analysis_agent.process(
                df_original, df_final, inspection_result, 
                cleaning_result, self.results['transformation'], verification_result
            )
            step_duration = (datetime.now() - step_start).total_seconds()
            
            if analysis_result['status'] == 'error':
                self._log_step(current_step, "Analysis Agent", "FAILED", 
                             analysis_result['message'], step_duration)
                return self._create_failure_result("Analysis failed", analysis_result)
            
            self.results['analysis'] = analysis_result
            self._log_step(current_step, "Analysis Agent", "SUCCESS", 
                         analysis_result['message'], step_duration)
            
            # Step 7: Visualization (optional)
            if create_visualizations:
                current_step += 1
                self._display_progress(current_step, total_steps, "Creating Visualizations")
                
                step_start = datetime.now()
                visualization_result = self.visualization_agent.process(
                    df_original, df_final, analysis_result
                )
                step_duration = (datetime.now() - step_start).total_seconds()
                
                if visualization_result['status'] == 'error':
                    self._log_step(current_step, "Visualization Agent", "FAILED", 
                                 visualization_result['message'], step_duration)
                    logger.warning("Visualization failed, continuing without charts")
                    self.results['visualization'] = visualization_result
                else:
                    self.results['visualization'] = visualization_result
                    self._log_step(current_step, "Visualization Agent", "SUCCESS", 
                                 visualization_result['message'], step_duration)
            
            # Step 8: Report Generation (optional)
            if generate_report:
                current_step += 1
                self._display_progress(current_step, total_steps, "Generating Report")
                
                step_start = datetime.now()
                report_result = self.report_agent.process(analysis_result, verification_result)
                step_duration = (datetime.now() - step_start).total_seconds()
                
                if report_result['status'] == 'error':
                    self._log_step(current_step, "Report Agent", "FAILED", 
                                 report_result['message'], step_duration)
                    logger.warning("Report generation failed, continuing without report")
                    self.results['report'] = report_result
                else:
                    self.results['report'] = report_result
                    self._log_step(current_step, "Report Agent", "SUCCESS", 
                                 report_result['message'], step_duration)
            
            # Save final cleaned dataset
            self._save_final_dataset(df_final, df_original)
            
            # Calculate total execution time
            total_duration = (datetime.now() - start_time).total_seconds()
            
            # Create final results summary
            final_result = self._create_success_result(df_original, df_final, total_duration)
            
            logger.info(f"{self.name}: Pipeline completed successfully in {total_duration:.1f} seconds")
            
            return final_result
            
        except Exception as e:
            error_msg = f"Pipeline execution failed: {str(e)}"
            logger.error(f"{self.name}: {error_msg}")
            self._log_step(current_step, "Supervisor Agent", "CRITICAL_FAILURE", error_msg, 0)
            
            return self._create_failure_result("Critical pipeline failure", {'error': str(e)})
    
    def _save_final_dataset(self, df_final: pd.DataFrame, df_original: pd.DataFrame):
        """Save the final cleaned dataset"""
        try:
            timestamp = get_timestamp()
            
            # Save cleaned dataset
            cleaned_filename = f"cleaned_dataset_{timestamp}.csv"
            cleaned_path = os.path.join(PATHS.data_processed, cleaned_filename)
            save_dataframe(df_final, cleaned_path)
            
            # Save original dataset copy (for reference)
            original_filename = f"original_dataset_{timestamp}.csv"
            original_path = os.path.join(PATHS.data_processed, original_filename)
            save_dataframe(df_original, original_path)
            
            self.results['file_outputs'] = {
                'cleaned_dataset': cleaned_path,
                'original_dataset_copy': original_path
            }
            
            logger.info(f"Final datasets saved:")
            logger.info(f"  - Cleaned: {cleaned_path}")
            logger.info(f"  - Original copy: {original_path}")
            
        except Exception as e:
            logger.error(f"Failed to save final datasets: {str(e)}")
    
    def _create_success_result(self, df_original: pd.DataFrame, df_final: pd.DataFrame, 
                              total_duration: float) -> Dict[str, Any]:
        """Create success result summary"""
        
        # Calculate key metrics
        quality_improvement = 0
        if 'analysis' in self.results and 'verification' in self.results:
            raw_score = self.results['analysis'].get('raw_dataset_analysis', {}).get('quality_score', 0)
            final_score = self.results['verification'].get('quality_score', 0)
            quality_improvement = final_score - raw_score
        
        # Count total actions
        total_actions = 0
        if 'cleaning' in self.results:
            total_actions += len(self.results['cleaning'].get('cleaning_actions', []))
        if 'transformation' in self.results:
            total_actions += len(self.results['transformation'].get('transformations', []))
        
        # Prepare file outputs
        file_outputs = self.results.get('file_outputs', {})
        if 'visualization' in self.results and self.results['visualization'].get('status') == 'success':
            file_outputs['visualization_pdf'] = self.results['visualization'].get('pdf_path')
        if 'report' in self.results and self.results['report'].get('status') == 'success':
            file_outputs['analysis_report'] = self.results['report'].get('report_path')
        
        return {
            'supervisor': self.name,
            'status': 'SUCCESS',
            'execution_summary': {
                'total_duration_seconds': total_duration,
                'total_duration_minutes': total_duration / 60,
                'agents_executed': len([r for r in self.results.values() if r.get('status') in ['success', 'warning']]),
                'total_actions_applied': total_actions,
                'quality_improvement': quality_improvement
            },
            'dataset_summary': {
                'original_shape': df_original.shape,
                'final_shape': df_final.shape,
                'rows_change': df_final.shape[0] - df_original.shape[0],
                'columns_change': df_final.shape[1] - df_original.shape[1],
                'memory_reduction_mb': (df_original.memory_usage(deep=True).sum() - 
                                      df_final.memory_usage(deep=True).sum()) / (1024 * 1024)
            },
            'agent_results': self.results,
            'file_outputs': file_outputs,
            'execution_log': self.execution_log,
            'message': f"Pipeline completed successfully! Quality improved by {quality_improvement:+.1f} points in {total_duration:.1f} seconds."
        }
    
    def _create_failure_result(self, reason: str, failed_result: Dict[str, Any]) -> Dict[str, Any]:
        """Create failure result summary"""
        return {
            'supervisor': self.name,
            'status': 'FAILED',
            'reason': reason,
            'failed_at': failed_result.get('agent', 'Unknown'),
            'error_details': failed_result.get('error', 'No error details available'),
            'partial_results': self.results,
            'execution_log': self.execution_log,
            'message': f"Pipeline failed: {reason}"
        }
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get a summary of the execution"""
        if not self.execution_log:
            return {'message': 'No execution data available'}
        
        successful_steps = [log for log in self.execution_log if log['status'] == 'SUCCESS']
        failed_steps = [log for log in self.execution_log if log['status'] in ['FAILED', 'CRITICAL_FAILURE']]
        total_duration = sum(log['duration_seconds'] for log in self.execution_log)
        
        return {
            'total_steps': len(self.execution_log),
            'successful_steps': len(successful_steps),
            'failed_steps': len(failed_steps),
            'total_duration_seconds': total_duration,
            'agents_used': list(set(log['agent'] for log in self.execution_log)),
            'execution_log': self.execution_log
        }
    
    def display_final_summary(self, result: Dict[str, Any]):
        """Display a user-friendly final summary"""
        print("\n" + "="*80)
        print("🎯 RTGS AI ANALYST - EXECUTION SUMMARY")
        print("="*80)
        
        if result['status'] == 'SUCCESS':
            print("✅ STATUS: PIPELINE COMPLETED SUCCESSFULLY")
            
            exec_summary = result['execution_summary']
            dataset_summary = result['dataset_summary']
            
            print(f"\n📊 PROCESSING RESULTS:")
            print(f"   • Duration: {exec_summary['total_duration_minutes']:.1f} minutes")
            print(f"   • Agents Executed: {exec_summary['agents_executed']}")
            print(f"   • Actions Applied: {exec_summary['total_actions_applied']}")
            print(f"   • Quality Improvement: {exec_summary['quality_improvement']:+.1f} points")
            
            print(f"\n📈 DATASET TRANSFORMATION:")
            print(f"   • Original Size: {dataset_summary['original_shape'][0]:,} rows × {dataset_summary['original_shape'][1]} columns")
            print(f"   • Final Size: {dataset_summary['final_shape'][0]:,} rows × {dataset_summary['final_shape'][1]} columns")
            print(f"   • Rows Changed: {dataset_summary['rows_change']:+,}")
            print(f"   • Columns Changed: {dataset_summary['columns_change']:+,}")
            print(f"   • Memory Optimized: {dataset_summary['memory_reduction_mb']:+.2f} MB")
            
            # Display file outputs
            file_outputs = result['file_outputs']
            if file_outputs:
                print(f"\n📁 OUTPUT FILES CREATED:")
                for file_type, file_path in file_outputs.items():
                    if file_path:
                        print(f"   • {file_type.replace('_', ' ').title()}: {file_path}")
            
            # Display quality score if available
            if 'verification' in result['agent_results']:
                quality_score = result['agent_results']['verification'].get('quality_score', 0)
                if quality_score >= 90:
                    status_emoji = "🟢"
                    status_text = "EXCELLENT"
                elif quality_score >= 75:
                    status_emoji = "🟡"
                    status_text = "GOOD"
                elif quality_score >= 60:
                    status_emoji = "🟠"
                    status_text = "ACCEPTABLE"
                else:
                    status_emoji = "🔴"
                    status_text = "NEEDS IMPROVEMENT"
                
                print(f"\n🏆 FINAL DATA QUALITY:")
                print(f"   • Score: {quality_score:.1f}/100 {status_emoji}")
                print(f"   • Status: {status_text}")
        
        else:
            print("❌ STATUS: PIPELINE FAILED")
            print(f"\n💥 FAILURE DETAILS:")
            print(f"   • Reason: {result['reason']}")
            print(f"   • Failed At: {result['failed_at']}")
            print(f"   • Error: {result.get('error_details', 'No details available')}")
        
        print("\n" + "="*80)
        print("🤖 Thank you for using RTGS AI Analyst!")
        print("="*80)


def main():
    """Main entry point for the RTGS AI Analyst"""
    
    # Setup logging
    setup_logging()
    
    print("🚀 RTGS AI Analyst - Multi-Agent Data Processing System")
    print("="*60)
    
    # Example usage with the demo dataset
    
    # demo_file = "data/raw/Hospitals.csv"
    # demo_file = "data/raw/agricultural_2019_5.csv"
    demo_file = "data/raw/consumption_detail_water_works.csv"
    
    if not os.path.exists(demo_file):
        print(f"❌ Demo file not found: {demo_file}")
        print("Please ensure the Hospitals.csv file is in the data/raw/ directory")
        return
    
    # Initialize supervisor
    supervisor = SupervisorAgent(interactive=True)
    
    print(f"\n📂 Processing dataset: {demo_file}")
    print("🔄 Starting multi-agent analysis pipeline...")
    
    # Run the complete pipeline
    result = supervisor.run_pipeline(
        file_path=demo_file,
        apply_transformations=True,
        create_visualizations=True,
        generate_report=True
    )
    
    # Display final summary
    supervisor.display_final_summary(result)
    
    # Display execution log summary
    exec_summary = supervisor.get_execution_summary()
    if exec_summary.get('failed_steps', 0) > 0:
        print(f"\n⚠️  Warning: {exec_summary['failed_steps']} steps had issues. Check logs for details.")
    
    return result


if __name__ == "__main__":
    result = main()