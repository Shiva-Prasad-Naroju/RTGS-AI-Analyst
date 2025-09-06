#!/usr/bin/env python3
"""
RTGS AI Analyst - Command Line Interface
Usage: python cli.py --file data/raw/Hospitals.csv [options]
"""
import click
import os
import sys
from pathlib import Path
from config import MODEL_CONFIG

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from main import SupervisorAgent
from utils import setup_logging
from config import PATHS

@click.command()
@click.option('--file', '-f', 'file_path', required=True, 
              help='Path to the dataset file (CSV or Excel)')
@click.option('--interactive/--no-interactive', default=True,
              help='Enable/disable interactive mode for cleaning confirmations')
@click.option('--transformations/--no-transformations', default=True,
              help='Enable/disable data transformations')
@click.option('--visualizations/--no-visualizations', default=True,
              help='Enable/disable visualization generation')
@click.option('--report/--no-report', default=True,
              help='Enable/disable report generation')
@click.option('--output-dir', '-o', default=None,
              help='Custom output directory (optional)')
@click.option('--log-level', default='INFO',
              type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']),
              help='Logging level')
@click.option('--verbose', '-v', is_flag=True,
              help='Enable verbose output')
def analyze_dataset(file_path, interactive, transformations, visualizations, 
                   report, output_dir, log_level, verbose):
    """
    RTGS AI Analyst - Automated Data Quality Analysis and Cleaning
    
    This tool uses a multi-agent system to automatically analyze, clean, and transform
    your dataset while providing comprehensive reports and visualizations.
    
    Example usage:
    
        python cli.py --file data/raw/Hospitals.csv
        
        python cli.py --file mydata.xlsx --no-interactive --output-dir results/
        
        python cli.py --file dataset.csv --no-transformations --verbose
    """
    
    # Setup logging
    setup_logging()
    
    # Banner
    click.echo("🚀 RTGS AI Analyst - Multi-Agent Data Processing System")
    click.echo("=" * 65)
    
    # Validate input file
    if not os.path.exists(file_path):
        click.echo(f"❌ Error: File not found: {file_path}", err=True)
        return 1
    
    # Check file extension
    file_ext = Path(file_path).suffix.lower()
    if file_ext not in ['.csv', '.xlsx', '.xls']:
        click.echo(f"❌ Error: Unsupported file format: {file_ext}", err=True)
        click.echo("Supported formats: .csv, .xlsx, .xls")
        return 1
    
    # Setup custom output directory if provided
    if output_dir:
        PATHS.outputs_reports = os.path.join(output_dir, "reports")
        PATHS.outputs_charts = os.path.join(output_dir, "charts")
        PATHS.data_processed = os.path.join(output_dir, "processed")
        PATHS.create_directories()
        click.echo(f"📁 Output directory: {output_dir}")
    
    # Display configuration
    click.echo(f"\n📂 Input file: {file_path}")
    click.echo(f"🔧 Configuration:")
    click.echo(f"   • Interactive mode: {'✅' if interactive else '❌'}")
    click.echo(f"   • Apply transformations: {'✅' if transformations else '❌'}")
    click.echo(f"   • Generate visualizations: {'✅' if visualizations else '❌'}")
    click.echo(f"   • Generate report: {'✅' if report else '❌'}")
    click.echo(f"   • Log level: {log_level}")
    
    if not interactive:
        click.echo("\n⚠️  Running in non-interactive mode - all cleaning actions will be auto-approved")
    
    # Confirm execution
    if interactive:
        if not click.confirm("\n🤖 Ready to start analysis?"):
            click.echo("Analysis cancelled.")
            return 0
    
    try:
        # Initialize supervisor
        click.echo(f"\n🔄 Initializing RTGS AI Analyst...")
        supervisor = SupervisorAgent(interactive=interactive)
        
        # Run the pipeline
        click.echo(f"🚀 Starting multi-agent analysis pipeline...")
        
        with click.progressbar(length=100, label='Processing') as bar:
            # We'll update this manually since we can't easily integrate with the supervisor's progress
            result = supervisor.run_pipeline(
                file_path=file_path,
                apply_transformations=transformations,
                create_visualizations=visualizations,
                generate_report=report
            )
            bar.update(100)
        
        # Display results
        click.echo(f"\n✅ Analysis completed!")
        
        if result['status'] == 'SUCCESS':
            # Success summary
            exec_summary = result['execution_summary']
            dataset_summary = result['dataset_summary']
            file_outputs = result['file_outputs']
            
            click.echo(f"\n📊 RESULTS SUMMARY:")
            click.echo(f"   • Processing time: {exec_summary['total_duration_minutes']:.1f} minutes")
            click.echo(f"   • Quality improvement: {exec_summary['quality_improvement']:+.1f} points")
            click.echo(f"   • Actions applied: {exec_summary['total_actions_applied']}")
            
            click.echo(f"\n📈 DATASET CHANGES:")
            click.echo(f"   • Shape: {dataset_summary['original_shape']} → {dataset_summary['final_shape']}")
            click.echo(f"   • Rows: {dataset_summary['rows_change']:+,}")
            click.echo(f"   • Columns: {dataset_summary['columns_change']:+,}")
            
            # Show output files
            if file_outputs:
                click.echo(f"\n📁 OUTPUT FILES:")
                for file_type, file_path in file_outputs.items():
                    if file_path and os.path.exists(file_path):
                        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                        click.echo(f"   • {file_type.replace('_', ' ').title()}: {file_path} ({file_size:.1f} MB)")
            
            # Quality score
            if 'verification' in result['agent_results']:
                quality_score = result['agent_results']['verification'].get('quality_score', 0)
                if quality_score >= 90:
                    status = "🟢 EXCELLENT"
                elif quality_score >= 75:
                    status = "🟡 GOOD"
                elif quality_score >= 60:
                    status = "🟠 ACCEPTABLE"
                else:
                    status = "🔴 NEEDS IMPROVEMENT"
                
                click.echo(f"\n🏆 FINAL QUALITY: {quality_score:.1f}/100 {status}")
            
            # Show next steps
            click.echo(f"\n🎯 NEXT STEPS:")
            click.echo(f"   • Review the generated report for detailed insights")
            click.echo(f"   • Check visualization charts for data quality comparison")
            click.echo(f"   • Use the cleaned dataset for your analysis")
            
            if verbose:
                # Show detailed execution log
                click.echo(f"\n📋 EXECUTION LOG:")
                for log_entry in result['execution_log']:
                    status_icon = "✅" if log_entry['status'] == 'SUCCESS' else "❌" if log_entry['status'] == 'FAILED' else "⚠️"
                    click.echo(f"   {status_icon} {log_entry['agent']}: {log_entry['message']} ({log_entry['duration_seconds']:.1f}s)")
            
            return 0
            
        else:
            # Failure summary
            click.echo(f"\n❌ ANALYSIS FAILED:")
            click.echo(f"   • Reason: {result['reason']}")
            click.echo(f"   • Failed at: {result['failed_at']}")
            click.echo(f"   • Error: {result.get('error_details', 'No details available')}")
            
            # Show partial results if any
            if result.get('partial_results'):
                click.echo(f"\n📋 PARTIAL RESULTS AVAILABLE:")
                for agent_name, agent_result in result['partial_results'].items():
                    status = agent_result.get('status', 'unknown')
                    status_icon = "✅" if status == 'success' else "❌" if status == 'error' else "⚠️"
                    click.echo(f"   {status_icon} {agent_name.title()}: {status}")
            
            return 1
    
    except KeyboardInterrupt:
        click.echo(f"\n⚠️  Analysis interrupted by user")
        return 1
    
    except Exception as e:
        click.echo(f"\n💥 Unexpected error: {str(e)}", err=True)
        if verbose:
            import traceback
            click.echo(f"\n🔍 Full traceback:")
            click.echo(traceback.format_exc())
        return 1


@click.command()
@click.option('--check-deps', is_flag=True, help='Check if all dependencies are installed')
@click.option('--setup-env', is_flag=True, help='Setup environment and create directories')
@click.option('--show-config', is_flag=True, help='Show current configuration')
def setup(check_deps, setup_env, show_config):
    """Setup and configuration commands for RTGS AI Analyst"""
    
    if check_deps:
        click.echo("🔍 Checking dependencies...")
        
        required_packages = [
            'pandas', 'numpy', 'matplotlib', 'seaborn', 'scipy', 
            'scikit-learn', 'reportlab', 'click', 'rich'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
                click.echo(f"   ✅ {package}")
            except ImportError:
                click.echo(f"   ❌ {package}")
                missing_packages.append(package)
        
        if missing_packages:
            click.echo(f"\n📦 Missing packages: {', '.join(missing_packages)}")
            click.echo(f"Install with: pip install {' '.join(missing_packages)}")
        else:
            click.echo(f"\n✅ All required dependencies are installed!")
    
    if setup_env:
        click.echo("🏗️  Setting up environment...")
        PATHS.create_directories()
        
        dirs_created = [
            PATHS.data_raw,
            PATHS.data_processed, 
            PATHS.outputs_reports,
            PATHS.outputs_charts
        ]
        
        for directory in dirs_created:
            if os.path.exists(directory):
                click.echo(f"   ✅ {directory}")
            else:
                click.echo(f"   ❌ Failed to create {directory}")
        
        click.echo(f"\n✅ Environment setup complete!")
    
    if show_config:
        click.echo("⚙️  Current Configuration:")
        click.echo(f"   • Data raw: {PATHS.data_raw}")
        click.echo(f"   • Data processed: {PATHS.data_processed}")
        click.echo(f"   • Reports output: {PATHS.outputs_reports}")
        click.echo(f"   • Charts output: {PATHS.outputs_charts}")
        
        # Check API keys
        if hasattr(MODEL_CONFIG, 'groq_api_key') and MODEL_CONFIG.groq_api_key:
            click.echo(f"   • Groq API: ✅ Configured")
        else:
            click.echo(f"   • Groq API: ❌ Not configured (set GROQ_API_KEY env var)")
        
        if hasattr(MODEL_CONFIG, 'openai_api_key') and MODEL_CONFIG.openai_api_key:
            click.echo(f"   • OpenAI API: ✅ Configured")
        else:
            click.echo(f"   • OpenAI API: ❌ Not configured (set OPENAI_API_KEY env var)")


@click.group()
def cli():
    """RTGS AI Analyst - Multi-Agent Data Processing System"""
    pass


# Add commands to the CLI group
cli.add_command(analyze_dataset, name='analyze')
cli.add_command(setup)


if __name__ == '__main__':
    cli()