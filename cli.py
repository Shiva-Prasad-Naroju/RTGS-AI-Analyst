"""
RTGS AI Analyst - CLI Interface
"""

import click
import logging
import sys
from pathlib import Path
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

# Import agents
from agents import (
    DataIngestionAgent,
    InspectorAgent, 
    CleaningAgent,
    TransformingAgent,
    VerificationAgent,
    AnalysisAgent
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)

console = Console()


class RTGSAIAnalyst:
    """Main orchestrator for RTGS AI Analyst"""
    
    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize agents
        self.ingestion_agent = DataIngestionAgent()
        self.inspector_agent = InspectorAgent()
        self.cleaning_agent = CleaningAgent()
        self.transforming_agent = TransformingAgent()
        self.verification_agent = VerificationAgent()
        self.analysis_agent = AnalysisAgent(output_dir)
        
        self.logger = logging.getLogger(__name__)
    
    def run_full_pipeline(self, csv_source: str, is_url: bool = False) -> bool:
        """Run the complete RTGS AI Analyst pipeline"""
        
        try:
            console.print(Panel.fit("🤖 RTGS AI Analyst - Healthcare Data Pipeline", style="bold blue"))
            console.print(f"Processing: {csv_source}")
            console.print()
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                
                # Step 1: Data Ingestion
                task1 = progress.add_task("🔄 Loading CSV data...", total=None)
                raw_df, metadata = self.ingestion_agent.load_csv(csv_source, is_url)
                progress.update(task1, description="✅ Data loaded successfully")
                progress.stop_task(task1)
                
                # Step 2: Dataset Inspection
                task2 = progress.add_task("🔍 Inspecting dataset...", total=None)
                action_plan = self.inspector_agent.inspect_dataset(raw_df)
                progress.update(task2, description="✅ Inspection completed")
                progress.stop_task(task2)
                
                # Step 3: Data Cleaning
                task3 = progress.add_task("🧹 Cleaning data...", total=None)
                cleaned_df, cleaning_log = self.cleaning_agent.clean_dataset(raw_df, action_plan)
                progress.update(task3, description="✅ Data cleaned")
                progress.stop_task(task3)
                
                # Step 4: Data Transformation
                task4 = progress.add_task("⚡ Transforming data...", total=None)
                transformed_df, transformation_log = self.transforming_agent.transform_dataset(cleaned_df)
                progress.update(task4, description="✅ Data transformed")
                progress.stop_task(task4)
                
                # Step 5: Verification
                task5 = progress.add_task("🔍 Verifying data quality...", total=None)
                success, verification_results = self.verification_agent.verify_dataset(
                    raw_df, transformed_df, cleaning_log, transformation_log
                )
                progress.update(task5, description="✅ Verification completed")
                progress.stop_task(task5)
                
                # Step 6: Analysis and Insights
                task6 = progress.add_task("📊 Generating insights...", total=None)
                analysis_results = self.analysis_agent.generate_analysis(
                    transformed_df, cleaning_log, transformation_log, verification_results
                )
                progress.update(task6, description="✅ Analysis completed")
                progress.stop_task(task6)
            
            # Final status
            if success:
                console.print(Panel.fit("🎉 Pipeline completed successfully!", style="bold green"))
                console.print(f"📁 Results saved to: {self.output_dir}")
            else:
                console.print(Panel.fit("⚠️ Pipeline completed with warnings", style="bold yellow"))
                console.print("Check verification results for details.")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            console.print(Panel.fit(f"❌ Pipeline failed: {e}", style="bold red"))
            return False


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """RTGS AI Analyst - Automated Healthcare Data Analysis Pipeline"""
    pass


@cli.command()
@click.option('--csv-file', '-f', help='Path to CSV file')
@click.option('--url', '-u', help='URL to CSV file from Telangana Open Data portal')
@click.option('--output-dir', '-o', default='outputs', help='Output directory for results')
def all(csv_file, url, output_dir):
    """Run the complete analysis pipeline"""
    
    if not csv_file and not url:
        console.print("❌ Please provide either --csv-file or --url")
        sys.exit(1)
    
    if csv_file and url:
        console.print("❌ Please provide either --csv-file OR --url, not both")
        sys.exit(1)
    
    # Initialize analyst
    analyst = RTGSAIAnalyst(output_dir)
    
    # Run pipeline
    if csv_file:
        success = analyst.run_full_pipeline(csv_file, is_url=False)
    else:
        success = analyst.run_full_pipeline(url, is_url=True)
    
    sys.exit(0 if success else 1)


@cli.command()
@click.option('--csv-file', '-f', required=True, help='Path to CSV file')
def inspect(csv_file):
    """Inspect a CSV file without processing"""
    
    try:
        ingestion_agent = DataIngestionAgent()
        inspector_agent = InspectorAgent()
        
        console.print("🔍 Inspecting CSV file...")
        df, metadata = ingestion_agent.load_csv(csv_file)
        action_plan = inspector_agent.inspect_dataset(df)
        
        console.print(f"✅ Inspection completed for {metadata['rows']} rows, {metadata['columns']} columns")
        
    except Exception as e:
        console.print(f"❌ Inspection failed: {e}")
        sys.exit(1)


@cli.command()
@click.option('--csv-file', '-f', required=True, help='Path to CSV file')
@click.option('--output-dir', '-o', default='outputs', help='Output directory')
def clean(csv_file, output_dir):
    """Clean a CSV file and save the result"""
    
    try:
        # Initialize agents
        ingestion_agent = DataIngestionAgent()
        inspector_agent = InspectorAgent()
        cleaning_agent = CleaningAgent()
        
        console.print("🧹 Cleaning CSV file...")
        
        # Load and inspect
        df, metadata = ingestion_agent.load_csv(csv_file)
        action_plan = inspector_agent.inspect_dataset(df)
        
        # Clean
        cleaned_df, cleaning_log = cleaning_agent.clean_dataset(df, action_plan)
        
        # Save result
        output_path = Path(output_dir) / "data" / "processed" / "cleaned_dataset.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cleaned_df.to_csv(output_path, index=False)
        
        console.print(f"✅ Cleaned dataset saved to: {output_path}")
        console.print(f"Applied {len(cleaning_log)} cleaning operations")
        
    except Exception as e:
        console.print(f"❌ Cleaning failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    cli()