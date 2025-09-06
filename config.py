"""
Configuration file for RTGS AI Analyst
"""
import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv
load_dotenv()

@dataclass
class ModelConfig:
    """Model configuration for different AI providers"""
    
    # ChatGroq Configuration
    groq_api_key: Optional[str] = None
    groq_model: str = "llama-3.1-8b-instant"
    groq_temperature: float = 0.1
    
    # OpenAI Configuration (fallback)
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-3.5-turbo"
    openai_temperature: float = 0.1
    
    def __post_init__(self):
        """Load API keys from environment variables"""
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

@dataclass
class ProjectPaths:
    """Project directory paths"""
    root_dir: str = "."
    data_raw: str = "data/raw"
    data_processed: str = "data/processed"
    outputs_reports: str = "outputs/reports"
    outputs_charts: str = "outputs/charts"
    
    def create_directories(self):
        """Create project directories if they don't exist"""
        for path in [self.data_raw, self.data_processed, self.outputs_reports, self.outputs_charts]:
            os.makedirs(path, exist_ok=True)

# Global configuration instances
MODEL_CONFIG = ModelConfig()
PATHS = ProjectPaths()

# Analysis configuration
ANALYSIS_CONFIG = {
    "outlier_threshold": 3.0,  # Z-score threshold for outliers
    "correlation_threshold": 0.8,  # High correlation threshold
    "missing_threshold": 0.05,  # 5% missing data threshold for action
    "skewness_threshold": 2.0,  # Skewness threshold for normality
}

# Visualization configuration
VIZ_CONFIG = {
    "figure_size": (12, 8),
    "color_palette": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"],
    "font_family": "Arial",
    "title_size": 16,
    "label_size": 12,
}

# Report configuration
REPORT_CONFIG = {
    "font_family": "Helvetica",  # Use Helvetica as Cambria alternative
    "title_size": 16,
    "subtitle_size": 14,
    "body_size": 11,
    "margin": 1,  # inch
}