"""
Configuration settings for RTGS AI Analyst
"""

import os
from pathlib import Path

# Directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = BASE_DIR / "outputs"
CHARTS_DIR = OUTPUTS_DIR / "charts"

# Create directories
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, OUTPUTS_DIR, CHARTS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Data quality thresholds
QUALITY_THRESHOLDS = {
    'null_percentage_warning': 10,
    'null_percentage_error': 25,
    'duplicate_percentage_warning': 5,
    'outlier_threshold': 1.5,  # IQR multiplier
    'correlation_threshold': 0.7,
    'max_categories_for_encoding': 20,
    'min_samples_for_normalization': 100
}

# Visualization settings
VIZ_SETTINGS = {
    'figure_size': (12, 8),
    'dpi': 300,
    'color_palette': 'viridis',
    'max_categories_in_plot': 10,
    'chart_style': 'seaborn-v0_8'
}

# Healthcare domain settings
HEALTHCARE_SETTINGS = {
    'bed_capacity_max': 10000,
    'establishment_year_min': 1800,
    'establishment_year_max': 2025,
    'geographic_inequality_threshold': 3,  # ratio threshold
    'capacity_variation_threshold': 1.0  # std/mean ratio
}

# Logging settings
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file_logging': True,
    'log_file': OUTPUTS_DIR / 'rtgs_analyst.log'
}