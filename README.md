# RTGS-AI-Analyst for hospital dataset:

🤖 **AI-Powered Healthcare Data Analysis Pipeline**

An intelligent, automated data processing system specifically designed for healthcare datasets from government open data portals like Telangana Open Data.

## 🌟 Features

### 🔧 **6-Agent Architecture**
- **Data Ingestion Agent**: Smart CSV loading with encoding detection
- **Inspector Agent**: Comprehensive data quality assessment  
- **Cleaning Agent**: Automated data cleaning and validation
- **Transforming Agent**: Feature engineering and data transformation
- **Verification Agent**: Quality assurance and validation
- **Analysis Agent**: AI-driven insights and policy recommendations

### 📊 **Advanced Analytics**
- Automated statistical analysis and reporting
- Healthcare-specific business logic validation
- Geographic equity analysis for policy planning
- Interactive visualizations and charts
- Policy recommendations based on data patterns

### 💻 **Developer-Friendly**
- CLI-first design with rich terminal interface
- Modular architecture for easy extension
- Comprehensive logging and audit trails
- Docker support for containerized deployment
- CI/CD pipeline with automated testing

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Shiva-Prasad-Naroju/RTGS-AI-Analyst.git
cd RTGS-AI-Analyst

# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .
```

### Basic Usage

```bash
# Analyze a local CSV file
python cli.py all --csv-file data/hospitals.csv

# Analyze from Telangana Open Data URL
python cli.py all --url "https://data.telangana.gov.in/dataset/hospital-data.csv"

# Quick inspection without full processing
python cli.py inspect --csv-file data/hospitals.csv

# Clean data only
python cli.py clean --csv-file data/hospitals.csv --output-dir cleaned/
```

## 📁 Project Structure

```
rtgs_ai_analyst/
├── agents/                 # AI Agent modules
│   ├── ingestion_agent.py     # Data loading & encoding
│   ├── inspector_agent.py     # Quality assessment
│   ├── cleaning_agent.py      # Data cleaning
│   ├── transforming_agent.py  # Feature engineering
│   ├── verification_agent.py  # Quality validation
│   └── analysis_agent.py      # Insights generation
├── data/
│   ├── raw/               # Original datasets
│   └── processed/         # Cleaned datasets
├── outputs/
│   ├── charts/           # Generated visualizations
│   ├── insights.md       # Analysis report
│   └── run_log.md        # Processing logs
├── cli.py                # Main CLI interface
├── config.py            # Configuration settings
├── utils.py             # Utility functions
├── example_usage.py     # Demo script
├── run_tests.py         # Testing suite
├── requirements.txt     # Dependencies
├── Dockerfile          # Container definition
├── docker-compose.yml  # Multi-service setup
└── README.md           # Documentation
```

## 🔍 Detailed Agent Workflow

### 1. Data Ingestion Agent
- **Encoding Detection**: Automatically detects CSV encoding (UTF-8, Latin-1, etc.)
- **Format Validation**: Handles malformed CSV files gracefully
- **Metadata Extraction**: Captures file statistics and structure info
- **Error Handling**: Robust error handling for network and file issues

### 2. Inspector Agent
- **Null Value Analysis**: Identifies missing data patterns
- **Duplicate Detection**: Finds and reports duplicate records
- **Type Inference**: Suggests appropriate data types for columns
- **Categorical Consistency**: Detects case/spacing inconsistencies
- **Outlier Detection**: Uses IQR method for anomaly detection
- **Business Rule Validation**: Healthcare-specific validation rules

### 3. Cleaning Agent
- **Smart Imputation**: Context-aware missing value handling
- **Deduplication**: Removes duplicate records with logging
- **Type Conversion**: Safe conversion of data types
- **Standardization**: Normalizes categorical values
- **Outlier Treatment**: Clips or corrects extreme values
- **Column Naming**: Standardizes column names

### 4. Transforming Agent
- **Feature Engineering**: Creates derived healthcare metrics
- **Datetime Processing**: Extracts useful date/time features
- **Encoding**: Binary, one-hot, and label encoding for categories
- **Aggregation**: Creates group-based statistical features
- **Normalization**: Standardizes numeric features when needed
- **Column Organization**: Logical ordering of columns

### 5. Verification Agent
- **Data Integrity**: Validates processing didn't corrupt data
- **Quality Metrics**: Calculates comprehensive quality scores
- **Business Logic**: Verifies healthcare domain constraints
- **Transformation Validation**: Ensures all steps completed correctly
- **Quality Grading**: Assigns A-F grades based on quality metrics

### 6. Analysis Agent
- **Statistical Analysis**: Comprehensive descriptive statistics
- **Domain Insights**: Healthcare-specific metric calculations
- **Geographic Analysis**: Identifies underserved areas
- **Policy Recommendations**: AI-driven policy suggestions
- **Visualization**: Creates charts and graphs automatically
- **Report Generation**: Produces markdown reports and summaries

## 📊 Output Examples

### CLI Output
```
🔍 Dataset Processing Summary
┌─────────────────────────────┬──────────────────────────────────┐
│ Metric                      │ Value                            │
├─────────────────────────────┼──────────────────────────────────┤
│ Final Shape                 │ 95 rows × 18 columns             │
│ Cleaning Operations         │ 8                                │
│ Transformation Operations   │ 12                               │
│ Quality Score               │ 85/100 (Grade: B)               │
│ Warnings                    │ 2                                │
│ Errors                      │ 0                                │
└─────────────────────────────┴──────────────────────────────────┘

📊 Key Healthcare Metrics
┌─────────────────────────────┬─────────┐
│ Metric                      │ Value   │
├─────────────────────────────┼─────────┤
│ Total Beds                  │ 24750   │
│ Average Beds Per Hospital   │ 260.53  │
│ Hospitals With Zero Beds    │ 3       │
└─────────────────────────────┴─────────┘

🎯 Policy Recommendations
1. 📍 Address geographic inequality: 3 locations have significantly fewer hospitals. Priority areas: Nizamabad, Khammam, Mahbubnagar
2. 🏥 Infrastructure development needed: 3 facilities have zero bed capacity
3. 📊 Improve data collection: 12.5% missing data affects analysis reliability
```

### Generated Reports
- **insights.md**: Comprehensive analysis with recommendations
- **Visualizations**: Distribution plots, correlation heatmaps, geographic charts
- **Processing Logs**: Complete audit trail of all operations
- **Quality Report**: Detailed data quality assessment

## 🎯 Use Cases

### Healthcare Policy Planning
- **Geographic Equity Analysis**: Identify underserved districts
- **Capacity Planning**: Optimize bed allocation across regions
- **Infrastructure Assessment**: Evaluate healthcare facility distribution
- **Resource Allocation**: Data-driven budget planning

### Data Quality Management
- **Automated Validation**: Continuous data quality monitoring
- **Standardization**: Consistent data formats across sources
- **Error Detection**: Identify and fix data collection issues
- **Compliance**: Ensure data meets quality standards

### Research and Analytics
- **Trend Analysis**: Identify patterns in healthcare data
- **Comparative Studies**: Benchmark performance across regions
- **Predictive Modeling**: Prepare data for ML/AI models
- **Report Generation**: Automated insights for stakeholders

## 🧪 Testing

```bash
# Run comprehensive tests
python run_tests.py

# Test individual agents
python -m pytest tests/ -v

# Test with sample data
python example_usage.py

## 🤝 Contributing
We welcome contributions! This project was built for a hackathon but is designed to be production-ready and extensible.

### Development Setup
```bash
git clone https://github.com/Shiva-Prasad-Naroju/RTGS-AI-Analyst.git
cd RTGS-AI-Analyst
pip install -r requirements.txt
pip install -e .
```

### Adding New Agents
1. Create new agent in `agents/` directory
2. Inherit from base agent class (if needed)
3. Add to `agents/__init__.py`
4. Update CLI interface in `cli.py`
5. Add tests in `run_tests.py`

### Extension Ideas
- Web dashboard for interactive analysis
- Real-time data processing capabilities
- Integration with more data sources
- Advanced ML/AI model integration
- Multi-language support for reports

## 📜 License

MIT License - see LICENSE file for details.

## 🏆 Hackathon Project

This project was developed for a healthcare data analysis hackathon, demonstrating the power of AI agents for automated data processing and policy insights generation.

**Team**: RTGS AI Analysts
**Focus**: Telangana Healthcare Data Analysis
**Goal**: Democratize data analysis for better healthcare policy decisions

---

*For more examples and advanced usage, check out the `examples/` directory and our documentation wiki.*
"""
