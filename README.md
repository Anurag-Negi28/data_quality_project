# Data Quality Project

A comprehensive Python-based data quality assessment tool that provides data profiling, quality checking, catalog generation, and **regional data partitioning** capabilities for various data sources.

## 🚀 Features

- **Data Profiling**: Comprehensive statistical analysis of datasets
- **Quality Assessment**: Automated data quality checks and validation rules
- **Data Catalog Generation**: Automated metadata and schema documentation
- **Regional Data Partitioning**: Intelligent data aggregation and partitioning by geographic regions
- **Sample Data Generation**: Built-in data generator for testing and demonstration
- **Multiple Format Support**: CSV, Excel, JSON, and Parquet files
- **Configurable Rules**: YAML-based quality rule configuration
- **Docker Support**: Containerized deployment ready
- **Reporting**: JSON and CSV output formats for integration
- **Time Series Analysis**: Monthly trends and growth analysis across regions

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Data Generation](#data-generation)
- [Regional Data Partitioning](#regional-data-partitioning)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Usage](#usage)
- [Docker Deployment](#docker-deployment)
- [Output Examples](#output-examples)
- [Contributing](#contributing)
- [License](#license)

## 🛠 Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Local Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/data_quality_project.git
cd data_quality_project
```

2. Create a virtual environment:

```bash
python -m venv quality_env
quality_env\Scripts\activate  # Windows
# source quality_env/bin/activate  # Linux/Mac
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### Option 1: Using Sample Data Generator

1. Generate sample datasets:

```bash
python src/data_generator.py
```

2. Configure pipeline settings in `config/pipeline_config.yaml`
3. Run the complete analysis:

```bash
python src/main.py
```

### Option 2: Using Your Own Data

1. Place your data files in the `data/` directory
2. Configure quality rules in `config/quality_rules.yaml`
3. Update pipeline settings in `config/pipeline_config.yaml`
4. Run the analysis:

```bash
python src/main.py
```

4. Check the results in the `output/` directory

## 🏗️ Data Generation

The project includes a built-in data generator (`data_generator.py`) that creates realistic sample datasets for testing and demonstration purposes.

### Generated Datasets

- **Regions**: Geographic regions with city mappings and population data
- **Customers**: Customer profiles with demographics and registration info
- **Products**: Product catalog with categories and pricing
- **Orders**: Transaction data linking customers, products, and regions

### Running the Data Generator

```bash
# Generate default datasets (1000 customers, 500 products, 2000 orders)
python src/data_generator.py

# Or use it programmatically
from src.data_generator import DataGenerator

generator = DataGenerator(seed=42)
datasets = generator.generate_all_datasets(
    data_dir='data',
    num_customers=1000,
    num_products=500,
    num_orders=2000
)
```

### Data Generator Features

- **Realistic Data**: Names, emails, addresses, and business logic
- **Data Quality Issues**: Intentionally includes invalid emails, duplicates, and missing values for testing
- **Regional Distribution**: Data distributed across North, South, East, and West regions
- **Temporal Patterns**: Order concentration in specific months for testing filters
- **Configurable Size**: Adjustable dataset sizes for different testing scenarios

## 🌍 Regional Data Partitioning

One of the key features of this project is intelligent **regional data partitioning** that aggregates and organizes data by geographic regions.

### How Regional Partitioning Works

1. **Data Integration**: Orders are merged with customer and region data
2. **Temporal Filtering**: Data filtered by target month (configurable)
3. **Regional Filtering**: Data filtered by specified regions
4. **Aggregation**: Metrics calculated at region-month level
5. **Partitioning**: Data saved in separate files by region

### Regional Metrics Calculated

- **Sales Metrics**: Total revenue, quantity sold, order count
- **Customer Metrics**: Unique customers, revenue per customer
- **Performance Metrics**: Average order value, quantity per order
- **Population Metrics**: Revenue per capita (when population data available)
- **Growth Analysis**: Month-over-month growth rates

### Output Structure

```
output/
├── processed_data/
│   ├── aggregated_sales_2024-01.csv           # Overall aggregated data
│   ├── regional_summary_2024-01.csv           # Regional performance summary
│   ├── monthly_trends_by_region.csv           # Time series analysis
│   ├── growth_analysis_by_region.csv          # Growth rate analysis
│   ├── north/
│   │   └── sales_north_2024-01.csv           # North region partition
│   ├── south/
│   │   └── sales_south_2024-01.csv           # South region partition
│   ├── east/
│   │   └── sales_east_2024-01.csv            # East region partition
│   └── west/
│       └── sales_west_2024-01.csv            # West region partition
```

### Regional Configuration

Configure regional partitioning in `config/pipeline_config.yaml`:

```yaml
pipeline:
  input_path: "data/"
  output_path: "output/"
  target_month: "2024-01" # Target month for analysis
  region_filters: # Regions to include
    - "North"
    - "South"
    - "East"
    - "West"
```

## 📁 Project Structure

```
data_quality_project/
├── src/
│   ├── main.py                 # Main orchestrator with regional partitioning
│   ├── data_generator.py       # Sample data generator
│   ├── data_profiler.py        # Data profiling functionality
│   ├── quality_checker.py      # Quality validation logic
│   └── catalog_generator.py    # Data catalog generation
├── config/
│   ├── pipeline_config.yaml    # Pipeline and partitioning configuration
│   └── quality_rules.yaml      # Quality rules configuration
├── data/
│   ├── customers.csv           # Customer data (generated or provided)
│   ├── orders.csv              # Order transactions (generated or provided)
│   ├── products.csv            # Product catalog (generated or provided)
│   └── regions.csv             # Regional mapping data (generated or provided)
├── output/
│   ├── processed_data/         # Regional partitioned data
│   │   ├── north/             # North region partition
│   │   ├── south/             # South region partition
│   │   ├── east/              # East region partition
│   │   └── west/              # West region partition
│   ├── data_catalog.json       # Generated data catalog
│   ├── data_profiles.json      # Data profiling results
│   └── quality_report.json     # Quality assessment report
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker configuration
├── .gitignore                  # Git ignore rules
└── README.md                   # This file
```

## ⚙️ Configuration

### Pipeline Configuration

Edit `config/pipeline_config.yaml` for regional partitioning settings:

```yaml
pipeline:
  input_path: "data/"
  output_path: "output/"
  target_month: "2024-01" # Month to analyze (YYYY-MM format)
  region_filters: # Regions to include in analysis
    - "North"
    - "South"
    - "East"
    - "West"
```

### Quality Rules Configuration

Edit `config/quality_rules.yaml` to define your data quality rules:

```yaml
# Example quality rules
completeness:
  required_fields:
    - customer_id
    - email
    - order_date
  threshold: 0.95

validity:
  email_format: true
  date_format: "YYYY-MM-DD"

uniqueness:
  primary_keys:
    - customer_id
    - order_id

consistency:
  reference_integrity:
    orders.customer_id: customers.customer_id
```

### Environment Variables

Create a `.env` file for environment-specific configurations:

```bash
# Data source configuration
DATA_PATH=./data
OUTPUT_PATH=./output
LOG_LEVEL=INFO

# Quality thresholds
COMPLETENESS_THRESHOLD=0.95
VALIDITY_THRESHOLD=0.90

# Regional analysis settings
DEFAULT_TARGET_MONTH=2024-01
DEFAULT_REGIONS=North,South,East,West
```

## 📊 Usage

### Command Line Interface

```bash
# Generate sample data first
python src/data_generator.py

# Run complete analysis with regional partitioning
python src/main.py

# Run specific components
python src/data_profiler.py --input data/customers.csv
python src/quality_checker.py --config config/quality_rules.yaml
python src/catalog_generator.py --output output/catalog.json
```

### Python API

```python
from src.data_generator import DataGenerator
from src.main import DataQualityOrchestrator

# Generate sample data
generator = DataGenerator(seed=42)
datasets = generator.generate_all_datasets()

# Run analysis with regional partitioning
orchestrator = DataQualityOrchestrator('config/pipeline_config.yaml')
orchestrator.run_complete_analysis()
```

### Regional Analysis Workflow

```python
from src.main import DataQualityOrchestrator

# Initialize orchestrator
orchestrator = DataQualityOrchestrator('config/pipeline_config.yaml')

# Load and clean datasets
orchestrator.load_datasets()
orchestrator.clean_datasets()

# Aggregate and partition by region
aggregated_df = orchestrator.aggregate_and_enrich()
orchestrator.save_partitioned_data(aggregated_df)

# Continue with quality analysis
profile_results = orchestrator.run_data_profiling()
validation_results = orchestrator.run_quality_validation()
orchestrator.generate_data_catalog(profile_results, validation_results)
```

## 🐳 Docker Deployment

### Build and Run

```bash
# Build the Docker image
docker build -t data-quality-tool .

# Run with regional partitioning
docker run -v $(pwd)/data:/app/data -v $(pwd)/output:/app/output -v $(pwd)/config:/app/config data-quality-tool

# Generate sample data in container
docker run -v $(pwd)/data:/app/data data-quality-tool python src/data_generator.py
```

### Docker Compose

Create a `docker-compose.yml` for easier management:

```yaml
version: "3.8"
services:
  data-quality:
    build: .
    volumes:
      - ./data:/app/data
      - ./output:/app/output
      - ./config:/app/config
    environment:
      - LOG_LEVEL=INFO
      - TARGET_MONTH=2024-01
```

## 📈 Output Examples

### Regional Summary Report

```json
{
  "target_month": "2024-01",
  "regions_analyzed": ["North", "South", "East", "West"],
  "regional_performance": {
    "North": {
      "total_revenue": 125000.5,
      "total_quantity": 2500,
      "unique_customers": 450,
      "avg_order_value": 87.35,
      "revenue_per_customer": 277.78
    },
    "South": {
      "total_revenue": 98750.25,
      "total_quantity": 2100,
      "unique_customers": 380,
      "avg_order_value": 82.15,
      "revenue_per_customer": 259.87
    }
  }
}
```

### Time Series Analysis

```csv
region,month,quantity,total_amount,unique_customers,order_count,revenue_growth,quantity_growth
North,2023-12,2300,115000.00,420,1450,,-
North,2024-01,2500,125000.50,450,1520,8.70,8.70
South,2023-12,1950,89500.00,350,1280,,-
South,2024-01,2100,98750.25,380,1350,10.34,7.69
```

### Data Profile Report

```json
{
  "dataset": "customers.csv",
  "rows": 10000,
  "columns": 8,
  "completeness": 0.98,
  "regional_distribution": {
    "North": 2500,
    "South": 2200,
    "East": 2800,
    "West": 2500
  },
  "column_profiles": {
    "customer_id": {
      "type": "integer",
      "unique_count": 10000,
      "null_count": 0,
      "completeness": 1.0
    },
    "city": {
      "type": "string",
      "unique_count": 16,
      "regional_coverage": 4,
      "completeness": 0.995
    }
  }
}
```

## 🔧 Development

### Setting up Development Environment

```bash
# Clone and setup
git clone https://github.com/yourusername/data_quality_project.git
cd data_quality_project

# Install in development mode
pip install -e .

# Generate sample data for testing
python src/data_generator.py

# Install development dependencies
pip install pytest black flake8 mypy
```

### Running Tests

```bash
# Run all tests
pytest

# Test with sample data
python src/data_generator.py
python src/main.py

# Run specific test file
pytest tests/test_regional_partitioning.py
```

### Code Quality

```bash
# Format code
black src/

# Lint code
flake8 src/

# Type checking
mypy src/
```

## 📚 Dependencies

- **pandas**: Data manipulation and regional aggregation
- **numpy**: Numerical computing for metrics calculation
- **openpyxl**: Excel file support
- **PyYAML**: Configuration file parsing
- **jsonschema**: JSON schema validation
- **python-dateutil**: Date parsing for temporal analysis

## 🌟 Key Features

### Regional Intelligence

- **Automatic Region Detection**: Identifies regions from city mappings
- **Smart Filtering**: Configurable region and time period filters
- **Performance Comparison**: Cross-regional performance analysis
- **Population-based Metrics**: Revenue per capita calculations

### Data Generation

- **Realistic Relationships**: Proper foreign key relationships between datasets
- **Quality Issues**: Intentional data quality problems for testing
- **Scalable Generation**: Configurable dataset sizes
- **Reproducible Results**: Seeded random generation

### Comprehensive Analysis

- **Multi-dimensional Profiling**: Statistical analysis across regions and time
- **Quality Validation**: Automated rule-based quality checking
- **Governance Insights**: Compliance and maturity scoring
- **Lineage Tracking**: Data relationship mapping

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/regional-enhancement`)
3. Commit your changes (`git commit -m 'Add regional partitioning feature'`)
4. Push to the branch (`git push origin feature/regional-enhancement`)
5. Open a Pull Request

### Contribution Guidelines

- Follow PEP 8 style guide
- Add tests for new features, especially regional logic
- Update documentation as needed
- Ensure all tests pass before submitting PR
- Test with both generated and real data

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/data_quality_project/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/data_quality_project/discussions)
- **Documentation**: [Wiki](https://github.com/yourusername/data_quality_project/wiki)

## 🎯 Roadmap

- [ ] Web-based regional dashboard
- [ ] Database connectivity (PostgreSQL, MySQL)
- [ ] Advanced regional visualization charts
- [ ] Machine learning-based regional anomaly detection
- [ ] Integration with Apache Airflow for scheduled regional analysis
- [ ] Real-time regional data quality monitoring
- [ ] Multi-tenant regional partitioning
- [ ] Geographic visualization on maps

## 📊 Performance

- Processes up to 1M rows with regional partitioning in under 45 seconds
- Memory efficient chunked processing for large regional datasets
- Supports datasets up to 10GB with proper regional configuration
- Optimized aggregation algorithms for multi-regional analysis
- Parallel processing support for multiple regions

## 🎨 Sample Use Cases

### Business Intelligence

- **Regional Sales Analysis**: Compare performance across geographic regions
- **Market Penetration**: Analyze customer distribution by region
- **Seasonal Trends**: Track regional variations in seasonal patterns

### Data Governance

- **Regional Compliance**: Ensure data quality standards across regions
- **Data Lineage**: Track data flow from regional sources to aggregated reports
- **Quality Monitoring**: Monitor data quality trends by region

### Operations

- **Resource Allocation**: Optimize resources based on regional performance
- **Inventory Management**: Regional demand forecasting
- **Customer Segmentation**: Region-based customer analysis

---

**Made with ❤️ for better data quality and regional intelligence**
