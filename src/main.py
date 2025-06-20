import os
import sys
import pandas as pd
from pathlib import Path

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_profiler import DataProfiler
from quality_checker import QualityChecker
from catalog_generator import CatalogGenerator

class DataQualityOrchestrator:
    """
    Main orchestrator class that coordinates data profiling, quality checking,
    and catalog generation for comprehensive data governance.
    """
    
    def __init__(self, data_dir, config_dir, output_dir):
        """
        Initialize the orchestrator with directory paths.
        
        Args:
            data_dir (str): Directory containing data files
            config_dir (str): Directory containing configuration files
            output_dir (str): Directory for output files
        """
        self.data_dir = Path(data_dir)
        self.config_dir = Path(config_dir)
        self.output_dir = Path(output_dir)
        
        # Initialize components
        self.profiler = DataProfiler()
        self.quality_checker = QualityChecker(self.config_dir / 'quality_rules.yaml')
        self.catalog_generator = CatalogGenerator()
        
        # Storage for loaded datasets
        self.datasets = {}
        
        # Ensure output directory exists
        self.output_dir.mkdir(exist_ok=True)
    
    def load_datasets(self):
        """Load all CSV datasets from the data directory."""
        print("Loading datasets...")
        
        csv_files = list(self.data_dir.glob('*.csv'))
        
        if not csv_files:
            print(f"No CSV files found in {self.data_dir}")
            return
        
        for csv_file in csv_files:
            try:
                dataset_name = csv_file.stem
                df = pd.read_csv(csv_file)
                self.datasets[dataset_name] = df
                print(f"✅ Loaded {dataset_name}: {len(df)} rows, {len(df.columns)} columns")
            except Exception as e:
                print(f"❌ Error loading {csv_file}: {e}")
        
        print(f"\nTotal datasets loaded: {len(self.datasets)}")
    
    def run_data_profiling(self):
        """Run data profiling on all loaded datasets."""
        print("\n" + "="*60)
        print("RUNNING DATA PROFILING")
        print("="*60)
        
        profile_results = {}
        
        for dataset_name, df in self.datasets.items():
            print(f"\nProfiling {dataset_name}...")
            try:
                profile = self.profiler.profile_dataset(df, dataset_name)
                profile_results[dataset_name] = profile
                
                # Print summary
                basic_info = profile['basic_info']
                quality_summary = profile['data_quality_summary']
                
                print(f"  📊 Rows: {basic_info['total_rows']}")
                print(f"  📊 Columns: {basic_info['total_columns']}")
                print(f"  📊 Memory: {basic_info['memory_usage_mb']} MB")
                print(f"  📊 Duplicates: {basic_info['duplicate_rows']}")
                print(f"  📊 Quality Score: {quality_summary['data_quality_score']}")
                print(f"  📊 Completeness: {quality_summary['overall_completeness']}%")
                
            except Exception as e:
                print(f"❌ Error profiling {dataset_name}: {e}")
        
        # Export profiling results
        profile_output_json = self.output_dir / 'data_profiles.json'
        profile_output_csv = self.output_dir / 'data_profiles.csv'
        
        self.profiler.export_profile(profile_output_json, 'json')
        self.profiler.export_profile(profile_output_csv, 'csv')
        
        return profile_results
    
    def run_quality_validation(self):
        """Run quality validation on all loaded datasets."""
        print("\n" + "="*60)
        print("RUNNING QUALITY VALIDATION")
        print("="*60)
        
        validation_results = {}
        
        for dataset_name, df in self.datasets.items():
            print(f"\nValidating {dataset_name}...")
            try:
                # Prepare reference data for integrity checks
                reference_data = {name: data for name, data in self.datasets.items() if name != dataset_name}
                
                validation_result = self.quality_checker.validate_dataset(
                    df, dataset_name, reference_data
                )
                validation_results[dataset_name] = validation_result
                
                # Print summary
                status = validation_result['overall_status']
                issues = validation_result['total_issues']
                status_emoji = "✅" if status == "PASS" else "❌"
                
                print(f"  {status_emoji} Status: {status}")
                print(f"  📋 Total Issues: {issues}")
                
                if issues > 0:
                    print("  🔍 Issues by column:")
                    for col_name, col_result in validation_result['column_validations'].items():
                        if col_result['status'] == 'FAIL':
                            print(f"    - {col_name}: {len(col_result['failed_rules'])} issues")
                
            except Exception as e:
                print(f"❌ Error validating {dataset_name}: {e}")
        
        # Export quality report
        quality_report_path = self.output_dir / 'quality_report.json'
        self.quality_checker.export_quality_report(quality_report_path)
        
        return validation_results
    
    def generate_data_catalog(self, profile_results, validation_results):
        """Generate comprehensive data catalog."""
        print("\n" + "="*60)
        print("GENERATING DATA CATALOG")
        print("="*60)
        
        for dataset_name in self.datasets.keys():
            print(f"\nAdding {dataset_name} to catalog...")
            
            try:
                profile_data = profile_results.get(dataset_name, {})
                quality_data = validation_results.get(dataset_name, {})
                file_path = str(self.data_dir / f"{dataset_name}.csv")
                
                self.catalog_generator.add_dataset_to_catalog(
                    dataset_name, profile_data, quality_data, file_path
                )
                
                print(f"  ✅ Added {dataset_name} to catalog")
                
            except Exception as e:
                print(f"❌ Error adding {dataset_name} to catalog: {e}")
        
        # Add sample lineage relationships
        self._add_sample_lineage()
        
        # Export catalog
        catalog_json_path = self.output_dir / 'data_catalog.json'
        catalog_csv_path = self.output_dir / 'data_catalog.csv'
        
        self.catalog_generator.export_catalog(catalog_json_path, 'json')
        self.catalog_generator.export_catalog(catalog_csv_path, 'csv')
        
        # Print catalog summary
        self.catalog_generator.print_catalog_summary()
    
    def _add_sample_lineage(self):
        """Add sample lineage relationships between datasets."""
        # Example: orders depends on customers and products
        if 'customers' in self.datasets and 'orders' in self.datasets:
            self.catalog_generator.add_lineage_relationship(
                'customers', 'orders', 'lookup', 
                'Orders reference customer information via customer_id'
            )
        
        if 'products' in self.datasets and 'orders' in self.datasets:
            self.catalog_generator.add_lineage_relationship(
                'products', 'orders', 'lookup',
                'Orders reference product information via product_id'
            )
    
    def generate_governance_report(self):
        """Generate a comprehensive governance report."""
        print("\n" + "="*60)
        print("GOVERNANCE INSIGHTS")
        print("="*60)
        
        metrics = self.catalog_generator.catalog_data.get('governance_metrics', {})
        
        print(f"\n📈 COMPLIANCE OVERVIEW")
        print(f"   Compliance Rate: {metrics.get('compliance_rate', 0)}%")
        print(f"   Average Quality Score: {metrics.get('average_quality_score', 0)}")
        print(f"   Governance Maturity: {metrics.get('governance_maturity_score', 0)}")
        
        print(f"\n🔍 DATA CLASSIFICATION")
        classification_dist = metrics.get('data_classification_distribution', {})
        for classification, count in classification_dist.items():
            print(f"   {classification.capitalize()}: {count} datasets")
        
        print(f"\n🔗 LINEAGE TRACKING")
        print(f"   Total Relationships: {metrics.get('total_lineage_relationships', 0)}")
        
        # Recommendations based on findings
        print(f"\n💡 RECOMMENDATIONS")
        
        if metrics.get('compliance_rate', 0) < 100:
            print("   - Address data quality issues in non-compliant datasets")
        
        if metrics.get('governance_maturity_score', 0) < 70:
            print("   - Improve metadata documentation and assign data owners")
        
        sensitive_datasets = sum(1 for d in self.catalog_generator.catalog_data['datasets'].values() 
                               if d['governance_info']['data_classification'] == 'sensitive')
        if sensitive_datasets > 0:
            print(f"   - Review access controls for {sensitive_datasets} sensitive datasets")
        
        if metrics.get('total_lineage_relationships', 0) == 0:
            print("   - Establish data lineage tracking for better governance")
    
    def run_complete_analysis(self):
        """Run the complete data quality and governance analysis."""
        print("🚀 Starting Data Quality and Governance Analysis")
        print("="*60)
        
        try:
            # Step 1: Load datasets
            self.load_datasets()
            
            if not self.datasets:
                print("❌ No datasets to analyze. Please check your data directory.")
                return
            
            # Step 2: Run profiling
            profile_results = self.run_data_profiling()
            
            # Step 3: Run quality validation
            validation_results = self.run_quality_validation()
            
            # Step 4: Generate catalog
            self.generate_data_catalog(profile_results, validation_results)
            
            # Step 5: Generate governance insights
            self.generate_governance_report()
            
            # Final summary
            print("\n" + "="*60)
            print("✅ ANALYSIS COMPLETE!")
            print("="*60)
            print(f"📁 Output files generated in: {self.output_dir}")
            print("   - data_profiles.json/csv (detailed profiling results)")
            print("   - quality_report.json (validation results)")
            print("   - data_catalog.json/csv (comprehensive catalog)")
            print("\n🎯 Use these artifacts for:")
            print("   - Data governance and compliance monitoring")
            print("   - Data quality improvement initiatives")
            print("   - Metadata management and documentation")
            print("   - Data lineage and impact analysis")
            
        except Exception as e:
            print(f"❌ Error during analysis: {e}")
            raise

def main():
    """Main function to run the data quality analysis."""
    # Set up directory paths
    current_dir = Path(__file__).parent.parent  # Go up one level from src/
    data_dir = current_dir / 'data'
    config_dir = current_dir / 'config'
    output_dir = current_dir / 'output'
    
    # Initialize and run orchestrator
    orchestrator = DataQualityOrchestrator(data_dir, config_dir, output_dir)
    orchestrator.run_complete_analysis()

if __name__ == "__main__":
    main()