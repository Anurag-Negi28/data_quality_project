import os
import sys
import pandas as pd
from pathlib import Path
import yaml

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_profiler import DataProfiler
from quality_checker import QualityChecker
from catalog_generator import CatalogGenerator

class DataQualityOrchestrator:
    """Main orchestrator class that coordinates data profiling, quality checking, cleaning, aggregation, and catalog generation."""
    
    def __init__(self, config_path):
        """Initialize with configuration file path."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)['pipeline']
        self.data_dir = Path(config['input_path'])
        self.output_dir = Path(config['output_path'])
        self.target_month = config['target_month']
        self.region_filters = config['region_filters']
        self.config_dir = Path(config_path).parent
        
        # Initialize components
        self.profiler = DataProfiler()
        self.quality_checker = QualityChecker(self.config_dir / 'quality_rules.yaml')
        self.catalog_generator = CatalogGenerator()
        self.datasets = {}
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

    def clean_datasets(self):
        """Clean datasets by handling missing values, standardizing dates, and removing duplicates."""
        print("\n" + "="*60)
        print("CLEANING DATASETS")
        print("="*60)
        for dataset_name, df in self.datasets.items():
            if dataset_name == 'regions':
                continue  # Skip reference data
            print(f"\nCleaning {dataset_name}...")
            try:
                # Handle missing values
                if dataset_name == 'customers':
                    df['email'] = df['email'].fillna('unknown@example.com')
                    df['age'] = df['age'].fillna(df['age'].median())
                    df['city'] = df['city'].fillna('Unknown')
                elif dataset_name == 'orders':
                    df['quantity'] = df['quantity'].fillna(0)
                    df['total_amount'] = df['total_amount'].fillna(0)
                
                # Standardize date formats
                date_cols = ['registration_date', 'order_date'] if dataset_name in ['customers', 'orders'] else []
                for col in date_cols:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col], errors='coerce').dt.strftime('%Y-%m-%d')
                
                # Remove duplicates
                original_len = len(df)
                df = df.drop_duplicates()
                print(f"  Removed {original_len - len(df)} duplicate rows")
                
                self.datasets[dataset_name] = df
                print(f"✅ Cleaned {dataset_name}")
            except Exception as e:
                print(f"❌ Error cleaning {dataset_name}: {e}")

    def aggregate_and_enrich(self):
        """Aggregate orders by region and month, enrich with region metadata."""
        print("\n" + "="*60)
        print("AGGREGATING AND ENRICHING DATA")
        print("="*60)
        if 'orders' in self.datasets and 'customers' in self.datasets and 'regions' in self.datasets:
            try:
                orders_df = self.datasets['orders'].copy()
                customers_df = self.datasets['customers'].copy()
                regions_df = self.datasets['regions'].copy()
                
                print(f"📊 Initial data sizes:")
                print(f"   Orders: {len(orders_df)} rows")
                print(f"   Customers: {len(customers_df)} rows") 
                print(f"   Regions: {len(regions_df)} rows")
                
                # Merge datasets
                merged_df = orders_df.merge(customers_df[['customer_id', 'city']], on='customer_id', how='left')
                print(f"   After customer merge: {len(merged_df)} rows")
                
                merged_df = merged_df.merge(regions_df, on='city', how='inner')
                print(f"   After region merge: {len(merged_df)} rows")
                
                # Extract month and add additional time dimensions
                merged_df['order_date'] = pd.to_datetime(merged_df['order_date'])
                merged_df['month'] = merged_df['order_date'].dt.to_period('M').astype(str)
                merged_df['year'] = merged_df['order_date'].dt.year
                merged_df['quarter'] = merged_df['order_date'].dt.quarter
                merged_df['day_of_week'] = merged_df['order_date'].dt.day_name()
                
                print(f"📅 Available months: {sorted(merged_df['month'].unique())}")
                print(f"🌍 Available regions: {sorted(merged_df['region'].unique())}")
                print(f"🎯 Target month: {self.target_month}")
                print(f"🎯 Target regions: {self.region_filters}")
                
                # Apply filters
                filtered_df = merged_df[
                    (merged_df['month'] == self.target_month) &
                    (merged_df['region'].isin(self.region_filters))
                ]
                print(f"   After filters: {len(filtered_df)} rows")
                
                if len(filtered_df) == 0:
                    print("⚠️  No data remains after filtering! Check your target_month and region_filters in config.")
                    # Create empty DataFrame with expected columns instead of None
                    empty_df = pd.DataFrame(columns=[
                        'region', 'month', 'quantity', 'total_amount', 
                        'order_count', 'avg_order_value', 'unique_customers'
                    ])
                    self.datasets['aggregated_sales'] = empty_df
                    return empty_df
                
                # Enhanced aggregation with more metrics
                aggregated_df = filtered_df.groupby(['region', 'month']).agg({
                    'quantity': 'sum',
                    'total_amount': ['sum', 'mean', 'count'],
                    'customer_id': 'nunique',
                    'order_id': 'count'
                }).round(2)
                
                # Flatten column names
                aggregated_df.columns = ['_'.join(col) if col[1] else col[0] for col in aggregated_df.columns]
                aggregated_df = aggregated_df.reset_index()
                
                # Rename columns for clarity
                aggregated_df = aggregated_df.rename(columns={
                    'quantity_sum': 'quantity',
                    'total_amount_sum': 'total_amount',
                    'total_amount_mean': 'avg_order_value',
                    'total_amount_count': 'order_count',
                    'customer_id_nunique': 'unique_customers',
                    'order_id_count': 'total_orders'
                })
                
                # Add calculated metrics
                aggregated_df['quantity_per_order'] = (
                    aggregated_df['quantity'] / aggregated_df['order_count']
                ).round(2)
                
                aggregated_df['revenue_per_customer'] = (
                    aggregated_df['total_amount'] / aggregated_df['unique_customers']
                ).round(2)
                
                # Add region metadata
                region_metadata = regions_df.groupby('region').first().reset_index()
                if 'population' in region_metadata.columns:
                    aggregated_df = aggregated_df.merge(
                        region_metadata[['region', 'population']], 
                        on='region', 
                        how='left'
                    )
                    aggregated_df['revenue_per_capita'] = (
                        aggregated_df['total_amount'] / aggregated_df['population']
                    ).round(4)
                
                print(f"✅ Aggregated data: {len(aggregated_df)} rows")
                print(f"📊 Metrics calculated: {list(aggregated_df.columns)}")
                
                self.datasets['aggregated_sales'] = aggregated_df
                return aggregated_df
                
            except Exception as e:
                print(f"❌ Error aggregating data: {e}")
                # Create empty DataFrame with expected columns instead of None
                empty_df = pd.DataFrame(columns=[
                    'region', 'month', 'quantity', 'total_amount',
                    'order_count', 'avg_order_value', 'unique_customers'
                ])
                self.datasets['aggregated_sales'] = empty_df
                return empty_df
        else:
            print("❌ Required datasets (orders, customers, regions) not found")
            # Create empty DataFrame with expected columns instead of None
            empty_df = pd.DataFrame(columns=[
                'region', 'month', 'quantity', 'total_amount',
                'order_count', 'avg_order_value', 'unique_customers'
            ])
            self.datasets['aggregated_sales'] = empty_df
            return empty_df

    def save_partitioned_data(self, aggregated_df):
        """Save aggregated data partitioned by region with comprehensive analysis."""
        print("\n" + "="*60)
        print("SAVING PARTITIONED DATA")
        print("="*60)
        
        # Check if DataFrame is empty
        if aggregated_df is None or len(aggregated_df) == 0:
            print("⚠️  No aggregated data to save - dataset is empty.")
            return
            
        try:
            # Create the main processed_data directory
            processed_data_dir = self.output_dir / 'processed_data'
            processed_data_dir.mkdir(parents=True, exist_ok=True)
            
            # Save overall aggregated data
            overall_file = processed_data_dir / f"aggregated_sales_{self.target_month}.csv"
            aggregated_df.to_csv(overall_file, index=False)
            print(f"✅ Saved overall aggregated data to {overall_file}")
            
            # Save data partitioned by region
            regions_processed = 0
            for region in aggregated_df['region'].unique():
                region_df = aggregated_df[aggregated_df['region'] == region]
                
                # Create region-specific directory
                region_dir = processed_data_dir / region.lower().replace(' ', '_')
                region_dir.mkdir(parents=True, exist_ok=True)
                
                # Save region-specific data
                region_file = region_dir / f"sales_{region.lower().replace(' ', '_')}_{self.target_month}.csv"
                region_df.to_csv(region_file, index=False)
                
                print(f"✅ Saved {region} data to {region_file} ({len(region_df)} rows)")
                regions_processed += 1
            
            # Save summary statistics by region
            self._save_regional_summary(aggregated_df, processed_data_dir)
            
            # Save time series analysis if applicable
            self.save_time_series_analysis(processed_data_dir)
            
            print(f"\n📊 Summary: Processed {regions_processed} regions")
            print(f"📁 All files saved to: {processed_data_dir}")
            
        except Exception as e:
            print(f"❌ Error saving partitioned data: {e}")

    def _save_regional_summary(self, aggregated_df, processed_data_dir):
        """Save regional summary statistics."""
        try:
            # Calculate regional summaries
            regional_summary = aggregated_df.groupby('region').agg({
                'quantity': ['sum', 'mean', 'count'],
                'total_amount': ['sum', 'mean', 'max', 'min']
            }).round(2)
            
            # Flatten column names
            regional_summary.columns = ['_'.join(col).strip() for col in regional_summary.columns]
            regional_summary = regional_summary.reset_index()
            
            # Add additional metrics
            regional_summary['avg_order_value'] = (
                regional_summary['total_amount_sum'] / regional_summary['quantity_count']
            ).round(2)
            
            regional_summary['quantity_per_order'] = (
                regional_summary['quantity_sum'] / regional_summary['quantity_count']
            ).round(2)
            
            # Sort by total amount descending
            regional_summary = regional_summary.sort_values('total_amount_sum', ascending=False)
            
            # Save summary
            summary_file = processed_data_dir / f"regional_summary_{self.target_month}.csv"
            regional_summary.to_csv(summary_file, index=False)
            
            print(f"✅ Saved regional summary to {summary_file}")
            
            # Print top performing regions
            print(f"\n🏆 TOP PERFORMING REGIONS ({self.target_month}):")
            for idx, row in regional_summary.head(3).iterrows():
                print(f"   {idx+1}. {row['region']}: ${row['total_amount_sum']:,.2f} revenue, {row['quantity_sum']:,} units")
                
        except Exception as e:
            print(f"❌ Error creating regional summary: {e}")

    def save_time_series_analysis(self, processed_data_dir):
        """Save time series analysis if multiple months of data exist."""
        try:
            if 'orders' not in self.datasets or 'customers' not in self.datasets:
                return
                
            orders_df = self.datasets['orders'].copy()
            customers_df = self.datasets['customers'].copy()
            regions_df = self.datasets.get('regions', pd.DataFrame())
            
            if len(regions_df) == 0:
                return
                
            # Merge all data
            full_df = orders_df.merge(customers_df[['customer_id', 'city']], on='customer_id', how='left')
            full_df = full_df.merge(regions_df, on='city', how='inner')
            
            # Extract time dimensions
            full_df['order_date'] = pd.to_datetime(full_df['order_date'])
            full_df['month'] = full_df['order_date'].dt.to_period('M').astype(str)
            
            # Check if we have multiple months
            unique_months = full_df['month'].nunique()
            if unique_months > 1:
                print(f"\n📈 Creating time series analysis for {unique_months} months...")
                
                # Monthly trends by region
                monthly_trends = full_df.groupby(['region', 'month']).agg({
                    'quantity': 'sum',
                    'total_amount': 'sum',
                    'customer_id': 'nunique',
                    'order_id': 'count'
                }).reset_index()
                
                monthly_trends.columns = ['region', 'month', 'quantity', 'total_amount', 'unique_customers', 'order_count']
                
                # Save monthly trends
                trends_file = processed_data_dir / "monthly_trends_by_region.csv"
                monthly_trends.to_csv(trends_file, index=False)
                print(f"✅ Saved monthly trends to {trends_file}")
                
                # Calculate growth rates
                monthly_trends['month_date'] = pd.to_datetime(monthly_trends['month'])
                monthly_trends = monthly_trends.sort_values(['region', 'month_date'])
                
                monthly_trends['revenue_growth'] = monthly_trends.groupby('region')['total_amount'].pct_change() * 100
                monthly_trends['quantity_growth'] = monthly_trends.groupby('region')['quantity'].pct_change() * 100
                
                # Save growth analysis
                growth_file = processed_data_dir / "growth_analysis_by_region.csv"
                monthly_trends.round(2).to_csv(growth_file, index=False)
                print(f"✅ Saved growth analysis to {growth_file}")
                
        except Exception as e:
            print(f"❌ Error creating time series analysis: {e}")

    def run_data_profiling(self):
        """Run data profiling on all loaded datasets."""
        print("\n" + "="*60)
        print("RUNNING DATA PROFILING")
        print("="*60)
        profile_results = {}
        for dataset_name, df in self.datasets.items():
            if dataset_name == 'regions':
                continue  # Skip reference data
                
            # Check if dataset is empty
            if len(df) == 0:
                print(f"\n⚠️  Skipping {dataset_name} - dataset is empty")
                continue
                
            print(f"\nProfiling {dataset_name}...")
            try:
                profile = self.profiler.profile_dataset(df, dataset_name)
                profile_results[dataset_name] = profile
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
                
        # Only export if we have results
        if profile_results:
            profile_output_json = self.output_dir / 'data_profiles.json'
            profile_output_csv = self.output_dir / 'data_profiles.csv'
            self.profiler.export_profile(profile_output_json, 'json')
            self.profiler.export_profile(profile_output_csv, 'csv')
            print(f"\n✅ Profile exported to {profile_output_json}")
            print(f"✅ Profile exported to {profile_output_csv}")
        else:
            print("\n⚠️  No profiles generated - all datasets were empty or skipped")
            
        return profile_results

    def run_quality_validation(self):
        """Run quality validation on all loaded datasets."""
        print("\n" + "="*60)
        print("RUNNING QUALITY VALIDATION")
        print("="*60)
        validation_results = {}
        for dataset_name, df in self.datasets.items():
            if dataset_name == 'regions':
                continue  # Skip reference data
                
            # Check if dataset is empty
            if len(df) == 0:
                print(f"\n⚠️  Skipping {dataset_name} - dataset is empty")
                continue
                
            print(f"\nValidating {dataset_name}...")
            try:
                reference_data = {name: data for name, data in self.datasets.items() 
                                if name != dataset_name and name != 'regions' and len(data) > 0}
                validation_result = self.quality_checker.validate_dataset(df, dataset_name, reference_data)
                validation_results[dataset_name] = validation_result
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
                
        # Only export if we have results
        if validation_results:
            quality_report_path = self.output_dir / 'quality_report.json'
            self.quality_checker.export_quality_report(quality_report_path)
            print(f"\n✅ Quality report exported to {quality_report_path}")
        else:
            print("\n⚠️  No quality validation results - all datasets were empty or skipped")
            
        return validation_results

    def generate_data_catalog(self, profile_results, validation_results):
        """Generate comprehensive data catalog."""
        print("\n" + "="*60)
        print("GENERATING DATA CATALOG")
        print("="*60)
        
        # Only process datasets that have profile results (non-empty datasets)
        datasets_to_catalog = [name for name in self.datasets.keys() 
                             if name != 'regions' and name in profile_results]
        
        if not datasets_to_catalog:
            print("⚠️  No datasets to add to catalog - all datasets were empty")
            return
            
        for dataset_name in datasets_to_catalog:
            print(f"\nAdding {dataset_name} to catalog...")
            try:
                profile_data = profile_results.get(dataset_name, {})
                quality_data = validation_results.get(dataset_name, {})
                file_path = str(self.data_dir / f"{dataset_name}.csv")
                self.catalog_generator.add_dataset_to_catalog(dataset_name, profile_data, quality_data, file_path)
                print(f"  ✅ Added {dataset_name} to catalog")
            except Exception as e:
                print(f"❌ Error adding {dataset_name} to catalog: {e}")
                
        self._add_sample_lineage()
        catalog_json_path = self.output_dir / 'data_catalog.json'
        catalog_csv_path = self.output_dir / 'data_catalog.csv'
        self.catalog_generator.export_catalog(catalog_json_path, 'json')
        self.catalog_generator.export_catalog(catalog_csv_path, 'csv')
        self.catalog_generator.print_catalog_summary()

    def _add_sample_lineage(self):
        """Add sample lineage relationships between datasets."""
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
            self.load_datasets()
            if not self.datasets:
                print("❌ No datasets to analyze. Please check your data directory.")
                return
            self.clean_datasets()
            aggregated_df = self.aggregate_and_enrich()
            self.save_partitioned_data(aggregated_df)
            profile_results = self.run_data_profiling()
            validation_results = self.run_quality_validation()
            self.generate_data_catalog(profile_results, validation_results)
            self.generate_governance_report()
            print("\n" + "="*60)
            print("✅ ANALYSIS COMPLETE!")
            print("="*60)
            print(f"📁 Output files generated in: {self.output_dir}")
            print("   - data_profiles.json/csv (detailed profiling results)")
            print("   - quality_report.json (validation results)")
            print("   - data_catalog.json/csv (comprehensive catalog)")
            print("   - processed_data/<region>/*.csv (partitioned aggregated data)")
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
    current_dir = Path(__file__).parent.parent
    config_path = current_dir / 'config' / 'pipeline_config.yaml'
    orchestrator = DataQualityOrchestrator(config_path)
    orchestrator.run_complete_analysis()

if __name__ == "__main__":
    main()