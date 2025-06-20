import pandas as pd
import json
from datetime import datetime

class CatalogGenerator:
    """
    A class to generate comprehensive data catalogs for governance and lineage tracking.
    This helps organizations maintain metadata and understand data relationships.
    """
    
    def __init__(self):
        self.catalog_data = {
            'catalog_metadata': {
                'created_timestamp': datetime.now().isoformat(),
                'version': '1.0.0',
                'description': 'Data catalog for governance and lineage tracking'
            },
            'datasets': {},
            'lineage': [],
            'governance_metrics': {}
        }
    
    def add_dataset_to_catalog(self, dataset_name, profile_data, quality_data, file_path):
        """
        Add a dataset to the catalog with comprehensive metadata.
        
        Args:
            dataset_name (str): Name of the dataset
            profile_data (dict): Data profiling results
            quality_data (dict): Data quality validation results
            file_path (str): Path to the data file
        """
        catalog_entry = {
            'dataset_name': dataset_name,
            'file_path': file_path,
            'added_timestamp': datetime.now().isoformat(),
            'basic_info': profile_data.get('basic_info', {}),
            'data_quality': {
                'overall_status': quality_data.get('overall_status', 'UNKNOWN'),
                'total_issues': quality_data.get('total_issues', 0),
                'quality_score': profile_data.get('data_quality_summary', {}).get('data_quality_score', 0)
            },
            'columns': self._build_column_metadata(profile_data, quality_data),
            'governance_info': self._build_governance_info(profile_data, quality_data),
            'lineage_info': {
                'upstream_dependencies': [],
                'downstream_dependencies': [],
                'transformation_applied': []
            }
        }
        
        self.catalog_data['datasets'][dataset_name] = catalog_entry
    
    def _build_column_metadata(self, profile_data, quality_data):
        """Build comprehensive column metadata."""
        columns_metadata = {}
        
        profile_columns = profile_data.get('columns', {})
        quality_columns = quality_data.get('column_validations', {})
        
        for column_name, column_profile in profile_columns.items():
            column_quality = quality_columns.get(column_name, {})
            
            columns_metadata[column_name] = {
                'data_type': column_profile.get('data_type', 'unknown'),
                'null_percentage': column_profile.get('null_percentage', 0),
                'unique_percentage': column_profile.get('unique_percentage', 0),
                'quality_status': column_quality.get('status', 'UNKNOWN'),
                'quality_issues': len(column_quality.get('failed_rules', [])),
                'statistics': self._extract_column_statistics(column_profile),
                'governance_tags': self._assign_governance_tags(column_name, column_profile)
            }
        
        return columns_metadata
    
    def _extract_column_statistics(self, column_profile):
        """Extract relevant statistics based on column type."""
        stats = {}
        
        # Common statistics
        stats['null_count'] = column_profile.get('null_count', 0)
        stats['unique_count'] = column_profile.get('unique_count', 0)
        
        # Type-specific statistics
        if 'min_value' in column_profile:
            stats.update({
                'min_value': column_profile.get('min_value'),
                'max_value': column_profile.get('max_value'),
                'mean': column_profile.get('mean'),
                'median': column_profile.get('median')
            })
        elif 'min_length' in column_profile:
            stats.update({
                'min_length': column_profile.get('min_length'),
                'max_length': column_profile.get('max_length'),
                'avg_length': column_profile.get('avg_length')
            })
        elif 'min_date' in column_profile:
            stats.update({
                'min_date': column_profile.get('min_date'),
                'max_date': column_profile.get('max_date'),
                'date_range_days': column_profile.get('date_range_days')
            })
        
        return stats
    
    def _assign_governance_tags(self, column_name, column_profile):
        """Assign governance tags based on column characteristics."""
        tags = []
        
        # Tag based on column name patterns
        if any(identifier in column_name.lower() for identifier in ['id', 'key']):
            tags.append('identifier')
        
        if any(pii in column_name.lower() for pii in ['email', 'phone', 'ssn', 'name']):
            tags.append('pii')
        
        if any(financial in column_name.lower() for financial in ['amount', 'price', 'cost', 'salary']):
            tags.append('financial')
        
        if any(date_field in column_name.lower() for date_field in ['date', 'time', 'created', 'updated']):
            tags.append('temporal')
        
        # Tag based on data characteristics
        if column_profile.get('unique_percentage', 0) == 100:
            tags.append('unique')
        
        if column_profile.get('null_percentage', 0) > 50:
            tags.append('high_null_rate')
        
        return tags
    
    def _build_governance_info(self, profile_data, quality_data):
        """Build governance information for the dataset."""
        return {
            'data_classification': self._classify_dataset(profile_data),
            'compliance_status': 'compliant' if quality_data.get('overall_status') == 'PASS' else 'non_compliant',
            'retention_period': 'not_specified',
            'access_level': 'internal',
            'data_owner': 'not_specified',
            'steward': 'not_specified',
            'last_updated': datetime.now().isoformat(),
            'review_status': 'pending'
        }
    
    def _classify_dataset(self, profile_data):
        """Classify dataset based on content analysis."""
        # Simple classification logic
        columns = profile_data.get('columns', {})
        
        has_pii = any(
            any(pii in col_name.lower() for pii in ['email', 'phone', 'name'])
            for col_name in columns.keys()
        )
        
        has_financial = any(
            any(fin in col_name.lower() for fin in ['amount', 'price', 'cost'])
            for col_name in columns.keys()
        )
        
        if has_pii:
            return 'sensitive'
        elif has_financial:
            return 'financial'
        else:
            return 'general'
    
    def add_lineage_relationship(self, source_dataset, target_dataset, transformation_type, description):
        """Add a lineage relationship between datasets."""
        lineage_entry = {
            'source_dataset': source_dataset,
            'target_dataset': target_dataset,
            'transformation_type': transformation_type,
            'description': description,
            'created_timestamp': datetime.now().isoformat()
        }
        
        self.catalog_data['lineage'].append(lineage_entry)
        
        # Update dataset lineage info
        if source_dataset in self.catalog_data['datasets']:
            self.catalog_data['datasets'][source_dataset]['lineage_info']['downstream_dependencies'].append(target_dataset)
        
        if target_dataset in self.catalog_data['datasets']:
            self.catalog_data['datasets'][target_dataset]['lineage_info']['upstream_dependencies'].append(source_dataset)
    
    def calculate_governance_metrics(self):
        """Calculate overall governance metrics."""
        datasets = self.catalog_data['datasets']
        
        if not datasets:
            return {}
        
        total_datasets = len(datasets)
        compliant_datasets = sum(1 for d in datasets.values() if d['data_quality']['overall_status'] == 'PASS')
        total_columns = sum(len(d['columns']) for d in datasets.values())
        
        # Calculate average quality score
        quality_scores = [d['data_quality']['quality_score'] for d in datasets.values()]
        avg_quality_score = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        # Calculate data classification distribution
        classifications = [d['governance_info']['data_classification'] for d in datasets.values()]
        classification_dist = {cls: classifications.count(cls) for cls in set(classifications)}
        
        metrics = {
            'total_datasets': total_datasets,
            'compliant_datasets': compliant_datasets,
            'compliance_rate': round((compliant_datasets / total_datasets) * 100, 2),
            'total_columns': total_columns,
            'average_quality_score': round(avg_quality_score, 2),
            'data_classification_distribution': classification_dist,
            'total_lineage_relationships': len(self.catalog_data['lineage']),
            'governance_maturity_score': self._calculate_maturity_score(datasets)
        }
        
        self.catalog_data['governance_metrics'] = metrics
        return metrics
    
    def _calculate_maturity_score(self, datasets):
        """Calculate governance maturity score (0-100)."""
        if not datasets:
            return 0
        
        total_score = 0
        for dataset in datasets.values():
            score = 0
            
            # Data quality (40 points)
            if dataset['data_quality']['overall_status'] == 'PASS':
                score += 40
            elif dataset['data_quality']['total_issues'] < 5:
                score += 20
            
            # Metadata completeness (30 points)
            if dataset['governance_info']['data_owner'] != 'not_specified':
                score += 10
            if dataset['governance_info']['steward'] != 'not_specified':
                score += 10
            if dataset['governance_info']['data_classification'] != 'general':
                score += 10
            
            # Lineage tracking (20 points)
            upstream = len(dataset['lineage_info']['upstream_dependencies'])
            downstream = len(dataset['lineage_info']['downstream_dependencies'])
            if upstream > 0 or downstream > 0:
                score += 20
            
            # Documentation (10 points)
            if len(dataset['columns']) > 0:
                score += 10
            
            total_score += score
        
        return round(total_score / len(datasets), 2)
    
    def export_catalog(self, output_path, format='json'):
        """Export the data catalog to file."""
        # Calculate governance metrics before export
        self.calculate_governance_metrics()
        
        if format.lower() == 'json':
            with open(output_path, 'w') as f:
                json.dump(self.catalog_data, f, indent=2, default=str)
        
        elif format.lower() == 'csv':
            # Create a flattened view for CSV export
            flattened_data = []
            
            for dataset_name, dataset_info in self.catalog_data['datasets'].items():
                for column_name, column_info in dataset_info['columns'].items():
                    row = {
                        'dataset_name': dataset_name,
                        'column_name': column_name,
                        'data_type': column_info['data_type'],
                        'null_percentage': column_info['null_percentage'],
                        'unique_percentage': column_info['unique_percentage'],
                        'quality_status': column_info['quality_status'],
                        'quality_issues': column_info['quality_issues'],
                        'governance_tags': ', '.join(column_info['governance_tags']),
                        'dataset_quality_score': dataset_info['data_quality']['quality_score'],
                        'dataset_status': dataset_info['data_quality']['overall_status'],
                        'data_classification': dataset_info['governance_info']['data_classification'],
                        'file_path': dataset_info['file_path']
                    }
                    
                    # Add statistics
                    for stat_name, stat_value in column_info['statistics'].items():
                        row[f'stat_{stat_name}'] = stat_value
                    
                    flattened_data.append(row)
            
            df = pd.DataFrame(flattened_data)
            df.to_csv(output_path, index=False)
        
        print(f"Data catalog exported to {output_path}")
    
    def print_catalog_summary(self):
        """Print a summary of the data catalog."""
        metrics = self.catalog_data.get('governance_metrics', {})
        
        print("\n" + "="*60)
        print("DATA CATALOG SUMMARY")
        print("="*60)
        print(f"Total Datasets: {metrics.get('total_datasets', 0)}")
        print(f"Compliant Datasets: {metrics.get('compliant_datasets', 0)}")
        print(f"Compliance Rate: {metrics.get('compliance_rate', 0)}%")
        print(f"Average Quality Score: {metrics.get('average_quality_score', 0)}")
        print(f"Governance Maturity Score: {metrics.get('governance_maturity_score', 0)}")
        print(f"Total Lineage Relationships: {metrics.get('total_lineage_relationships', 0)}")
        
        print("\nData Classification Distribution:")
        classification_dist = metrics.get('data_classification_distribution', {})
        for classification, count in classification_dist.items():
            print(f"  {classification.capitalize()}: {count}")
        
        print("\nDataset Details:")
        for dataset_name, dataset_info in self.catalog_data['datasets'].items():
            print(f"\n  📊 {dataset_name}")
            print(f"    Status: {dataset_info['data_quality']['overall_status']}")
            print(f"    Quality Score: {dataset_info['data_quality']['quality_score']}")
            print(f"    Rows: {dataset_info['basic_info'].get('total_rows', 0)}")
            print(f"    Columns: {dataset_info['basic_info'].get('total_columns', 0)}")
            print(f"    Classification: {dataset_info['governance_info']['data_classification']}")
        
        print("="*60)