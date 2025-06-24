import pandas as pd
import numpy as np
from datetime import datetime
import json

class DataProfiler:
    """
    A class to profile datasets and generate comprehensive metadata.
    This helps in understanding data structure, quality, and characteristics.
    """
    
    def __init__(self):
        self.profile_results = {}
    
    def profile_dataset(self, df, dataset_name):
        """
        Profile a pandas DataFrame and return comprehensive metadata.
        
        Args:
            df (pd.DataFrame): The dataset to profile
            dataset_name (str): Name of the dataset
            
        Returns:
            dict: Comprehensive profile of the dataset
        """
        # Check if DataFrame is empty
        if df is None or len(df) == 0:
            profile = {
                'dataset_name': dataset_name,
                'profiling_timestamp': datetime.now().isoformat(),
                'basic_info': {
                    'total_rows': 0,
                    'total_columns': len(df.columns) if df is not None else 0,
                    'memory_usage_mb': 0.0,
                    'duplicate_rows': 0
                },
                'columns': {},
                'data_quality_summary': {
                    'overall_completeness': 0.0,
                    'columns_with_nulls': 0,
                    'completely_null_columns': 0,
                    'data_quality_score': 0.0
                }
            }
            self.profile_results[dataset_name] = profile
            return profile
        
        profile = {
            'dataset_name': dataset_name,
            'profiling_timestamp': datetime.now().isoformat(),
            'basic_info': self._get_basic_info(df),
            'columns': self._profile_columns(df),
            'data_quality_summary': self._get_quality_summary(df)
        }
        
        self.profile_results[dataset_name] = profile
        return profile
    
    def _get_basic_info(self, df):
        """Get basic information about the dataset."""
        if df is None or len(df) == 0:
            return {
                'total_rows': 0,
                'total_columns': len(df.columns) if df is not None else 0,
                'memory_usage_mb': 0.0,
                'duplicate_rows': 0
            }
        
        return {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'memory_usage_mb': round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
            'duplicate_rows': int(df.duplicated().sum())
        }
    
    def _profile_columns(self, df):
        """Profile each column in the dataset."""
        if df is None or len(df) == 0:
            return {}
        
        column_profiles = {}
        
        for column in df.columns:
            try:
                column_profiles[column] = self._profile_single_column(df[column], column)
            except Exception as e:
                print(f"Warning: Error profiling column '{column}': {e}")
                column_profiles[column] = self._get_empty_column_profile()
        
        return column_profiles
    
    def _profile_single_column(self, series, column_name):
        """Profile a single column/series."""
        # Check if series is empty
        if len(series) == 0:
            return self._get_empty_column_profile()
        
        try:
            profile = {
                'data_type': str(series.dtype),
                'null_count': int(series.isnull().sum()),
                'null_percentage': round((series.isnull().sum() / len(series)) * 100, 2),
                'unique_count': int(series.nunique()),
                'unique_percentage': round((series.nunique() / len(series)) * 100, 2)
            }
            
            # Add most frequent value info
            if len(series.dropna()) > 0:
                value_counts = series.value_counts()
                if len(value_counts) > 0:
                    profile['most_frequent_value'] = str(value_counts.index[0])
                    profile['most_frequent_count'] = int(value_counts.iloc[0])
                else:
                    profile['most_frequent_value'] = None
                    profile['most_frequent_count'] = 0
            else:
                profile['most_frequent_value'] = None
                profile['most_frequent_count'] = 0
            
            # Add type-specific statistics
            if pd.api.types.is_numeric_dtype(series):
                profile.update(self._get_numeric_stats(series))
            elif pd.api.types.is_datetime64_any_dtype(series):
                profile.update(self._get_datetime_stats(series))
            else:
                profile.update(self._get_text_stats(series))
            
            return profile
            
        except Exception as e:
            print(f"Warning: Error in detailed profiling for column '{column_name}': {e}")
            return self._get_empty_column_profile()
    
    def _get_empty_column_profile(self):
        """Return a default profile for empty columns."""
        return {
            'data_type': 'object',
            'null_count': 0,
            'null_percentage': 0.0,
            'unique_count': 0,
            'unique_percentage': 0.0,
            'most_frequent_value': None,
            'most_frequent_count': 0,
            'min_value': None,
            'max_value': None,
            'mean': None,
            'median': None,
            'std_deviation': None
        }
    
    def _get_numeric_stats(self, series):
        """Get statistics for numeric columns."""
        non_null_series = series.dropna()
        
        if len(non_null_series) == 0:
            return {
                'min_value': None,
                'max_value': None,
                'mean': None,
                'median': None,
                'std_deviation': None,
                'zero_count': 0,
                'negative_count': 0,
                'q25': None,
                'q75': None
            }
        
        try:
            stats = {
                'min_value': float(non_null_series.min()),
                'max_value': float(non_null_series.max()),
                'mean': round(float(non_null_series.mean()), 2),
                'median': float(non_null_series.median()),
                'std_deviation': round(float(non_null_series.std()), 2) if non_null_series.std() is not pd.NaType else 0.0,
                'zero_count': int((series == 0).sum()),
                'negative_count': int((series < 0).sum()),
                'q25': float(non_null_series.quantile(0.25)),
                'q75': float(non_null_series.quantile(0.75))
            }
            
            # Handle potential NaN values
            for key, value in stats.items():
                if pd.isna(value):
                    stats[key] = None
                    
            return stats
            
        except Exception as e:
            print(f"Warning: Error calculating numeric stats: {e}")
            return {
                'min_value': None,
                'max_value': None,
                'mean': None,
                'median': None,
                'std_deviation': None,
                'zero_count': 0,
                'negative_count': 0,
                'q25': None,
                'q75': None
            }
    
    def _get_datetime_stats(self, series):
        """Get statistics for datetime columns."""
        non_null_series = series.dropna()
        
        if len(non_null_series) == 0:
            return {
                'min_date': None,
                'max_date': None,
                'date_range_days': None,
                'unique_dates': 0
            }
        
        try:
            min_date = non_null_series.min()
            max_date = non_null_series.max()
            
            return {
                'min_date': str(min_date) if pd.notna(min_date) else None,
                'max_date': str(max_date) if pd.notna(max_date) else None,
                'date_range_days': (max_date - min_date).days if pd.notna(min_date) and pd.notna(max_date) else None,
                'unique_dates': int(non_null_series.nunique())
            }
        except Exception as e:
            print(f"Warning: Error calculating datetime stats: {e}")
            return {
                'min_date': None,
                'max_date': None,
                'date_range_days': None,
                'unique_dates': 0
            }
    
    def _get_text_stats(self, series):
        """Get statistics for text columns."""
        non_null_series = series.dropna()
        
        if len(non_null_series) == 0:
            return {
                'min_length': None,
                'max_length': None,
                'avg_length': None,
                'empty_string_count': 0,
                'whitespace_only_count': 0
            }
        
        try:
            # Convert to string and calculate lengths
            string_series = non_null_series.astype(str)
            lengths = string_series.str.len()
            
            return {
                'min_length': int(lengths.min()) if len(lengths) > 0 else None,
                'max_length': int(lengths.max()) if len(lengths) > 0 else None,
                'avg_length': round(float(lengths.mean()), 2) if len(lengths) > 0 else None,
                'empty_string_count': int((string_series == '').sum()),
                'whitespace_only_count': int(string_series.str.strip().eq('').sum())
            }
        except Exception as e:
            print(f"Warning: Error calculating text stats: {e}")
            return {
                'min_length': None,
                'max_length': None,
                'avg_length': None,
                'empty_string_count': 0,
                'whitespace_only_count': 0
            }
    
    def _get_quality_summary(self, df):
        """Get overall data quality summary."""
        if df is None or len(df) == 0:
            return {
                'overall_completeness': 0.0,
                'columns_with_nulls': 0,
                'completely_null_columns': 0,
                'data_quality_score': 0.0
            }
        
        try:
            total_cells = df.shape[0] * df.shape[1]
            
            if total_cells == 0:
                return {
                    'overall_completeness': 0.0,
                    'columns_with_nulls': 0,
                    'completely_null_columns': 0,
                    'data_quality_score': 0.0
                }
            
            null_cells = df.isnull().sum().sum()
            
            return {
                'overall_completeness': round(((total_cells - null_cells) / total_cells) * 100, 2),
                'columns_with_nulls': int(df.isnull().any().sum()),
                'completely_null_columns': int(df.isnull().all().sum()),
                'data_quality_score': self._calculate_quality_score(df)
            }
        except Exception as e:
            print(f"Warning: Error calculating quality summary: {e}")
            return {
                'overall_completeness': 0.0,
                'columns_with_nulls': 0,
                'completely_null_columns': 0,
                'data_quality_score': 0.0
            }
    
    def _calculate_quality_score(self, df):
        """Calculate a simple data quality score (0-100)."""
        if df is None or len(df) == 0 or df.shape[0] * df.shape[1] == 0:
            return 0.0
        
        try:
            # Completeness score (0-100)
            total_cells = df.shape[0] * df.shape[1]
            null_cells = df.isnull().sum().sum()
            completeness = ((total_cells - null_cells) / total_cells) * 100
            
            # Uniqueness factor (simplified)
            if df.shape[0] > 0:
                uniqueness_factor = min((df.nunique().sum() / df.shape[0]) * 10, 30)
            else:
                uniqueness_factor = 0
            
            # Consistency factor (no duplicate rows)
            if df.shape[0] > 0:
                duplicate_factor = max(0, 20 - (df.duplicated().sum() / df.shape[0]) * 20)
            else:
                duplicate_factor = 20
            
            # Weighted score
            score = (completeness * 0.6) + (uniqueness_factor * 0.2) + (duplicate_factor * 0.2)
            return round(min(max(score, 0), 100), 2)
            
        except Exception as e:
            print(f"Warning: Error calculating quality score: {e}")
            return 0.0
    
    def export_profile(self, output_path, format='json'):
        """Export profiling results to file."""
        try:
            if not self.profile_results:
                print("No profile results to export.")
                return
            
            if format.lower() == 'json':
                with open(output_path, 'w') as f:
                    json.dump(self.profile_results, f, indent=2, default=str)
            elif format.lower() == 'csv':
                # Flatten the profile for CSV export
                flattened_data = []
                for dataset_name, profile in self.profile_results.items():
                    basic_info = profile.get('basic_info', {})
                    quality_summary = profile.get('data_quality_summary', {})
                    
                    # Add dataset-level information
                    dataset_row = {
                        'dataset_name': dataset_name,
                        'column_name': '__DATASET_SUMMARY__',
                        'data_type': 'summary',
                        'total_rows': basic_info.get('total_rows', 0),
                        'total_columns': basic_info.get('total_columns', 0),
                        'memory_usage_mb': basic_info.get('memory_usage_mb', 0),
                        'duplicate_rows': basic_info.get('duplicate_rows', 0),
                        'overall_completeness': quality_summary.get('overall_completeness', 0),
                        'data_quality_score': quality_summary.get('data_quality_score', 0)
                    }
                    flattened_data.append(dataset_row)
                    
                    # Add column-level information
                    for column_name, column_profile in profile.get('columns', {}).items():
                        row = {
                            'dataset_name': dataset_name,
                            'column_name': column_name,
                            **column_profile
                        }
                        flattened_data.append(row)
                
                if flattened_data:
                    df = pd.DataFrame(flattened_data)
                    df.to_csv(output_path, index=False)
                else:
                    print("No data to export to CSV.")
                    return
            
            print(f"Profile exported to {output_path}")
            
        except Exception as e:
            print(f"Error exporting profile: {e}")
    
    def print_profile_summary(self, dataset_name=None):
        """Print a summary of the profile results."""
        if not self.profile_results:
            print("No profile results available.")
            return
        
        datasets_to_print = [dataset_name] if dataset_name else list(self.profile_results.keys())
        
        for name in datasets_to_print:
            if name not in self.profile_results:
                print(f"No profile found for dataset: {name}")
                continue
            
            profile = self.profile_results[name]
            basic_info = profile.get('basic_info', {})
            quality_summary = profile.get('data_quality_summary', {})
            
            print(f"\n📊 Profile Summary for: {name}")
            print(f"   Rows: {basic_info.get('total_rows', 0):,}")
            print(f"   Columns: {basic_info.get('total_columns', 0)}")
            print(f"   Memory: {basic_info.get('memory_usage_mb', 0)} MB")
            print(f"   Duplicates: {basic_info.get('duplicate_rows', 0):,}")
            print(f"   Completeness: {quality_summary.get('overall_completeness', 0)}%")
            print(f"   Quality Score: {quality_summary.get('data_quality_score', 0)}/100")