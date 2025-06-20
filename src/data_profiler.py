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
        return {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'memory_usage_mb': round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
            'duplicate_rows': df.duplicated().sum()
        }
    
    def _profile_columns(self, df):
        """Profile each column in the dataset."""
        column_profiles = {}
        
        for column in df.columns:
            column_profiles[column] = self._profile_single_column(df[column])
        
        return column_profiles
    
    def _profile_single_column(self, series):
        """Profile a single column/series."""
        profile = {
            'data_type': str(series.dtype),
            'null_count': series.isnull().sum(),
            'null_percentage': round((series.isnull().sum() / len(series)) * 100, 2),
            'unique_count': series.nunique(),
            'unique_percentage': round((series.nunique() / len(series)) * 100, 2)
        }
        
        # Add type-specific statistics
        if pd.api.types.is_numeric_dtype(series):
            profile.update(self._get_numeric_stats(series))
        elif pd.api.types.is_datetime64_any_dtype(series):
            profile.update(self._get_datetime_stats(series))
        else:
            profile.update(self._get_text_stats(series))
        
        return profile
    
    def _get_numeric_stats(self, series):
        """Get statistics for numeric columns."""
        return {
            'min_value': series.min() if not series.empty else None,
            'max_value': series.max() if not series.empty else None,
            'mean': round(series.mean(), 2) if not series.empty else None,
            'median': series.median() if not series.empty else None,
            'std_deviation': round(series.std(), 2) if not series.empty else None,
            'zero_count': (series == 0).sum(),
            'negative_count': (series < 0).sum()
        }
    
    def _get_datetime_stats(self, series):
        """Get statistics for datetime columns."""
        return {
            'min_date': series.min() if not series.empty else None,
            'max_date': series.max() if not series.empty else None,
            'date_range_days': (series.max() - series.min()).days if not series.empty else None
        }
    
    def _get_text_stats(self, series):
        """Get statistics for text columns."""
        non_null_series = series.dropna()
        if len(non_null_series) == 0:
            return {
                'min_length': None,
                'max_length': None,
                'avg_length': None,
                'empty_string_count': 0
            }
        
        lengths = non_null_series.astype(str).str.len()
        return {
            'min_length': lengths.min(),
            'max_length': lengths.max(),
            'avg_length': round(lengths.mean(), 2),
            'empty_string_count': (non_null_series == '').sum()
        }
    
    def _get_quality_summary(self, df):
        """Get overall data quality summary."""
        total_cells = df.shape[0] * df.shape[1]
        null_cells = df.isnull().sum().sum()
        
        return {
            'overall_completeness': round(((total_cells - null_cells) / total_cells) * 100, 2),
            'columns_with_nulls': df.isnull().any().sum(),
            'completely_null_columns': df.isnull().all().sum(),
            'data_quality_score': self._calculate_quality_score(df)
        }
    
    def _calculate_quality_score(self, df):
        """Calculate a simple data quality score (0-100)."""
        # This is a simplified scoring system
        completeness = (1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        uniqueness = (df.nunique().sum() / df.shape[0]) * 10  # Simplified uniqueness factor
        
        # Weighted score (completeness has higher weight)
        score = (completeness * 0.7) + (min(uniqueness, 30) * 0.3)
        return round(min(score, 100), 2)
    
    def export_profile(self, output_path, format='json'):
        """Export profiling results to file."""
        if format.lower() == 'json':
            with open(output_path, 'w') as f:
                json.dump(self.profile_results, f, indent=2, default=str)
        elif format.lower() == 'csv':
            # Flatten the profile for CSV export
            flattened_data = []
            for dataset_name, profile in self.profile_results.items():
                for column_name, column_profile in profile['columns'].items():
                    row = {
                        'dataset_name': dataset_name,
                        'column_name': column_name,
                        **column_profile
                    }
                    flattened_data.append(row)
            
            df = pd.DataFrame(flattened_data)
            df.to_csv(output_path, index=False)
        
        print(f"Profile exported to {output_path}")