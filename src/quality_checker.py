import pandas as pd
import yaml
import re
import json
from datetime import datetime

class QualityChecker:
    """
    A class to validate data quality based on predefined rules.
    This ensures data meets business requirements and maintains consistency.
    """
    
    def __init__(self, rules_config_path):
        """
        Initialize the quality checker with rules configuration.
        
        Args:
            rules_config_path (str): Path to the quality rules YAML file
        """
        self.rules_config = self._load_rules(rules_config_path)
        self.validation_results = {}
    
    def _load_rules(self, config_path):
        """Load quality rules from YAML configuration file."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            print(f"Error loading rules configuration: {e}")
            return {}
    
    def validate_dataset(self, df, dataset_name, reference_data=None):
        """
        Validate a dataset against its quality rules.
        
        Args:
            df (pd.DataFrame): Dataset to validate
            dataset_name (str): Name of the dataset
            reference_data (dict): Reference datasets for integrity checks
            
        Returns:
            dict: Validation results
        """
        if dataset_name not in self.rules_config.get('datasets', {}):
            print(f"No rules found for dataset: {dataset_name}")
            return {}
        
        dataset_rules = self.rules_config['datasets'][dataset_name]
        validation_result = {
            'dataset_name': dataset_name,
            'validation_timestamp': datetime.now().isoformat(),
            'total_rows': len(df),
            'column_validations': {},
            'overall_status': 'PASS',
            'total_issues': 0
        }
        
        # Validate each column
        for column_name, column_rules in dataset_rules.get('columns', {}).items():
            if column_name in df.columns:
                column_result = self._validate_column(
                    df[column_name], column_name, column_rules, reference_data
                )
                validation_result['column_validations'][column_name] = column_result
                
                if column_result['status'] == 'FAIL':
                    validation_result['overall_status'] = 'FAIL'
                
                validation_result['total_issues'] += len(column_result['failed_rules'])
            else:
                # Column doesn't exist in dataset
                validation_result['column_validations'][column_name] = {
                    'status': 'FAIL',
                    'error': f"Column '{column_name}' not found in dataset",
                    'failed_rules': [{'rule': 'column_exists', 'message': f"Column '{column_name}' missing"}]
                }
                validation_result['overall_status'] = 'FAIL'
                validation_result['total_issues'] += 1
        
        self.validation_results[dataset_name] = validation_result
        return validation_result
    
    def _validate_column(self, series, column_name, rules, reference_data):
        """Validate a single column against its rules."""
        result = {
            'column_name': column_name,
            'status': 'PASS',
            'passed_rules': [],
            'failed_rules': []
        }
        
        for rule in rules:
            rule_result = self._apply_rule(series, rule, reference_data)
            
            if rule_result['passed']:
                result['passed_rules'].append(rule_result)
            else:
                result['failed_rules'].append(rule_result)
                result['status'] = 'FAIL'
        
        return result
    
    def _apply_rule(self, series, rule, reference_data):
        """Apply a single validation rule to a series."""
        rule_type = rule['rule_type']
        description = rule.get('description', f'{rule_type} validation')
        
        try:
            if rule_type == 'not_null':
                null_count = series.isnull().sum()
                passed = null_count == 0
                message = f"Found {null_count} null values" if not passed else "No null values found"
            
            elif rule_type == 'unique':
                duplicate_count = series.duplicated().sum()
                passed = duplicate_count == 0
                message = f"Found {duplicate_count} duplicate values" if not passed else "All values are unique"
            
            elif rule_type == 'data_type':
                expected_type = rule['expected_type']
                actual_type = str(series.dtype)
                passed = actual_type == expected_type
                message = f"Expected {expected_type}, got {actual_type}" if not passed else f"Data type is correct ({actual_type})"
            
            elif rule_type == 'range':
                min_val = rule.get('min_value')
                max_val = rule.get('max_value')
                
                violations = 0
                if min_val is not None:
                    violations += (series < min_val).sum()
                if max_val is not None:
                    violations += (series > max_val).sum()
                
                passed = violations == 0
                range_desc = f"[{min_val}, {max_val}]" if min_val is not None and max_val is not None else f">= {min_val}" if min_val is not None else f"<= {max_val}"
                message = f"Found {violations} values outside range {range_desc}" if not passed else f"All values within range {range_desc}"
            
            elif rule_type == 'regex':
                pattern = rule['pattern']
                non_null_series = series.dropna()
                if len(non_null_series) == 0:
                    passed = True
                    message = "No values to validate (all null)"
                else:
                    matches = non_null_series.astype(str).str.match(pattern)
                    violations = (~matches).sum()
                    passed = violations == 0
                    message = f"Found {violations} values not matching pattern" if not passed else "All values match pattern"
            
            elif rule_type == 'min_length':
                min_length = rule['min_value']
                non_null_series = series.dropna()
                if len(non_null_series) == 0:
                    passed = True
                    message = "No values to validate (all null)"
                else:
                    violations = (non_null_series.astype(str).str.len() < min_length).sum()
                    passed = violations == 0
                    message = f"Found {violations} values shorter than {min_length} characters" if not passed else f"All values meet minimum length of {min_length}"
            
            elif rule_type == 'reference_integrity':
                if reference_data is None:
                    passed = False
                    message = "Reference data not provided for integrity check"
                else:
                    ref_table = rule['reference_table']
                    ref_column = rule['reference_column']
                    
                    if ref_table in reference_data:
                        ref_values = reference_data[ref_table][ref_column].values
                        violations = (~series.isin(ref_values)).sum()
                        passed = violations == 0
                        message = f"Found {violations} values not in reference table" if not passed else "All values exist in reference table"
                    else:
                        passed = False
                        message = f"Reference table '{ref_table}' not found"
            
            else:
                passed = False
                message = f"Unknown rule type: {rule_type}"
        
        except Exception as e:
            passed = False
            message = f"Error applying rule: {str(e)}"
        
        return {
            'rule': rule_type,
            'description': description,
            'passed': passed,
            'message': message
        }
    
    def generate_quality_report(self):
        """Generate a comprehensive quality report."""
        report = {
            'report_timestamp': datetime.now().isoformat(),
            'summary': {
                'total_datasets': len(self.validation_results),
                'passed_datasets': sum(1 for r in self.validation_results.values() if r['overall_status'] == 'PASS'),
                'failed_datasets': sum(1 for r in self.validation_results.values() if r['overall_status'] == 'FAIL'),
                'total_issues': sum(r['total_issues'] for r in self.validation_results.values())
            },
            'dataset_results': self.validation_results
        }
        
        return report
    
    def export_quality_report(self, output_path):
        """Export quality report to JSON file."""
        report = self.generate_quality_report()
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Quality report exported to {output_path}")
        return report