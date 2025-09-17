#!/usr/bin/env python3
"""
CSV Search Script - Filter CSV data by column criteria

Usage:
    python search_csv.py <csv_file> [column1=value1] [column2=value2] ...

Examples:
    python search_csv.py data.csv chosen_movement_type=same_finger
    python search_csv.py data.csv chosen_movement_type=same_row chosen_finger_coordination=adjacent_fingers
    python search_csv.py data.csv slider_value=">-20" chosen_selected=1
    python search_csv.py data.csv --count chosen_movement_type=same_finger
    python search_csv.py data.csv --columns  # List all available columns
    python search_csv.py data.csv --select user_id,chosen_bigram,slider_value chosen_selected=1
    python search_csv.py data.csv --table chosen_movement_type=same_finger  # Table format

    # This example compares bigrams in the output with different column separations but same row separations:
    poetry run python3 search_table.py output/nonProlific/analyze_objectives/comprehensive_bigram_diagnosis.csv --select chosen_bigram,unchosen_bigram,chosen_row_separation,unchosen_row_separation,chosen_col_separation,unchosen_col_separation chosen_col_separation=0 unchosen_col_separation=">0" in_col_separation_analysis=1    
"""

import pandas as pd
import sys
import argparse
import re
import io
from typing import List, Dict, Any


def parse_value(value_str: str) -> Any:
    """Parse a value string, handling numeric comparisons and data types."""
    # Handle comparison operators for numeric values
    if value_str.startswith(('>', '<', '>=', '<=', '!=')):
        return value_str
    
    # Try to convert to appropriate type
    if value_str.lower() in ['true', 'false']:
        return value_str.lower() == 'true'
    
    try:
        # Try integer first
        if '.' not in value_str:
            return int(value_str)
        else:
            return float(value_str)
    except ValueError:
        return value_str


def apply_filter(df: pd.DataFrame, column: str, value: Any) -> pd.DataFrame:
    """Apply a filter to the dataframe based on column and value."""
    if column not in df.columns:
        print(f"Warning: Column '{column}' not found in CSV. Available columns:")
        print(", ".join(df.columns.tolist()))
        return df
    
    # Handle comparison operators
    if isinstance(value, str) and any(value.startswith(op) for op in ['>', '<', '>=', '<=', '!=']):
        op = ''.join(c for c in value if c in '><!=')
        num_str = value[len(op):]
        
        try:
            num_value = float(num_str)
            if op == '>':
                return df[df[column] > num_value]
            elif op == '<':
                return df[df[column] < num_value]
            elif op == '>=':
                return df[df[column] >= num_value]
            elif op == '<=':
                return df[df[column] <= num_value]
            elif op == '!=':
                return df[df[column] != num_value]
        except ValueError:
            print(f"Error: Could not parse numeric value '{num_str}' for comparison")
            return df
    
    # Handle exact matches (case-insensitive for strings)
    if isinstance(value, str) and not isinstance(df[column].iloc[0], (int, float)):
        return df[df[column].str.lower() == value.lower()]
    else:
        return df[df[column] == value]


def main():
    parser = argparse.ArgumentParser(description='Search and filter CSV data by column criteria')
    parser.add_argument('csv_file', help='Path to the CSV file')
    parser.add_argument('criteria', nargs='*', help='Search criteria in format column=value')
    parser.add_argument('--count', '-c', action='store_true', help='Show only the count of matching rows')
    parser.add_argument('--columns', action='store_true', help='List all available columns and exit')
    parser.add_argument('--output', '-o', help='Save results to output file')
    parser.add_argument('--head', '-n', type=int, help='Show only first N results')
    parser.add_argument('--table', '-t', action='store_true', help='Use table format instead of CSV format')
    parser.add_argument('--select', '-s', help='Show only specific columns (comma-separated)')
    parser.add_argument('--debug', action='store_true', help='Show debug information')
    
    args = parser.parse_args()
    
    try:
        # Read the CSV file
        df = pd.read_csv(args.csv_file)
        print(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
        
        # If --columns flag is used, list columns and exit
        if args.columns:
            print("\nAvailable columns:")
            for i, col in enumerate(df.columns, 1):
                print(f"{i:2d}. {col}")
            return
        
        # Parse search criteria
        if not args.criteria:
            print("No search criteria provided. Showing first 5 rows:")
            if args.table:
                print(df.head().to_string(index=False))
            else:
                print(df.head().to_csv(index=False))
            return
        
        filters = {}
        for criterion in args.criteria:
            if '=' not in criterion:
                print(f"Error: Invalid criterion '{criterion}'. Use format column=value")
                continue
                
            column, value_str = criterion.split('=', 1)
            filters[column] = parse_value(value_str)
        
        # Apply filters
        filtered_df = df.copy()
        for column, value in filters.items():
            print(f"Filtering by {column} = {value}")
            filtered_df = apply_filter(filtered_df, column, value)
        
        print(f"Found {len(filtered_df)} matching rows")
        
        # Show results
        if args.count:
            print(f"Count: {len(filtered_df)}")
        else:
            if args.head:
                result_df = filtered_df.head(args.head)
            else:
                result_df = filtered_df
            
            # Select specific columns if requested
            if args.select:
                selected_cols = [col.strip() for col in args.select.split(',')]
                missing_cols = [col for col in selected_cols if col not in result_df.columns]
                if missing_cols:
                    print(f"Warning: Columns not found: {', '.join(missing_cols)}")
                available_cols = [col for col in selected_cols if col in result_df.columns]
                if available_cols:
                    result_df = result_df[available_cols]
                else:
                    print("Error: No valid columns selected")
                    return
            
            if len(result_df) > 0:
                # Debug output
                if args.debug:
                    print(f"Debug: Using {'table' if args.table else 'CSV'} format")
                    print(f"Debug: Result shape: {result_df.shape}")
                
                # Output format
                if args.table:
                    print("\nMatching rows (table format):")
                    print(result_df.to_string(index=False))
                else:
                    print("\nMatching rows (CSV format):")
                    # Use pandas to_csv for clean output
                    csv_output = result_df.to_csv(index=False)
                    print(csv_output.rstrip())
                
                # Save to file if requested
                if args.output:
                    result_df.to_csv(args.output, index=False)
                    print(f"\nResults saved to {args.output}")
            else:
                print("No matching rows found.")
    
    except FileNotFoundError:
        print(f"Error: File '{args.csv_file}' not found")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()