import os
import pandas as pd
from collections import Counter
import argparse

def analyze_browsing_mode_distribution(data_dir):
    mode_counts = Counter()
    file_counts = {1: [], 2: [], 3: []}
    
    for file in os.listdir(data_dir):
        if file.endswith('.csv') and not file.startswith('.'):
            file_path = os.path.join(data_dir, file)
            try:
                df = pd.read_csv(file_path)
                if 'browsing_mode' in df.columns:
                    mode = df['browsing_mode'].mode()[0]
                    mode_counts[mode] += 1
                    file_counts[mode].append(file)
            except Exception as e:
                print(f"Error reading file: {file}, {e}")
    
    print("\nNumber of files per browsing_mode:")
    print(f"browsing_mode=1 (Watching Videos): {mode_counts[1]} files")
    print(f"browsing_mode=2 (Browsing Infinite Feed): {mode_counts[2]} files")
    print(f"browsing_mode=3 (Other Activities): {mode_counts[3]} files")
    
    print("\nList of files for each browsing_mode:")
    for mode, files in file_counts.items():
        print(f"\nFiles with browsing_mode={mode}:")
        for file in files:
            print(f"  - {file}")

def main():
    # Get all CSV files
    mode_counts = {}
    
    # Iterate through all CSV files
    for file in os.listdir(args.data_dir):
        if file.endswith('.csv') and not file.startswith('.'):
            file_path = os.path.join(args.data_dir, file)
            try:
                # Read CSV file
                df = pd.read_csv(file_path)
                if 'browsing_mode' in df.columns:
                    # Get browsing_mode, skip if it doesn't exist
                    mode = df['browsing_mode'].mode()[0]
                    mode_counts[mode] = mode_counts.get(mode, 0) + 1
            except Exception as e:
                print(f"Error reading file: {file}, {e}")
    
    # Print distribution results
    print("Browsing Mode distribution:")
    print(f"Mode 1 (Watching Videos): {mode_counts.get(1, 0)} files")
    print(f"Mode 2 (Browsing Infinite Feed): {mode_counts.get(2, 0)} files")
    print(f"Mode 3 (Other Activities): {mode_counts.get(3, 0)} files")
    print(f"Other modes: {sum(count for mode, count in mode_counts.items() if mode not in [1, 2, 3])} files")
    
    print(f"\nTotal: {sum(mode_counts.values())} files")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check browsing_mode distribution")
    parser.add_argument("--data_dir", type=str, required=True, help="Data directory path")
    args = parser.parse_args()
    main() 