import pandas as pd
import sys
import os

def merge_csv_files_from_args(file_paths, output_filename="dataset.csv"):
    """
    Merges specified CSV files by path, expecting the defined structure.

    Args:
        file_paths (list): A list of paths to the CSV files to merge.
        output_filename (str): The name for the output merged CSV file.
                               Defaults to "dataset.csv".
    """
    # Define the expected fields based on your description
    expected_fields = ['bounce_frame', 'vx_before', 'vy_before', 'bounce_x', 'bounce_y', 'height_category']

    if not file_paths:
        print("No input files provided.")
        print(f"Usage: python {sys.argv[0]} file1.csv file2.csv [file3.csv ...]")
        return

    list_df = []

    print(f"Attempting to read and merge {len(file_paths)} files...")

    for filename in file_paths:
        if not os.path.exists(filename):
            print(f"Error: File not found: {filename}. Skipping.")
            continue
        if not filename.lower().endswith('.csv'):
             print(f"Warning: File {filename} does not appear to be a CSV. Skipping.")
             continue

        try:
            df = pd.read_csv(filename)

            # Optional: Add a check to see if the columns match the expected fields
            if not all(field in df.columns for field in expected_fields):
                 print(f"Warning: File {filename} does not contain all expected fields. Skipping.")
                 continue

            # Optional: Reorder columns to match the expected order
            # This prevents issues if columns are in a different order in different files
            df = df[expected_fields]

            list_df.append(df)
            print(f"Successfully read {filename}")
        except Exception as e:
            print(f"Error reading file {filename}: {e}")

    if not list_df:
        print("No valid CSV files were read. Merging cancelled.")
        return

    # Concatenate all dataframes in the list
    merged_df = pd.concat(list_df, ignore_index=True)

    # Save the merged dataframe to the output CSV file
    try:
        merged_df.to_csv(output_filename, index=False)
        print(f"\nSuccessfully merged {len(list_df)} files into {output_filename}")
        print(f"Merged dataframe shape: {merged_df.shape}")
    except Exception as e:
        print(f"Error writing merged file {output_filename}: {e}")

if __name__ == "__main__":
    # sys.argv[0] is the script name itself.
    # We want arguments starting from index 1.
    input_files = sys.argv[1:]

    merge_csv_files_from_args(input_files, output_filename="dataset.csv")
