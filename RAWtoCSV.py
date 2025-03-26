import wfdb
import pandas as pd
import os

def convert_mitbih_dataset(data_dir, output_dir):
    """
    Converts all MIT-BIH records in a directory into CSV files.

    Parameters:
    - data_dir: Directory containing MIT-BIH `.dat`, `.hea`, and `.atr` files.
    - output_dir: Directory where CSV files will be saved.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Identify all records in the directory (excluding file extensions)
    records = sorted(set(f.split('.')[0] for f in os.listdir(data_dir) if f.endswith('.dat')))

    if not records:
        print("No MIT-BIH records found in the directory!")
        return

    print(f"Processing {len(records)} records...")

    for record_name in records:
        try:
            print(f"Processing record: {record_name}...")

            # Load the signal and annotation data
            record_path = os.path.join(data_dir, record_name)
            record = wfdb.rdrecord(record_path)  # Reads .dat and .hea
            annotation = wfdb.rdann(record_path, 'atr')  # Reads .atr file

            # Extract ECG signal data
            signals = record.p_signal
            df = pd.DataFrame(signals, columns=record.sig_name)

            # Add time column
            df['Time'] = [i / record.fs for i in range(len(df))]  # Convert samples to seconds

            # Add annotations at corresponding timestamps
            df['Annotation'] = ''
            for i, idx in enumerate(annotation.sample):
                if idx < len(df):
                    df.at[idx, 'Annotation'] = annotation.symbol[i]

            # Save CSV file
            output_csv = os.path.join(output_dir, f"{record_name}.csv")
            df.to_csv(output_csv, index=False)
            print(f"Saved: {output_csv}")

        except Exception as e:
            print(f"Error processing {record_name}: {e}")

# Example usage
data_directory = "./mit-bih-arrhythmia-database-1.0.0"  # Path to your MIT-BIH dataset folder
output_directory = "./csv_output"  # Folder to save CSV files

convert_mitbih_dataset(data_directory, output_directory)

