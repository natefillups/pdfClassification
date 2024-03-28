import os
import csv

METADATA_FILEPATH = os.path.join("PdfFiles", "Training", "training_metadata.csv")


def write_logs(metadata_list):
    try:
        with open(METADATA_FILEPATH, 'w', newline='') as f:
            writer = csv.writer(f)

            for row in metadata_list:
                writer.writerow(row)

        print(f"Data successfully written to {METADATA_FILEPATH}.")
    except Exception as e:
        print(f"Error writing to file {METADATA_FILEPATH}: {e}.")