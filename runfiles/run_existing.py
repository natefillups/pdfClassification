import os
import io
import csv
import argparse

from Run.Functions import run_model 

RAW_METADATA = os.path.join("PdfFiles", "Process", "process_metadata.csv")
OUTPUT_FILE = os.path.join("invalid_files.csv")

def run_existing(arguments):
    """
        Run the models against the existing metadata generated by the Run command.
        Takes the metadata and creates the invalid files based on the model number chosen.

        @input arguments: the number of the model that is to be run
    """

    metadata_list = []

    with io.open(RAW_METADATA) as f:
        reader = csv.reader(f)
        for row in reader:
            row  = [int(val) if val.isdigit() else float(val) if val.replace('.','',1).isdigit() else val for val in row]
            metadata_list.append(row)
    
    # Run the metdata against the chosen model
    invalid_files = run_model.process_metadata(metadata_list, arguments)

    # Format the data and pipe the data to the OUTPUT_FILE and the console
    run_model.format_data(metadata_list, invalid_files, OUTPUT_FILE)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to run a model based on existing data")
    parser.add_argument("argument", nargs="?", default=1, help="Model number to be run")
    args = parser.parse_args()

    run_existing(args.argument)

    print(f"Invalid files sent to {OUTPUT_FILE}")