import os
import io
import argparse
import csv

from Train.Functions import pdf_processing
from Run.Functions import run_model

PDF_DIRECTORY = os.path.join("PdfFiles", "Process")
OUTPUT_FILE = os.path.join("invalid_files.csv")
RAW_METADATA = os.path.join("PdfFiles", "Process", "process_metadata.csv")

def run(arguments):
    """
        Run takes the models, and the PDF files that are in the PdfFiles.Process directory, extracts the metadata, and then runs the metadata against the model that is input.    
    
        @input arguments: the number of the model that is to be run
    """
    
    # First extract the metadata from the files in the PdfFiles.Process directory
    metadata_list = pdf_processing.pdf_processing(PDF_DIRECTORY)

    with io.open(RAW_METADATA, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(metadata_list)

    # Run the metdata against the chosen model
    invalid_files = run_model.process_metadata(metadata_list, arguments)

    # Format the data and pipe the data to the OUTPUT_FILE and the console
    run_model.format_data(metadata_list, invalid_files, OUTPUT_FILE)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to run a model")
    parser.add_argument("argument", nargs="?", default=1, help="Model number to be run")
    args = parser.parse_args()

    run(args.argument)

    print(f"Invalid files sent to {OUTPUT_FILE}")