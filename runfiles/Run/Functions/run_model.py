import os
import pandas
import csv

from collections import defaultdict
from joblib import load

def process_metadata(metadata_list, argument):
    """
        Process the metadata from the PDF files and runs the chosen model

        @input metadata_list[]: metadata from each page of the PDF file
        @input argument: the chosen model to be run
        @input RAW_DATA_OUTPUT: where the raw metadata will be stored for easier future processing
    """

    # Load the model of choice
    filepath = os.path.join("runfiles", "Train", "Models", f"model{argument}.pkl")
    
    model = load(filepath)

    # Take the metadata besides the header
    x = [[row[i] for i in range(2, 6)] for row in metadata_list[1:]]

    # Get the predictions based on the model
    predictions = model.predict(x)

    return predictions 
    
def format_data(metadata_list, invalid_files, OUTPUT_FILE):
    """
        Takes the metadata list and invalid file list and formats them in a {filename},{page_numbers} format.

        @input metadata_list[]: PDF metadata
        @input invalid_files[]: File predictions based on the model
        @input OUTPUT_FILE: The file where the data will be written
    """

    page_dict = defaultdict(set)

    for metadata, valid in zip(metadata_list[1:], invalid_files):
        if not valid: page_dict[metadata[0]].add(metadata[1])

    for key in page_dict:
        page_dict[key] = {value + 1 for value in page_dict[key]}

    with open(OUTPUT_FILE, 'w') as f:
        f.write(f"{metadata_list[0][0].ljust(40, ' ')}{metadata_list[0][1]}\n")
        for key in page_dict:
            f.write(f"{key.ljust(40, ' ')}{page_dict[key]}\n")