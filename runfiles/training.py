import os

from Train.Functions.training_data import training_data
from Train.Functions.pdf_processing import pdf_processing
from Train.Functions.log_creation import write_logs
from Train.Functions.model_creation import generate_models

TRAINING_DATA_FILEPATH = os.path.join("PdfFiles", "Training", "training_data.csv")
PDF_DIRECTORY = os.path.join("PdfFiles", "Training")

if __name__ == "__main__":
    # Get the invalid file data from the classified csv file
    invalid_pdf_data = training_data(TRAINING_DATA_FILEPATH)
    
    # Process the PDF directory and process each file
    metadata_list = pdf_processing(PDF_DIRECTORY, invalid_pdf_data, 1)

    # Write to logs
    write_logs(metadata_list)

    # Generate models
    generate_models(metadata_list)