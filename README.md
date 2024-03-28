# pdfClassification

Problem Statement:

There are many documents being scanned at my company and there is no automized way to do quality-control besides manual review. To automate this process I have created a machine learning model to take metadata from the manually reviewed documents and then automatically process the unprocessed documents by running it through the machine learning model.

Local Setup:

1. Create a virtualenv for Python 3.12.1.
2. Run the command `pip install -r requirements.txt`
3. Add the desired PDF files and run

How to run:

To train the data based on the `PdfFiles/Training` files and the `PdfFiles/training_data.csv`:
Run the command `python classification.py train`.

To run the files in `PdfFiles/Process`:
Run the command `python classification.py run <model-number>`.

After the `run` command is executed, the files in the `PdfFiles/Process` directory have been processed and the metadata is stored. To run the `run` command without reprocessing the documents use the command `python classification.py run-existing <model-number>`

Training:

Training the model will take in the information of the invalid files from the `PdfFiles/Training/training_data.csv` files in the format `file_name, page_number`. It will process the pre-classified files that are in `PdfFiles/Training` and output the metadata results to `PdfFiles/Training/training_metadata.csv`. The training will then run the metadata through five different algorithms and display the results for each algorithm in the console.

Classifying New PDFs:

To classify new PDFs the program will take the models in `runfiles/Train/Models` directory and run the chosen model by the users arguments. It will then take the model and run it against the PDF files metadata generated from the PDF files in `PdfFiles/Process`. It will then output the invalid files to the `invalid_files.csv` file.

If the metadata is already generated for the `PdfFiles/Process` files (or if the `run` command has already been executed) the `run-existing` command can be used. The `run-existing` command is similar to the `run` command except that it will use the already generated metadata from the `run` command and process the data. 
