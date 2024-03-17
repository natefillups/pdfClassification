# pdfClassification

How to run:

To train the data based on the `PdfFiles/Training` files and the `PdfFiles/training_data.csv`:
`python classification.py train`

To run the files in `PdfFiles/Process`:
`python classification.py run`

Training:

Training the model will take in the information of the invalid files from the `PdfFiles/training_data.csv` files in the format `file_name, page_number`. It will process the pre-classified files that are in `PdfFiles/Training` and output the results to `PdfFiles/training_metadata.csv`.

Classifying New PDFs:

To classify new PDFs the program will take the `PdfFiles/training_metadata.csv` file and run it through multiple algorithms. It will then take the one with the largest accuracy and run the model against PDF files in `PdfFiles/Process`. It will then output the invalid files to the `PdfFiles/invalid_files.csv`.