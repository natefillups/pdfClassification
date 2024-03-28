import os
import cv2
import fitz
import numpy

HEADERS = ["Pdf File", "Page Number", "Image Quality", "Edge Detection", "Noise Level", "Aspect Ratio", "Invalid"]

def extract_metadata(image):
    """
        Extract metadata from the image

        @input image: the image of the file to be analyzed
        @output metadata: the metadata extracted from the image
    """

    metadata = list()

    # Convert image to grayscale
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Image quality metrics
    laplacian_var = cv2.Laplacian(grey, cv2.CV_64F).var()
    metadata.append(laplacian_var)

    # Edge detection metrics
    edges = cv2.Canny(grey, 100, 200)
    edge_density_var = numpy.sum(edges) / (grey.shape[0] * grey.shape[1])
    metadata.append(edge_density_var)

    # Noise level metrics
    noise_var = numpy.std(grey)
    metadata.append(noise_var)

    # Aspect ratio 
    aspect_ratio = image.shape[1] / image.shape[0]
    metadata.append(aspect_ratio)

    return metadata

def handle_pdf(filename, filepath, invalid_pages, training_flag):
    """
        Processes the file and returns the metadata

        @input filename: the PDF filename
        @input filepath: the filepath of the PDF file
        @input invalid_pages[]: the invalid pages of the PDF file
        @output metadata[]: a list of metadata from the PDF file
    """

    metadata_list = []

    pdf_file = fitz.open(filepath)

    # Loops through each page in the pdf_file
    for page_number in range(len(pdf_file)):
        page = pdf_file.load_page(page_number)
        
        # Convert the PDF page to an image for easier processing
        pixels = page.get_pixmap()
        image = numpy.frombuffer(pixels.samples, dtype=numpy.uint8).reshape((pixels.height, pixels.width, pixels.n))

        metadata = list()
        metadata.append(filename)
        metadata.append(page_number)

        # Extract metadata from the image
        extracted_metadata = extract_metadata(image)

        for item in extracted_metadata:
            metadata.append(item)

        # Add if the page is classified as invalid
        if(training_flag):
            if page_number in invalid_pages:
                metadata.append(1)
            else:
                metadata.append(0)

        # Append metadata to total list
        metadata_list.append(metadata[:])

    return metadata_list

def pdf_processing(directory, pdf_data=list(), training_flag=0):
    """ 
        Processed all the PDFs in the directory

        @input directory: the directory where the PDF files are stored
        @input pdf_data: the classified pdf data
        @output metadata_list: the metadata from each of the PDF files in {directory}
    """

    metadata_list = []
    metadata_list.append(HEADERS)

    for filename in os.listdir(directory):
        # Loop through each PDF file in the given directory
        if filename.endswith('.pdf') or filename.endswith('.pdf'):
            filepath = os.path.join(directory, filename)

            # Retrieves the invalid pages for the PDF file
            invalid_pages = []

            if(training_flag):
                if filename in pdf_data:
                    invalid_pages = pdf_data[filename]

            # Gets the metadata from the file and appends it to the list
            metadata = handle_pdf(filename, filepath, invalid_pages, training_flag)

            for list in metadata:
                metadata_list.append(list)

    return metadata_list