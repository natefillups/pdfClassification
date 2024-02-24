import csv

# Open the CSV file
with open('logs.csv', newline='') as csvfile:
    # Create a CSV reader object
    reader = csv.reader(csvfile)
    
    # Iterate over each row in the CSV file
    for row in reader:
        # Extract data from each row
        fileName = row[0]
        pageNumber = int(row[1])
        edgeData = float(row[2])
        laplacianData = float(row[3])
        noiseData = float(row[4])
        pdfHeight = int(row[5])
        pdfWidth = int(row[6])
        valid = bool(int(row[7]))  # Convert '1' or '0' to True or False
        
        # Print the extracted data
        print("File Name:", fileName)
        print("Page Number:", pageNumber)
        print("Edge Data:", edgeData)
        print("Laplacian Data:", laplacianData)
        print("Noise Data:", noiseData)
        print("PDF Height:", pdfHeight)
        print("PDF Width:", pdfWidth)
        print("Valid:", valid)
        print()  # Add an empty line for clarity
