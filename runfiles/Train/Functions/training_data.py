import csv

def training_data(filepath):
    pdf_data = dict()

    try:
        with open(filepath) as f:

            contents = csv.reader(f)

            for row in contents:
                if row[0] in pdf_data:
                    pdf_data[row[0]].append(int(row[1])-1)
                else:
                    pdf_data[row[0]] = list()
                    pdf_data[row[0]].append(int(row[1])-1)
                    
    except FileNotFoundError:
        print(f"File {filepath} not found.")
    
    return pdf_data 