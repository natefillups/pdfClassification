import pandas as pd
from joblib import load

# Load the trained model
model = load('./training/model.joblib')

# Read the CSV file into a pandas DataFrame
column_names = ['fileName', 'pageNumber', 'edgeData', 'laplacianData', 'noiseData', 'pdfHeight', 'pdfWidth', 'valid']
df = pd.read_csv('./logs/pdfLogs.csv', names=column_names, header=None)

# Extract features
X = df[['edgeData', 'laplacianData', 'noiseData', 'pdfHeight', 'pdfWidth']]

# Predict using the loaded model
predictions = model.predict(X)

# Output whether each file is valid or not
for file, pageNumber, valid in zip(df['fileName'], df['pageNumber'], predictions):
    if(not valid):
        print(f"{file} Page {pageNumber}")
