import pandas as pd
from joblib import load

# Load the trained model
model = load('random_forest_model.joblib')

# Read the CSV file into a pandas DataFrame
column_names = ['fileName', 'pageNumber', 'edgeData', 'laplacianData', 'noiseData', 'pdfHeight', 'pdfWidth', 'valid']
df = pd.read_csv('pdfLogs.csv', names=column_names, header=None)

# Extract features
X = df[['pageNumber', 'edgeData', 'laplacianData', 'noiseData', 'pdfHeight', 'pdfWidth']]

# Predict using the loaded model
predictions = model.predict(X)

# Output whether each file is valid or not
for file, valid in zip(df['fileName'], predictions):
    print(f"{file}: {'Valid' if valid else 'Invalid'}")
