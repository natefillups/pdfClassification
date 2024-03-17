import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump

# Provide column names explicitly
column_names = ['fileName', 'pageNumber', 'edgeData', 'laplacianData', 'noiseData', 'pdfHeight', 'pdfWidth', 'tone', 'ttwo', 'tthree', 'tfour', 'tfive', 'tsix', 'tseven', 'teight', 'valid']

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('logs/trainingLogs.csv', names=column_names, header=None)

# Extract features and target variable
X = df[['edgeData', 'laplacianData', 'noiseData', 'pdfHeight', 'pdfWidth', 'tone', 'ttwo', 'tthree', 'tfour', 'tfive', 'tsix', 'tseven', 'teight']]
y = df['valid']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Support Vector Machine classifier
model = SVC()

# Train the model
model.fit(X_train, y_train)

# Save the trained model
dump(model, 'training/model2.joblib')

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

