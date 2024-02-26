import os
from PyPDF2 import PdfReader
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import fitz
import datetime

# Function to extract features from an image
# def extract_features(image):
#     features = []
#
#     # Color Histograms
#     hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
#     features.extend(hist.flatten())
#
#     # Convert image to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#     # Edge Detection (Canny edge detector)
#     edges = cv2.Canny(gray, 100, 200)
#     edge_density = np.sum(edges) / (gray.shape[0] * gray.shape[1])
#     features.append(edge_density)
#
#     # Corner Detection (Harris corner detector)
#     corners = cv2.cornerHarris(gray, 2, 3, 0.04)
#     num_corners = np.sum(corners > 0.01 * corners.max())
#     features.append(num_corners)
#
#     # Image Quality Metrics
#     laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
#     features.append(laplacian_var)
#
#     # Contrast
#     mean, std_dev = cv2.meanStdDev(gray)
#     contrast = std_dev / mean if mean != 0 else 0
#     features.append(contrast[0][0])
#
#     # Noise Level
#     noise_level = np.std(gray)
#     features.append(noise_level)
#
#     # Resolution (Width and Height)
#     resolution = image.shape[1], image.shape[0]
#     features.extend(resolution)
#
#     return features
def extract_features(image):
    features = []

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Edge Detection (Canny edge detector)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges) / (gray.shape[0] * gray.shape[1])
    features.append(edge_density)

    # Image Quality Metrics
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    features.append(laplacian_var)

    # Noise Level
    noise_level = np.std(gray)
    features.append(noise_level)

    # Resolution (Width and Height)
    resolution = image.shape[1], image.shape[0]
    features.extend(resolution)

    # Color Histograms (RGB color space)
    hist_r = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([image], [1], None, [256], [0, 256])
    hist_b = cv2.calcHist([image], [2], None, [256], [0, 256])
    features.extend(np.concatenate((hist_r, hist_g, hist_b)).flatten())

    # Texture Features (LBP)
    lbp = local_binary_pattern(gray, 8, 1, method='uniform')
    lbp_hist, _ = np.histogram(lbp, bins=np.arange(0, 10), density=True)
    features.extend(lbp_hist)

    # Shape Features (Number of contours)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_contours = len(contours)
    features.append(num_contours)

    return features


# Function to process a PDF document
def process_pdf(pdf_path, label):
    features_list = []
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap()
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))

        # Extract features from the image
        features = extract_features(img)
        features_list.append(features)
        log_file.write(f"Page {page_num + 1}: Features: {features}, Label: {label}\n")

    doc.close()
    return features_list, [label] * len(features_list)

# Function to process all PDF documents in a directory
def process_pdf_directory(directory):
    X = []
    y = []
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(directory, filename)
            features, labels = process_pdf(pdf_path, label=1)  # Assuming all pages are high quality
            X.extend(features)
            y.extend(labels)
    return X, y

# Process PDF documents in the specified directory
pdf_directory = '.'
log_file = 'logs/trainingLogs.csv'
with open(log_file, 'w') as log_file:
    X, y = process_pdf_directory(pdf_directory)


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = clf.predict(X_test)

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
