import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier
import joblib

MODEL_DIRECTORY = os.path.join("runfiles", "Train", "Models")

model_list = [
    RandomForestClassifier(),
    SVC(),
    GaussianNB(),
    MLPClassifier(),
    XGBClassifier()
]

model_list_names = [
    "RandomForestClassifier",
    "SVC",
    "GaussianNB",
    "MLPClassifier",
    "XGBClassifier"
]

def generate_model(model_number, model, x_train, x_test, y_train, y_test):
    """
        Generate ML models based on metadata and given model.

        @input metadata[]: List of metadata from PDF files
        @input model: Model function to be used
    """

    model.fit(x_train, y_train)

    filepath = os.path.join(MODEL_DIRECTORY, f"model{model_number+1}.pkl")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)

    y_pred = model.predict(x_test)

    print(f"Accuracy: {accuracy_score(y_test, y_pred)}, for Model: {model_number+1}, Model Type: {str(model_list_names[model_number])}")

def generate_models(metadata):
    """
        Generate ML models based on metadata.

        @input metadata[]: List of metadata from PDF files
    """

    x = [[row[i] for i in range(2, 6)] for row in metadata[1:]]
    y = [row[6] for row in metadata[1:]]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    for model_number, model in enumerate(model_list):
        generate_model(model_number, model, x_train, x_test, y_train, y_test)