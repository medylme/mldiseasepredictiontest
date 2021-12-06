import numpy as np
import pandas as pd
from scipy.stats import mode
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

TRAININGDATA_PATH = "dataset/Training.csv"
data = pd.read_csv(TRAININGDATA_PATH).dropna(axis = 1)

encoder = LabelEncoder()
data["prognosis"] = encoder.fit_transform(data["prognosis"])

disease_counts = data["prognosis"].value_counts()
temp_df = pd.DataFrame({
    "Disease": disease_counts.index,
    "Counts": disease_counts.values
})

# ----------------------------------------------------------------

# Splitting the data into 80:20 format
# i.e. 80% of the dataset will be used for training the model
# and 20% of the data will be used to evaluate the performance of the models.
X = data.iloc[:,:-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test =train_test_split(
  X, y, test_size = 0.2, random_state = 24)

# Printing results
print(f"Train: {X_train.shape}, {y_train.shape}")
print(f"Test: {X_test.shape}, {y_test.shape}")
