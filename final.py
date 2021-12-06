# Imports
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

# Reading the training dataset
TRAININGDATA_PATH = "dataset/Training.csv"
data = pd.read_csv(TRAININGDATA_PATH).dropna(axis = 1)
encoder = LabelEncoder()
data["prognosis"] = encoder.fit_transform(data["prognosis"])
X = data.iloc[:,:-1]
y = data.iloc[:, -1]

# Initializing and training the models
final_svm_model = SVC()
final_nb_model = GaussianNB()
final_rf_model = RandomForestClassifier(random_state=18)
final_svm_model.fit(X.values, y)
final_nb_model.fit(X.values, y)
final_rf_model.fit(X.values, y)

# Creating a symptom index dictionary to encode the
# input symptoms into a numerical form
symptoms = X.columns.values
symptom_index = {}
for index, value in enumerate(symptoms):
	symptom = " ".join([i.capitalize() for i in value.split("_")])
	symptom_index[symptom] = index
data_dict = {
	"symptom_index":symptom_index,
	"predictions_classes":encoder.classes_
}

# The actual function
# Takes a string containing symptoms, separated by commmas
# Outputs the generated predictions
def predictDisease(symptoms):
	# Splitting the input data, which is seperated by commas
    symptoms = symptoms.title().split(",")

	# Reshaping the input data and converting it
    # into a suitable format for the model predictions
    input_data = [0] * len(data_dict["symptom_index"])
    for symptom in symptoms:
        index = data_dict["symptom_index"][symptom]
        input_data[index] = 1
    input_data = np.array(input_data).reshape(1,-1)

    # Generating the individual predictions
    rf_prediction = data_dict["predictions_classes"][final_rf_model.predict(input_data)[0]]
    nb_prediction = data_dict["predictions_classes"][final_nb_model.predict(input_data)[0]]
    svm_prediction = data_dict["predictions_classes"][final_svm_model.predict(input_data)[0]]

    # Taking the mode of all predictions
    final_prediction = mode([rf_prediction, nb_prediction, svm_prediction])[0][0]

	# Returning the results as a dictionary
    predictions = {
        "Prediction #1 (SVC)": rf_prediction,
        "Prediction #2 (Gaussian NB)": nb_prediction,
        "Prediction #3 (Random Forest)": nb_prediction,
        "Final Prediction":final_prediction
    }
    return predictions


print("==============================================================================================================")
print("DISEASE PREDICTION")
print("Input symptoms seperated by a comma i.e. 'Itching,Skin Rash,Nodal Skin Eruptions,Dischromic Patches'")
print("For a list of symptoms, see README.md")
print("==============================================================================================================")
input = input("-> ")
output = predictDisease(input)

print('\n')
print('\n'.join("{}: {}".format(k, v) for k, v in output.items()))
print("==============================================================================================================")
