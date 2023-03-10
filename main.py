# Loader
import os
import numpy as np
import pandas as pd
from scipy.stats import mode
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import itertools
import threading
import time
import sys


def animate():
    for c in itertools.cycle(['|', '/', '-', '\\']):
        if not loading:
            break
        sys.stdout.write('\rLoading... ' + c)
        sys.stdout.flush()
        time.sleep(0.1)


loading = True
t = threading.Thread(target=animate)
t.start()


class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def clearConsole():
    command = 'clear'
    if os.name in ('nt', 'dos'):
        command = 'cls'
    os.system(command)


def predictDisease(symptoms):
    try:
        symptoms = symptoms.title().split(",")

        # Reshape the input data and convert it into a suitable format for the model predictions
        input_data = [0] * len(data_dict["symptom_index"])
        for symptom in symptoms:
            index = data_dict["symptom_index"][symptom]
            input_data[index] = 1
        input_data = np.array(input_data).reshape(1, -1)

        rf_prediction = data_dict["predictions_classes"][final_rf_model.predict(input_data)[
            0]]
        nb_prediction = data_dict["predictions_classes"][final_nb_model.predict(input_data)[
            0]]
        svm_prediction = data_dict["predictions_classes"][final_svm_model.predict(input_data)[
            0]]
        final_prediction = mode(
            [rf_prediction, nb_prediction, svm_prediction])[0][0]

        predictions = {
            "Prediction #1 (SVC)": rf_prediction,
            "Prediction #2 (Gaussian NB)": nb_prediction,
            "Prediction #3 (Random Forest)": nb_prediction,
            "Mode": final_prediction
        }
        return predictions
    except KeyError as e:
        print(
            f"Invalid input: {e}. Please enter valid symptoms separated by commas.")
    except Exception as e:
        print(f"An error occurred: {e}. Please try again.")


# Read training dataset
TRAININGDATA_PATH = "dataset/Training.csv"
data = pd.read_csv(TRAININGDATA_PATH).dropna(axis=1)
encoder = LabelEncoder()
data["prognosis"] = encoder.fit_transform(data["prognosis"])
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Initialize and train models
final_svm_model = SVC()
final_nb_model = GaussianNB()
final_rf_model = RandomForestClassifier(random_state=18)
final_svm_model.fit(X.values, y)
final_nb_model.fit(X.values, y)
final_rf_model.fit(X.values, y)

# Create a symptom index dictionary to encode the input symptoms into a numerical form
symptoms = X.columns.values
symptom_index = {}
for index, value in enumerate(symptoms):
    symptom = " ".join([i.capitalize() for i in value.split("_")])
    symptom_index[symptom] = index
data_dict = {
    "symptom_index": symptom_index,
    "predictions_classes": encoder.classes_
}

# Stop the loader
loading = False
clearConsole()

print("==============================================================================================================")
print(color.BOLD + "MACHINE LEARNING DISEASE PREDICTION TEST\n" + color.END)
print("Input symptoms seperated by a comma i.e. 'Itching,Skin Rash,Dischromic Patches'")
print("For a list of symptoms, see README.md")
print("==============================================================================================================")

input = input("-> ")
try:
    if ", " in input:
        raise IndentationError()
    output = predictDisease(input)
    print(color.RED + '\n'.join("{}: {}".format(k, v)
          for k, v in output.items()))
except IndentationError:
    print("Error: Please seperate the symptoms by just a comma (no spaces).")
except:
    print("Error. The program will close.")
finally:
    print(color.END + "==============================================================================================================")
