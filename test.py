import yaml
import numpy as np
from dataset import *
from models import *
from visualize import *
from utils import *
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models

with open(os.path.join('config.yaml')) as stream:
    config = yaml.safe_load(stream)

if(preprocess_required(config["dataset"])):
    preprocess_data(config["dataset"])

labels = config["dataset"]["labels"]
n_classes = len(labels)
model_name = config["train"]["model_name"]
output_path = config["output_path"]
saved_models_path = config["saved_models_path"]

X_test, y_test = load_test_data(config["dataset"])

X_test = X_test.astype('float32')
X_test /= 255

path = saved_models_path + '\\' + model_name + '.h5'
model = models.load_model(path)

incorrect = 0

for X, y in zip(X_test, y_test):
    X = np.reshape(X, (1, X.shape[0], X.shape[1], X.shape[2]))
    y_pred = model.predict(X)
    y_pred = np.argmax(y_pred, axis=1)
    y_pred = y_pred[0]
    if y_pred != y:
        incorrect += 1
print('Number of incorrect predictions: ' + str(incorrect))
