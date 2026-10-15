import pandas as pd
from sklearn.metrics import accuracy_score
import pickle
import logging
import mlflow
import os
logger=logging.getLogger("Model_Training_Logs")
logger.setLevel(logging.INFO)
file_handler=logging.FileHandler('Logs/model_evaluation.log')
file_handler.setLevel(logging.INFO)
formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
model_path="model.pkl"
x_test_path="X_test.csv"
y_test_path="y_test.csv"

def load_model(model_path):
    logger.info("Loading the trained model from disk")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    logger.info("Model loaded successfully")
    return model
def predictions(x_test_path, model):
    logger.info("Loading test data for predictions")
    X_test = pd.read_csv(x_test_path)
    logger.info("Test data loaded successfully")
    logger.info("Making predictions on the test data")
    y_pred = model.predict(X_test)
    logger.info("Predictions made successfully")
    return y_pred

# Evaluate model
def evaluate_model(y_test_path, y_pred):
    logger.info("Loading true labels for evaluation")
    y_test = pd.read_csv(y_test_path)
    logger.info("True labels loaded successfully")
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Model evaluation completed. Accuracy: {accuracy}")
    return accuracy
    mlflow.log_metric("accuracy", accuracy)

if __name__=="__main__":
    model=load_model(model_path)
    y_pred=predictions(x_test_path, model)
    accuracy=evaluate_model(y_test_path, y_pred)
    print(f"Final accuracy: {accuracy}")