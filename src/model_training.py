
import pickle
import mlflow
import mlflow.sklearn

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import logging
import os
import pickle
logger=logging.getLogger("Model_Training_Logs")
logger.setLevel(logging.INFO)
stream_handler=logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
file_handler=logging.FileHandler('Logs/model_training.log')
file_handler.setLevel(logging.INFO)
formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
logger.addHandler(file_handler)

features_path="features.csv"
target_path="target.csv"
logger.info("MLflow tracking URI set to DagsHub")
mlflow.set_tracking_uri("https://dagshub.com/deepak999-gif/practise.mlflow")
import dagshub
dagshub.init(repo_owner='deepak999-gif', repo_name='practise', mlflow=True)
mlflow.set_experiment("ML_Pipeline_Experiment with decision tree classifier")
logger.info("MLflow experiment set to ML_Pipeline_Experiment with decision tree classifier")

def load_data(features_path, target_path):
    logger.info("Loading features and target data")
    X = pd.read_csv(features_path)
    y = pd.read_csv(target_path)
    logger.info("Features and target data loaded successfully")
    return X, y


def split_train_test(X,y):
    logger.info("Splitting data into training and testing sets")
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=40)
    logger.info("Data split successfully")
    return X_train, X_test, y_train, y_test

def save_data(X_train, X_test, y_train, y_test):
    X_train.to_csv("X_train.csv", index=False)
    y_train.to_csv("y_train.csv", index=False)
    X_test.to_csv("X_test.csv", index=False)
    y_test.to_csv("y_test.csv", index=False)
    
# Create and train baseline model
def model_training(X_train, y_train):
    with mlflow.start_run():
      logger.info("Training the baseline model")
      model = DecisionTreeClassifier(random_state=42)
      model.fit(X_train, y_train)
      mlflow.sklearn.log_model(
    model,
    name="model",
    skops_trusted_types=["sklearn.tree._tree.Tree"]
)
      mlflow.log_artifact("Logs/model_training.log")
      mlflow.log_artifact('src/model_training.py')
    
    logger.info("Baseline model trained successfully")
    return model

def saving_model(model):
    logger.info("Saving the trained model to disk")
    with open ('model.pkl','wb')as f:
      pickle.dump(model,f)
    logger.info("Model saved successfully")
    
if __name__=="__main__":
    X,y=load_data(features_path, target_path)
    X_train, X_test, y_train, y_test=split_train_test(X, y)
    save_data(X_train, X_test, y_train, y_test)
    model=model_training(X_train, y_train)
    saving_model(model)
    
    