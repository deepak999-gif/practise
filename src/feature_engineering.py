import os
import logging 
import pandas as pd
from sklearn.model_selection import train_test_split
logger=logging.getLogger("Feature_Engineering_Logs")
logger.setLevel(logging.INFO)
stream_handler=logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
file_handler=logging.FileHandler('Logs/feature_engineering.log')
file_handler.setLevel(logging.INFO)
formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
logger.addHandler(file_handler)
path="data/processed/ml_pipeline_dummy_data.csv"

def save_data(df,file_name):
    logger.info("Saving the feature_engineered data to the specified path")
    path=os.path.join("data", "feature_engineered")
    os.makedirs(path, exist_ok=True)
    df.to_csv(os.path.join(path,file_name),index=False)
    logger.info(f"Feature-engineered data saved to {os.path.join(path,file_name)}")
    
def age_to_experience(path):
    logger.info("Starting feature engineering: creating new columns experience to age ratio")
    df=pd.read_csv(path)
    logger.info("Read the data from the csv file")
    df['experience_to_age_ratio'] = df['experience_years'] / df['age']
    logger.info("Feature engineering completed successfully")
    return df
# Split features and target
def split_features_target(df):
    logger.info("Splitting features and target")
    X = df.drop(columns=["promoted"])
    y = df["promoted"]
    logger.info("Features and target split successfully")
    X.to_csv("features.csv", index=False)
    logger.info("Features saved to features.csv")
    y.to_csv("target.csv", index=False)
    logger.info("Target saved to target.csv")
    return df




if __name__=="__main__":
    df=age_to_experience(path)
    df=split_features_target(df)
    save_data(df,"ml_pipeline_dummy_data.csv")