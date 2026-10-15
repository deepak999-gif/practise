import os
import logging
import pandas as pd
logger=logging.getLogger("Data_Preprocessing_Logs")
logger.setLevel(logging.INFO)
stream_handler=logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
file_handler=logging.FileHandler('Logs/data_preprocessing.log')
file_handler.setLevel(logging.INFO)
formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
logger.addHandler(file_handler)
path="data/raw/ml_pipeline_dummy_data.csv"

def save_data(df,file_name):
    logger.info("Saving the processed data to the specified path")
    path=os.path.join("data", "processed")
    os.makedirs(path, exist_ok=True)
    df.to_csv(os.path.join(path,file_name),index=False)
    logger.info(f"Processed data saved to {os.path.join(path,file_name)}")
    
def fill_na_values(path):
    logger.info("Starting data preprocessing")
    df=pd.read_csv(path)
    logger.info("Read the data from the csv file")
    df['experience_years']=df['experience_years'].fillna(df['experience_years'].median())
    logger.info("Filled missing values in 'experience_years' column with median")
    df['salary']=df['salary'].fillna(df['salary'].median())
    logger.info("Filled missing values in 'salary' column with median")
    return df
def categorical_encoding(df):
    logger.info("Starting categorical encoding")
    df = pd.get_dummies(df, columns=["city"], dtype=int)
    logger.info("Categorical encoding completed successfully")
    return df

if __name__=="__main__":
    df=fill_na_values(path)
    df=categorical_encoding(df)
    save_data(df,"ml_pipeline_dummy_data.csv")
    
