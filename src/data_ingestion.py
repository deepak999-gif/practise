import pandas as pd
import os
os.makedirs('Logs', exist_ok=True)
import logging
logger=logging.getLogger("Data_Ingestion_Logs")
logger.setLevel(logging.INFO)
console_handler=logging.StreamHandler()
console_handler.setLevel(logging.INFO)
file_handler=logging.FileHandler('Logs/data_ingestion.log')
file_handler.setLevel(logging.INFO)
formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)
path=r"C:\Users\LENOVO\Downloads\categories.csv"
def ingest_data(path):
    logger.info("Starting data ingestion process")
    df=pd.read_csv(path)
    logger.info("readed the data from the csv file")
    paths=os.path.join("data","raw")
    os.makedirs(paths,exist_ok=True)
    df.to_csv(os.path.join(paths,"categories.csv"),index=False)
    logger.info("Data ingestion completed successfully")
    
if __name__=="__main__":
    ingest_data(path)