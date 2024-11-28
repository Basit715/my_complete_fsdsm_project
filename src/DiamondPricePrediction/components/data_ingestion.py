import pandas as pd
import numpy as np
import os
import sys

from src.DiamondPricePrediction.logger import logging
from src.DiamondPricePrediction.exception import customException


from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from pathlib import Path


class DataIngestionConfig:
          raw_data_path = os.path.join("artifacts", "raw.csv")
          train_data_path = os.path.join("artifacts", "train1.csv")
          test_data_path = os.path.join("artifacts", "test.csv")

class DataIngestion:
     def __init__(self):
          self.ingestion_config = DataIngestionConfig()
     
     def initiate_data_ingestion(self):
          logging.info("Data ingestion started")
          
          
          try:
               my_data = pd.read_csv(Path(os.path.join("notebooks/data", "train.csv")))
               logging.info("Read the dataset as a dataframe")
               
               
               os.makedirs(os.path.dirname(os.path.join(self.ingestion_config.raw_data_path)), exist_ok=True)
               my_data.to_csv(self.ingestion_config.raw_data_path, index=False)
               logging.info("saved the raw data in artifacts folder")
               
               
               logging.info("perform train test split")
               train_data,test_data = train_test_split(my_data, test_size=0.25,random_state=25)
               logging.info("train test split completed")
               
               train_data.to_csv(self.ingestion_config.train_data_path, index=False)
               test_data.to_csv(self.ingestion_config.test_data_path, index=False)
               
               logging.info("Data ingestion part completed")
               
               return(
                    self.ingestion_config.train_data_path,
                    self.ingestion_config.test_data_path
               )
               
          except Exception as e:
               logging.info("Exception occured at the data ingestion stage")
               raise customException(e,sys)
     