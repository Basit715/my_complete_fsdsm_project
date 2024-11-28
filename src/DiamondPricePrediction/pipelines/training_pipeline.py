from src.DiamondPricePrediction.components.data_ingestion import DataIngestion
from src.DiamondPricePrediction.components.data_transformation import DataTransformation
from src.DiamondPricePrediction.components.model_trainer import ModelTrainer

import os
import sys
from src.DiamondPricePrediction.logger import logging
from src.DiamondPricePrediction.exception import customException
import pandas as pd


obj = DataIngestion()
train_path,test_path = obj.initiate_data_ingestion()


obj1 = DataTransformation()
train_array,test_array = obj1.initiate_data_transformation(train_path,test_path)

obj2 = ModelTrainer()
obj2.InitiateModelTraining(train_array,test_array)