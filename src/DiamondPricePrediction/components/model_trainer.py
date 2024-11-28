import pandas as pd
import numpy as np
import os
import sys

from src.DiamondPricePrediction.logger import logging
from src.DiamondPricePrediction.exception import customException
from dataclasses import dataclass
from src.DiamondPricePrediction.utils.utils import save_object
from src.DiamondPricePrediction.utils.utils import evaluate_model

from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet




class ModelTrainingCinfiguration:
     trained_model_file_path = os.path.join("artifacts", 'model.pkl')

class ModelTrainer:
     def __init__(self):
          self.model_trainer_config = ModelTrainingCinfiguration()
     
     
     
     
     def InitiateModelTraining(self):
          x_train,y_train,x_test,y_test = (
               train_array[:,:-1]
               train_array[:,-1]
               test_array[:,:-1]
               test_array[:,-1]
          )
          models = {
     'LinearRegression': LinearRegression(),
     'Lasso': Lasso(),
     'Ridge': Ridge(),
     'ElasticNet': ElasticNet()
          }
          
          model_report:dict = evaluate_model(x_train,y_train,x_test,y_test)
          print(model_report)
          print("======================================\n")
          logging.info("Model Report", model_report)
          
          
          best_model_score = max(sorted(model_report.values()))
          
          best_model_name = list(model_report.keys())[
               list(model_report.values()).index(best_model_score)
          ]
          
          best_model = models[best_model_name]
          print(f"Best model Found, Model name: {best_model}, R2 scor: {best_model_score}")
          print("\n===================================================\n")
          
          
          save_object(
               file_path = self.model_trainer_config.trained_model_file_path,
               obj = best_model
          )