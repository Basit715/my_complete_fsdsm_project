import os
import sys
import pandas as pd
import numpy as np

from src.DiamondPricePrediction.logger import logging
from src.DiamondPricePrediction.exception import customException
from src.DiamondPricePrediction.utils.utils import load_object

class PredictPipeline:
     def __init__(self):
          pass
     
     def predict(self,features):
          try:
               preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
               model_path = os.path.join("artifacts", "model.pkl")
               
               preprocessor = load_object(preprocessor_path)
               model = load_object(model_path)
               
               scaled_data = preprocessor.transform(features)
               prediction = model.predict(scaled_data)
               
               return prediction
               
               
          except Exception as e:
               logging.info('Exception occured in predict function of predict pipeline module')
               raise customException(e,sys)
          
          
class customData:
     def __init__(self, carat:float,depth:float,table:float,x:float,y:float,z:float,cut:str,color:str,clarity:str):
          self.carat = carat
          self.depth = depth
          self.table = table
          self.x = x
          self.y = y
          self.z = z
          self.cut = cut
          self.color = color
          self.clarity = clarity