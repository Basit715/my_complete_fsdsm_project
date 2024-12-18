import logging
import os
from datetime import datetime

LOG_FILE = f'{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.log'
LOG_PATH = os.path.join(os.getcwd(),"logs")


LOG_DIR = os.makedirs(LOG_PATH, exist_ok=True)

LOG_FILE_PATH = os.path.join(LOG_PATH, LOG_FILE)


logging.basicConfig(level=logging.INFO, filename=LOG_FILE_PATH,format="[%(asctime)s]  %(lineno)d  %(name)s %(levelname)s %(message)s ")