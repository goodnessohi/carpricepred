import logging
import os
from datetime import datetime

# Create a logs directory if it doesn't exist
logs_path = os.path.join(os.getcwd(), 'logs')
os.makedirs(logs_path, exist_ok=True)

# Set the log file path with the current date and time
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)
print("LOG_FILE_PATH:", LOG_FILE_PATH)  # <-- Added print statement

# Configure logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s] %(lineno)d %(name)s - %(levelname)s-%(message)s ",
    level=logging.INFO,
)