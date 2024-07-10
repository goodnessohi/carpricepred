from src.exception import CustomException
from src.logger import logging

import sys, os

sys.path.insert(0, '.')

#Simulating an error to test that exception and logger work well
try:
    a = 1/0
    logging.info("Division beung attempted")
except ZeroDivisionError:
    error_detail = sys.exc_info()
    logging.info("Division failed because divisor is Zero(0)")
    print("Division failed because divisor is Zero(0)")
    raise CustomException("Division failed because divisor is Zero(0)", error_detail, sys)


