import sys
from src.logger import logging

def error_message_detail(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()
    filename = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occurred in python script name [{0}] line number [{1}] error message [{2}]".format(
        filename, exc_tb.tb_lineno, str(error)
    )
    return error_message

import traceback

class CustomException(Exception):
    def __init__(self, error_message, error_detail=None):
        super().__init__(error_message)
        self.error_message = error_message
        self.error_detail = error_detail

    def __str__(self):
        if self.error_detail:
            _, _, tb = self.error_detail.exc_info()
            detailed_message = f"{self.error_message} \nTraceback: {traceback.format_tb(tb)}"
            return detailed_message
        return self.error_message