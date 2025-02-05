import logging
import os
import sys


def file_handler_filter(log_record):
    return log_record.levelno == logging.INFO 

def configure_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(message)s')
    sh.setFormatter(formatter)
    logger.addHandler(sh)

def configure_logger_file_handler(save_path):
    logger = logging.getLogger()
    fh = logging.FileHandler(save_path)
    fh.setLevel(logging.INFO)
    fh.addFilter(file_handler_filter)
    formatter = logging.Formatter('%(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)