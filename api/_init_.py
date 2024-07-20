import os
import sys
from logging import INFO, getLogger, Formatter, FileHandler, StreamHandler

from dotenv import load_dotenv

# Set up logging to file
log_file = 'app.log'
formatter = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler = FileHandler(log_file)
file_handler.setFormatter(formatter)

# Get the root logger and add the file handler
root_logger = getLogger()
root_logger.setLevel(INFO)
root_logger.addHandler(file_handler)

# Also log to stdout
stream_handler = StreamHandler(sys.stdout)
stream_handler.setLevel(INFO)
stream_handler.setFormatter(formatter)
root_logger.addHandler(stream_handler)


def start_app():
    # When environment is dev
    if os.path.exists(".env_dev"):
        load_dotenv('.env_dev')
        root_logger.info('Development environment running')
    # When environment is prod
    else:
        load_dotenv()
        root_logger.info('Production environment running')
