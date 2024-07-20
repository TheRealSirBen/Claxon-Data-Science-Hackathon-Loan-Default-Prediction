import sys
from logging import INFO, getLogger, Formatter, FileHandler, StreamHandler
from os import makedirs
from os.path import exists

from dotenv import load_dotenv

makedirs('logs', exist_ok=True)
makedirs('predictions', exist_ok=True)


def setup_logger(name, log_file, level=INFO):
    formatter = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    file_handler = FileHandler(log_file)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)

    stream_handler = StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level)

    logger = getLogger(name)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


# Set up app logger
app_logger = setup_logger('app', 'logs/app.log')

# Set up datadrift logger
datadrift_logger = setup_logger('datadrift', 'logs/datadrift.log')


def start_app():
    # When environment is dev
    if exists(".env_dev"):
        load_dotenv('.env_dev')
        app_logger.info('Development environment running')
    # When environment is prod
    else:
        load_dotenv()
        app_logger.info('Production environment running')
