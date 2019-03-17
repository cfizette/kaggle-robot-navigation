import sys
import logging

def setup_logger(logfile, console_out=False):
    # Create file if does not exist
    with open(logfile, 'a') as f:
        pass

    logger = logging.getLogger()
    logger.setLevel('INFO')

    filehandler = logging.FileHandler(logfile)
    logger.addHandler(filehandler)

    if console_out:
        console_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(console_handler)

    return logger