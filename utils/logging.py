import sys
import logging
from uuid import uuid4

def setup_logger(logfile, console_out=False, name=str(uuid4())):
    # Create file if does not exist
    with open(logfile, 'a') as f:
        pass

    logger = logging.getLogger(name)
    logger.setLevel('INFO')

    filehandler = logging.FileHandler(logfile)
    logger.addHandler(filehandler)

    if console_out:
        console_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(console_handler)

    return logger