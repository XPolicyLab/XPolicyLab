import logging

def get_logger(name=__name__, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger

def setup_logging(level=logging.INFO):
    logging.basicConfig(level=level, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
