import logging
import sys


def setup_logging(level=logging.INFO):
    logger = logging.getLogger()
    logger.setLevel(level)

    c_handler = logging.StreamHandler(sys.stdout)  # Console
    f_handler = logging.FileHandler("project.log")  # File

    log_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    c_handler.setFormatter(log_format)
    f_handler.setFormatter(log_format)

    if not logger.handlers:
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)

    return logger
