import logging
import os
from pytorch_lightning.utilities import rank_zero_only

def get_pylogger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""
    logger = logging.getLogger(name)

    # Configure log level
    logger.setLevel(logging.DEBUG)  # Set the desired log level

    # Create a "logs" directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)

    # Create a FileHandler to write log messages to a file
    log_filename = f"logs/{name}.log"
    file_handler = logging.FileHandler(log_filename)

    # Define log message format
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    file_handler.setFormatter(formatter)

    # Add the FileHandler to the logger
    logger.addHandler(file_handler)

    # Mark log levels for rank zero only
    logging_levels = (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    )
    for level in logging_levels:
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger
