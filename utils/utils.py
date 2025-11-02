from pydantic import BaseModel, Field
import os
import logging
import time
from functools import wraps


class TopicModel(BaseModel):
    """Pydantic model for structured topic list output.

    Attributes:
        topics_list (list[str]): List of topics generated from the text.
    """
    topics_list: list[str] = Field(description="List of topics generated from the text.")


LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("ScreenLoom")
logger.setLevel(logging.INFO)  # Default level; can be overridden

if not logger.handlers:
    # File handler with rotation
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    main_handler = logging.handlers.RotatingFileHandler(
        filename=os.path.join(LOG_DIR, "logs.log"),
        maxBytes=1024 * 1024,   # 1 MB
        backupCount=5   # Keep 5 backup files
    )
    main_handler.setLevel(logging.INFO)
    main_handler.setFormatter(formatter)
    logger.addHandler(main_handler)

    error_logs_handler = logging.handlers.RotatingFileHandler(
        filename=os.path.join(LOG_DIR, "errors.log"),
        maxBytes=1024 * 1024,  # 1 MB
        backupCount=5
    )
    error_logs_handler.setLevel(logging.ERROR) 
    error_logs_handler.setFormatter(formatter)
    logger.addHandler(error_logs_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)  # Show all levels in console
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


def get_logger():
    """Returns the configured logger instance.

    Returns:
        logging.Logger: The ScreenLoom logger with file and console handlers.
    """
    return logger


def time_execution(func):
    """Decorator to log the execution time of a function.

    Args:
        func (callable): The function to decorate.

    Returns:
        callable: Wrapped function with timing and logging.
    """
    log_level = logging.INFO
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.log(log_level, "Function '%s' started.", func.__name__)
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.log(log_level, "TIME_PERF: Function '%s' took %.4f seconds to execute", func.__name__, execution_time)
        return result
    return wrapper

def generate_excel_columns(n):
    """
    Generate Excel-style column names (A, B, ..., Z, AA, AB, ...).
    
    Args:
        n (int): Number of column names to generate.
    
    Returns:
        list: List of column names.
    """
    columns = []
    for i in range(n):
        name = ""
        while True:
            name = chr(65 + (i % 26)) + name
            i //= 26
            if i == 0:
                break
        columns.append(name)
    return columns