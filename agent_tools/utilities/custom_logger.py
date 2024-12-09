import sys
import logging
import coloredlogs
from typing import Any

def non_empty_check(variable: Any, expected_type: Any, variable_name: str) -> None:
    if not isinstance(variable, expected_type):
        raise TypeError(f"{variable_name} must be of type {expected_type}")
    
    if not variable:
        raise ValueError(f"{variable_name} cannot be empty")
    
    return

def set_level(log_level: str) -> int:
    """
    Convert string log level to logging constant.

    :param log_level: String representation of log level (info, debug, warning, error, exception)
    :returns: Logging level constant
    :raises:
        TypeError: If log_level is not string
        ValueError: If log_level is empty
    """
    non_empty_check(variable=log_level, expected_type=str, variable_name="log_level")
    log_level = log_level.lower()

    if "info" in log_level:
        return logging.INFO
    elif "deb" in log_level:
        return logging.DEBUG
    elif "warn" in log_level:
        return logging.WARNING
    elif "err" in log_level:
        return logging.ERROR
    elif "exc" in log_level:
        return logging.CRITICAL
    else:
        return logging.INFO

def null_logger() -> logging.Logger:
    """
    Create a logger that suppresses all output.

    :returns: Logger with NullHandler
    """
    null_logger = logging.getLogger("null_logger")
    null_logger.addHandler(logging.NullHandler())
    return null_logger

def setup_logger(name: str, log_level: str = "INFO") -> logging.Logger:
    """
    Configure and return a logger with console output and colored formatting.

    :param name: Name for the logger instance
    :param log_level: Logging level (info, debug, warning, error, exception)
    :returns: Configured logger instance
    :raises:
        TypeError: If name is not string
        ValueError: If name is empty

    Features:
        - Colored output using coloredlogs
        - Console handler writing to stdout
        - Formatted output with level, timestamp, file, line number
        - Prevents duplicate handlers
    """
    non_empty_check(variable=name, expected_type=str, variable_name="logger name")
    logger = logging.getLogger(name)
    logger.setLevel(set_level(log_level))
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(set_level(log_level))
    formatter = logging.Formatter(
        "%(levelname)s - %(asctime)s - %(filename)s: %(lineno)d - %(message)s"
    )
    console_handler.setFormatter(formatter)
    coloredlogs.install(level="DEBUG", logger=logger)
    if not logger.hasHandlers():
        logger.addHandler(console_handler)
    return logger

def get_logger(name: str, null_logger: bool = False) -> logging.Logger:
    """
    Retrieve a logger by name.

    :param name: Name for the logger instance
    :returns: Logger instance
    :raises:
        TypeError: If name is not string
        ValueError: If name is empty
    """

    if null_logger:
        return null_logger()
    
    non_empty_check(variable=name, expected_type=str, variable_name="name")
    return logging.getLogger(name)
