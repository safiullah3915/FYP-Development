"""
Logging configuration for Flask Recommendation Service
"""
import logging
import sys
from pathlib import Path
from config import LOG_LEVEL, LOG_FILE

# Create logs directory if it doesn't exist
LOG_DIR = Path(__file__).parent.parent / 'logs'
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE_PATH = LOG_DIR / LOG_FILE


def setup_logger(name: str = __name__) -> logging.Logger:
    """
    Set up a logger with file and console handlers
    
    Args:
        name: Logger name (usually __name__)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    
    # File handler
    file_handler = logging.FileHandler(LOG_FILE_PATH)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def get_logger(name: str = __name__) -> logging.Logger:
    """
    Get a logger instance (convenience function)
    
    Args:
        name: Logger name (usually __name__)
    
    Returns:
        Logger instance
    """
    return setup_logger(name)


