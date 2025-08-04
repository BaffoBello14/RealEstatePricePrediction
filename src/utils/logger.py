"""
Simple logger utility
"""
import logging
import yaml
from pathlib import Path
from typing import Optional

def setup_logger(config_path: str) -> logging.Logger:
    """
    Set up logger with configuration from YAML file.
    
    Args:
        config_path: Path to the configuration YAML file
        
    Returns:
        Configured logger instance
    """
    # Load configuration
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Get logging configuration
    logging_config = config.get('logging', {})
    level = logging_config.get('level', 'INFO')
    format_str = logging_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = logging_config.get('file', 'logs/pipeline.log')
    
    # Create logs directory if it doesn't exist
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    handlers = [
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_str,
        handlers=handlers,
        force=True
    )
    
    # Return main logger
    logger = logging.getLogger('ML_Pipeline')
    return logger

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a simple logger.
    
    Args:
        name: Logger name, defaults to 'ML_Pipeline' if None
        
    Returns:
        Logger instance
    """
    if name is None:
        name = 'ML_Pipeline'
        
    logger = logging.getLogger(name)
    
    # Only add handlers if not already configured
    if not logger.handlers and not logging.getLogger().handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    return logger