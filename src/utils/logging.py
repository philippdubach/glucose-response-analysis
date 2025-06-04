"""Logging utilities."""

import logging
import logging.config
import yaml
from pathlib import Path

def setup_logging(config_path: str = None):
    """Setup logging configuration."""
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "config" / "logging.yaml"
    
    if Path(config_path).exists():
        with open(config_path, 'r') as f:
            logging_config = yaml.safe_load(f)
        logging.config.dictConfig(logging_config)
    else:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

def get_logger(name: str) -> logging.Logger:
    """Get logger instance."""
    return logging.getLogger(name)