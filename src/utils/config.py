"""Configuration management utilities."""

import yaml
import os
from pathlib import Path
from typing import Dict, Any

class Config:
    """Configuration manager for the glucose response analysis project."""
    
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
        
        self.config_path = Path(config_path)
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def get(self, key: str, default=None):
        """Get configuration value using dot notation."""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_data_path(self, filename: str = None) -> Path:
        """Get data file path."""
        base_path = Path(self.get('data.raw_data_path', 'data/raw/'))
        if filename:
            return base_path / filename
        return base_path
    
    def get_results_path(self, subfolder: str = None) -> Path:
        """Get results path."""
        base_path = Path("results")
        if subfolder:
            return base_path / subfolder
        return base_path

# Global config instance
config = Config()