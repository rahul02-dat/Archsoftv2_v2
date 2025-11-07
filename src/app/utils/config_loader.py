import yaml
import os
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class ConfigLoader:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()
        self._apply_env_overrides()
        
    def _load_config(self) -> Dict:
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
                logger.info(f"Configuration loaded from {self.config_path}")
                return config
        except FileNotFoundError:
            logger.error(f"Config file not found: {self.config_path}")
            return {}
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}
            
    def _apply_env_overrides(self):
        env_mappings = {
            'RTSP_URL': ('camera', 'rtsp_url'),
            'MONGO_URI': ('database', 'mongo_uri'),
            'API_HOST': ('api', 'host'),
            'API_PORT': ('api', 'port'),
        }
        
        for env_var, (section, key) in env_mappings.items():
            value = os.getenv(env_var)
            if value:
                if section not in self.config:
                    self.config[section] = {}
                    
                if env_var == 'API_PORT':
                    value = int(value)
                    
                self.config[section][key] = value
                logger.info(f"Override from env: {section}.{key} = {value}")
                
    def get(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)
        
    def __getitem__(self, key: str) -> Any:
        return self.config[key]
        
    def __contains__(self, key: str) -> bool:
        return key in self.config