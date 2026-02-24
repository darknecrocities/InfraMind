import logging
import os
import yaml

def setup_logging(config_path="config/config.yaml"):
    """
    Setup logging configuration based on config.yaml
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        log_level = config.get('logging', {}).get('level', 'INFO')
        log_format = config.get('logging', {}).get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        logging.basicConfig(level=log_level, format=log_format)
        return logging.getLogger("InfraMind")
    except Exception as e:
        # Fallback to default logging if config fails
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger("InfraMind")
        logger.warning(f"Failed to load logging config: {e}. Using defaults.")
        return logger
