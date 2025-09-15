"""
Einfaches Logging-System für die Bundeskanzler KI.
"""

import logging
import sys
from pathlib import Path

def setup_simple_logging():
    """
    Richtet einfaches Logging für API und Memory ein.
    
    Returns:
        Tuple von (api_logger, memory_logger)
    """
    # Erstelle logs Verzeichnis falls es nicht existiert
    logs_dir = Path("./logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Konfiguriere API Logger
    api_logger = logging.getLogger('bundeskanzler_api')
    api_logger.setLevel(logging.INFO)
    
    # Konfiguriere Memory Logger  
    memory_logger = logging.getLogger('memory_system')
    memory_logger.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console Handler für beide Logger
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # File Handler für API
    api_file_handler = logging.FileHandler(logs_dir / 'api.log')
    api_file_handler.setFormatter(formatter)
    api_logger.addHandler(api_file_handler)
    
    # File Handler für Memory
    memory_file_handler = logging.FileHandler(logs_dir / 'memory.log')
    memory_file_handler.setFormatter(formatter)
    memory_logger.addHandler(memory_file_handler)
    
    # Füge Console Handler zu beiden Loggern hinzu
    api_logger.addHandler(console_handler)
    memory_logger.addHandler(console_handler)
    
    return api_logger, memory_logger
