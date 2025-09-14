"""
Einfaches, funktionierendes Logging-Setup für Bundeskanzler KI
"""
import logging
import logging.handlers
import json
import os
from pathlib import Path
from datetime import datetime


class StructuredFormatter(logging.Formatter):
    """Custom Formatter für strukturierte Logs im JSON-Format"""
    
    def format(self, record):
        # Basis Log-Daten
        log_data = {
            'timestamp': self.formatTime(record, '%Y-%m-%d %H:%M:%S'),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Extra-Daten hinzufügen falls vorhanden
        for attr in ['component', 'action', 'user_id', 'error', 'response_time', 
                     'message_length', 'query', 'top_k', 'content_length']:
            if hasattr(record, attr):
                log_data[attr] = getattr(record, attr)
                
        return json.dumps(log_data, ensure_ascii=False)


def setup_simple_logging():
    """Konfiguriert einfaches strukturiertes Logging"""
    logs_dir = Path("./logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Console Handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)
    
    # File Handlers mit strukturiertem Format
    api_handler = logging.handlers.RotatingFileHandler(
        logs_dir / "api.log", maxBytes=10*1024*1024, backupCount=5
    )
    api_handler.setFormatter(StructuredFormatter())
    api_handler.setLevel(logging.DEBUG)
    
    memory_handler = logging.handlers.RotatingFileHandler(
        logs_dir / "memory.log", maxBytes=10*1024*1024, backupCount=5
    )
    memory_handler.setFormatter(StructuredFormatter())
    memory_handler.setLevel(logging.DEBUG)
    
    error_handler = logging.handlers.RotatingFileHandler(
        logs_dir / "errors.log", maxBytes=10*1024*1024, backupCount=5
    )
    error_handler.setFormatter(StructuredFormatter())
    error_handler.setLevel(logging.ERROR)
    
    # API Logger
    api_logger = logging.getLogger("bundeskanzler_api")
    api_logger.setLevel(logging.DEBUG)
    api_logger.handlers.clear()  # Alte Handler entfernen
    api_logger.addHandler(console_handler)
    api_logger.addHandler(api_handler)
    api_logger.addHandler(error_handler)
    api_logger.propagate = False
    
    # Memory Logger
    memory_logger = logging.getLogger("memory")
    memory_logger.setLevel(logging.DEBUG)
    memory_logger.handlers.clear()  # Alte Handler entfernen
    memory_logger.addHandler(memory_handler)
    memory_logger.addHandler(error_handler)
    memory_logger.propagate = False
    
    # Initialisierungs-Log
    api_logger.info("Einfaches Logging-System initialisiert", extra={
        "component": "logging",
        "action": "setup"
    })
    
    return api_logger, memory_logger