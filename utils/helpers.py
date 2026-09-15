"""Helper utility functions"""
import time
from datetime import datetime

def format_uptime(start_time: float) -> str:
    """Format uptime in human-readable form"""
    uptime = time.time() - start_time
    hours = int(uptime // 3600)
    minutes = int((uptime % 3600) // 60)
    return f'{hours}h {minutes}m'

def truncate_text(text: str, max_length: int = 150) -> str:
    """Truncate text to max length with ellipsis"""
    if len(text) <= max_length:
        return text
    return text[:max_length] + '...'

def format_timestamp(timestamp: float = None) -> str:
    """Format timestamp as readable string"""
    if timestamp is None:
        timestamp = time.time()
    return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
