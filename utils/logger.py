# utils/logger.py
"""
Compatibility wrapper so that tests importing `utils.logger` continue to work.
"""
from .logging import CSVLogger, JSONLLogger
