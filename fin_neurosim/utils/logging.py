"""
Logging yardımcı modülü: Yapılandırılmış loglama.

Bu modül, sistem genelinde tutarlı loglama sağlar.
"""

import logging
import sys
from typing import Optional


def setup_logger(
    name: str,
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Yapılandırılmış logger oluşturur.

    Args:
        name: Logger adı (genellikle __name__).
        level: Log seviyesi (default: INFO).
        format_string: Özel format string (opsiyonel).

    Returns:
        Yapılandırılmış logger instance'ı.
    """
    if format_string is None:
        format_string = (
            '%(asctime)s - %(name)s - %(levelname)s - '
            '%(message)s'
        )

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Handler zaten varsa ekleme
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(format_string)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

