"""
Zaman yardımcı modülü: Tarih ve zaman işlemleri.

Bu modül, zaman damgaları, tazelik skorları ve
tarih aralığı hesaplamaları için yardımcı fonksiyonlar sağlar.
"""

from datetime import datetime, timedelta
from typing import Optional


def calculate_freshness_score(
    timestamp: datetime,
    max_age_hours: int = 24,
    reference_time: Optional[datetime] = None
) -> float:
    """
    Veri tazeliği skoru hesaplar.

    Veri ne kadar yeni ise, skor 1'e o kadar yakın olur.
    Veri max_age_hours'dan eski ise, skor 0'a yakın olur.

    Args:
        timestamp: Verinin zaman damgası.
        max_age_hours: Maksimum kabul edilebilir yaş (saat, default: 24).
        reference_time: Referans zaman (default: şimdi).

    Returns:
        Tazelik skoru (0-1 arası).
    """
    if reference_time is None:
        reference_time = datetime.utcnow()

    age_hours = (reference_time - timestamp).total_seconds() / 3600.0

    if age_hours <= 0:
        return 1.0

    if age_hours >= max_age_hours:
        return 0.0

    # Lineer interpolasyon
    freshness = 1.0 - (age_hours / max_age_hours)
    return max(0.0, min(1.0, freshness))


def get_time_range(
    days_back: int = 30
) -> tuple[datetime, datetime]:
    """
    Belirli bir gün geriye giden zaman aralığı döndürür.

    Args:
        days_back: Kaç gün geriye gidileceği (default: 30).

    Returns:
        (başlangıç_zamanı, bitiş_zamanı) tuple'ı.
    """
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=days_back)
    return (start_time, end_time)


def format_timestamp(timestamp: datetime) -> str:
    """
    Zaman damgasını ISO 8601 formatında string'e çevirir.

    Args:
        timestamp: Zaman damgası.

    Returns:
        ISO 8601 formatında string.
    """
    return timestamp.isoformat() + 'Z'

