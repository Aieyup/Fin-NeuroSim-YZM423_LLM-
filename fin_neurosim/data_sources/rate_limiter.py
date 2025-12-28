"""
Rate limiter modülü: API çağrılarını sınırlama.

Bu modül, API çağrılarının rate limit'leri aşmamasını sağlar.
"""

import asyncio
import time
from collections import defaultdict
from typing import Dict, Optional
from datetime import datetime, timedelta

from fin_neurosim.utils.logging import setup_logger

logger = setup_logger(__name__)


class RateLimiter:
    """
    API çağrılarını sınırlayan rate limiter.

    Her API için ayrı rate limit kuralları tanımlanabilir.
    """

    def __init__(self):
        """Rate limiter'ı başlatır."""
        self._limits: Dict[str, Dict[str, int]] = {}
        self._call_history: Dict[str, list] = defaultdict(list)

    def add_limit(
        self,
        api_name: str,
        max_calls: int,
        time_window_seconds: int
    ) -> None:
        """
        Bir API için rate limit kuralı ekler.

        Args:
            api_name: API adı (örn: "alpha_vantage", "fred").
            max_calls: Zaman penceresi içinde izin verilen maksimum çağrı sayısı.
            time_window_seconds: Zaman penceresi (saniye).
        """
        self._limits[api_name] = {
            "max_calls": max_calls,
            "time_window": time_window_seconds
        }
        logger.info(
            f"Rate limit eklendi: {api_name} - "
            f"{max_calls} çağrı / {time_window_seconds} saniye"
        )

    async def wait_if_needed(self, api_name: str) -> None:
        """
        Gerekirse rate limit'e takılmamak için bekler.

        Args:
            api_name: API adı.
        """
        if api_name not in self._limits:
            return

        limit_config = self._limits[api_name]
        max_calls = limit_config["max_calls"]
        time_window = limit_config["time_window"]

        now = time.time()
        cutoff_time = now - time_window

        # Eski çağrıları temizle
        self._call_history[api_name] = [
            call_time for call_time in self._call_history[api_name]
            if call_time > cutoff_time
        ]

        # Rate limit aşıldıysa bekle
        if len(self._call_history[api_name]) >= max_calls:
            oldest_call = min(self._call_history[api_name])
            wait_time = time_window - (now - oldest_call) + 0.1
            if wait_time > 0:
                logger.warning(
                    f"Rate limit aşıldı: {api_name}. "
                    f"{wait_time:.2f} saniye bekleniyor..."
                )
                await asyncio.sleep(wait_time)

        # Çağrıyı kaydet
        self._call_history[api_name].append(time.time())

