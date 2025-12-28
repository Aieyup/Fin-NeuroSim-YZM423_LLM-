"""
Retry yardımcı modülü: Hata durumlarında yeniden deneme mekanizması.

Bu modül, API çağrıları ve diğer kritik işlemler için
yeniden deneme (retry) mantığı sağlar.
"""

import asyncio
import logging
from typing import Callable, TypeVar, Optional, List
from functools import wraps

T = TypeVar('T')

logger = logging.getLogger(__name__)


async def async_retry(
    func: Callable[..., T],
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
    *args,
    **kwargs
) -> T:
    """
    Asenkron fonksiyon için retry mekanizması.

    Args:
        func: Çağrılacak asenkron fonksiyon.
        max_attempts: Maksimum deneme sayısı (default: 3).
        delay: İlk gecikme süresi (saniye, default: 1.0).
        backoff: Gecikme çarpanı (default: 2.0).
        exceptions: Yakalanacak exception tipleri.
        *args: Fonksiyon argümanları.
        **kwargs: Fonksiyon keyword argümanları.

    Returns:
        Fonksiyonun döndürdüğü değer.

    Raises:
        Son denemede hata devam ederse, exception fırlatılır.
    """
    current_delay = delay
    last_exception = None

    for attempt in range(1, max_attempts + 1):
        try:
            return await func(*args, **kwargs)
        except exceptions as e:
            last_exception = e
            if attempt < max_attempts:
                logger.warning(
                    f"Deneme {attempt}/{max_attempts} başarısız: {e}. "
                    f"{current_delay:.2f} saniye sonra tekrar denenecek."
                )
                await asyncio.sleep(current_delay)
                current_delay *= backoff
            else:
                logger.error(
                    f"Tüm {max_attempts} deneme başarısız. Son hata: {e}"
                )

    raise last_exception


def retry_sync(
    func: Callable[..., T],
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
    *args,
    **kwargs
) -> T:
    """
    Senkron fonksiyon için retry mekanizması.

    Args:
        func: Çağrılacak senkron fonksiyon.
        max_attempts: Maksimum deneme sayısı (default: 3).
        delay: İlk gecikme süresi (saniye, default: 1.0).
        backoff: Gecikme çarpanı (default: 2.0).
        exceptions: Yakalanacak exception tipleri.
        *args: Fonksiyon argümanları.
        **kwargs: Fonksiyon keyword argümanları.

    Returns:
        Fonksiyonun döndürdüğü değer.

    Raises:
        Son denemede hata devam ederse, exception fırlatılır.
    """
    import time

    current_delay = delay
    last_exception = None

    for attempt in range(1, max_attempts + 1):
        try:
            return func(*args, **kwargs)
        except exceptions as e:
            last_exception = e
            if attempt < max_attempts:
                logger.warning(
                    f"Deneme {attempt}/{max_attempts} başarısız: {e}. "
                    f"{current_delay:.2f} saniye sonra tekrar denenecek."
                )
                time.sleep(current_delay)
                current_delay *= backoff
            else:
                logger.error(
                    f"Tüm {max_attempts} deneme başarısız. Son hata: {e}"
                )

    raise last_exception

