"""
Piyasa verisi API modülü: Alpha Vantage ve Polygon entegrasyonu.

Bu modül, canlı piyasa verilerini (hisse senedi fiyatları, VIX, vb.)
çeker.
"""

import os
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime

import httpx
from alpha_vantage.timeseries import TimeSeries

from fin_neurosim.data_sources.rate_limiter import RateLimiter
from fin_neurosim.utils.logging import setup_logger
from fin_neurosim.utils.retry import async_retry
from fin_neurosim.utils.time_utils import calculate_freshness_score

logger = setup_logger(__name__)


class MarketAPI:
    """
    Piyasa verisi API istemcisi.

    Alpha Vantage ve diğer piyasa verisi sağlayıcılarıyla entegrasyon.
    """

    def __init__(self, rate_limiter: Optional[RateLimiter] = None):
        """
        MarketAPI'yi başlatır.

        Args:
            rate_limiter: Rate limiter instance'ı (opsiyonel).
        """
        self.api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        if not self.api_key:
            logger.warning("ALPHA_VANTAGE_API_KEY bulunamadı!")

        self.rate_limiter = rate_limiter or RateLimiter()
        if rate_limiter:
            self.rate_limiter.add_limit("alpha_vantage", max_calls=5, time_window_seconds=60)

    async def get_intraday_data(
        self,
        symbol: str,
        interval: str = "5min"
    ) -> Dict[str, Any]:
        """
        Gün içi (intraday) piyasa verisi çeker.

        Args:
            symbol: Hisse senedi sembolü (örn: "TSLA", "AAPL").
            interval: Veri aralığı (örn: "1min", "5min", "15min", "60min").

        Returns:
            Piyasa verisi dictionary'si.
        """
        if not self.api_key:
            logger.error("Alpha Vantage API anahtarı bulunamadı!")
            return {"error": "API anahtarı bulunamadı"}

        await self.rate_limiter.wait_if_needed("alpha_vantage")

        try:
            ts = TimeSeries(key=self.api_key, output_format='json')
            data, meta_data = await asyncio.to_thread(
                ts.get_intraday,
                symbol=symbol,
                interval=interval,
                outputsize='compact'
            )

            return {
                "symbol": symbol,
                "interval": interval,
                "data": data,
                "metadata": meta_data,
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            error_msg = str(e)
            logger.warning(f"Alpha Vantage API hatası: {error_msg}")
            
            # Premium endpoint hatası için fallback
            if "premium" in error_msg.lower():
                logger.info("Premium endpoint gerekli, basit quote kullanılıyor...")
                try:
                    return await self.get_quote(symbol)
                except Exception as e2:
                    logger.error(f"Quote API de başarısız: {e2}")
            
            return {"error": error_msg, "symbol": symbol}

    async def get_vix_data(self) -> Dict[str, Any]:
        """
        VIX (Volatility Index) verisi çeker.

        Returns:
            VIX verisi dictionary'si.
        """
        # VIX için özel endpoint veya genel market verisi kullanılabilir
        # Bu örnek implementasyon için basitleştirilmiş
        return await self.get_intraday_data("VIX", interval="60min")

    async def get_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Anlık fiyat teklifi (quote) çeker.

        Args:
            symbol: Hisse senedi sembolü.

        Returns:
            Quote verisi dictionary'si.
        """
        if not self.api_key:
            return {"error": "API anahtarı bulunamadı"}

        await self.rate_limiter.wait_if_needed("alpha_vantage")

        try:
            async with httpx.AsyncClient() as client:
                url = "https://www.alphavantage.co/query"
                params = {
                    "function": "GLOBAL_QUOTE",
                    "symbol": symbol,
                    "apikey": self.api_key
                }
                response = await client.get(url, params=params, timeout=10.0)
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Quote API hatası: {e}")
            return {"error": str(e)}

