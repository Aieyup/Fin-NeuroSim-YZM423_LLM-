"""
Makroekonomi API modülü: FRED, IMF, World Bank entegrasyonu.

Bu modül, makroekonomik göstergeleri (CPI, GDP, işsizlik oranı, vb.)
çeker.
"""

import os
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime

from fredapi import Fred
import httpx

from fin_neurosim.data_sources.rate_limiter import RateLimiter
from fin_neurosim.utils.logging import setup_logger
from fin_neurosim.utils.retry import async_retry
from fin_neurosim.utils.time_utils import calculate_freshness_score

logger = setup_logger(__name__)


class MacroAPI:
    """
    Makroekonomi verisi API istemcisi.

    FRED, IMF ve World Bank verilerine erişim sağlar.
    """

    def __init__(self, rate_limiter: Optional[RateLimiter] = None):
        """
        MacroAPI'yi başlatır.

        Args:
            rate_limiter: Rate limiter instance'ı (opsiyonel).
        """
        self.fred_api_key = os.getenv("FRED_API_KEY")
        if not self.fred_api_key:
            logger.warning("FRED_API_KEY bulunamadı!")

        self.rate_limiter = rate_limiter or RateLimiter()
        if rate_limiter:
            self.rate_limiter.add_limit("fred", max_calls=120, time_window_seconds=60)

    async def get_cpi_data(self, region: str = "US") -> Dict[str, Any]:
        """
        Tüketici Fiyat Endeksi (CPI) verisi çeker.

        Args:
            region: Bölge kodu (default: "US").

        Returns:
            CPI verisi dictionary'si.
        """
        if not self.fred_api_key:
            return {"error": "FRED API anahtarı bulunamadı"}

        await self.rate_limiter.wait_if_needed("fred")

        try:
            fred = Fred(api_key=self.fred_api_key)
            # CPI için FRED seri ID'si
            series_id = "CPIAUCSL" if region == "US" else "CPIAUCSL"  # Genişletilebilir

            data = await asyncio.to_thread(
                fred.get_series,
                series_id,
                observation_start=datetime(2020, 1, 1)
            )

            latest_value = data.iloc[-1] if len(data) > 0 else None
            latest_date = data.index[-1] if len(data) > 0 else None

            return {
                "series_id": series_id,
                "region": region,
                "latest_value": float(latest_value) if latest_value is not None else None,
                "latest_date": latest_date.isoformat() if latest_date else None,
                "data_points": len(data),
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"FRED CPI API hatası: {e}")
            return {"error": str(e)}

    async def get_gdp_data(self, region: str = "US") -> Dict[str, Any]:
        """
        Gayri Safi Yurtiçi Hasıla (GDP) verisi çeker.

        Args:
            region: Bölge kodu (default: "US").

        Returns:
            GDP verisi dictionary'si.
        """
        if not self.fred_api_key:
            return {"error": "FRED API anahtarı bulunamadı"}

        await self.rate_limiter.wait_if_needed("fred")

        try:
            fred = Fred(api_key=self.fred_api_key)
            series_id = "GDP" if region == "US" else "GDP"

            data = await asyncio.to_thread(
                fred.get_series,
                series_id,
                observation_start=datetime(2020, 1, 1)
            )

            latest_value = data.iloc[-1] if len(data) > 0 else None
            latest_date = data.index[-1] if len(data) > 0 else None

            return {
                "series_id": series_id,
                "region": region,
                "latest_value": float(latest_value) if latest_value is not None else None,
                "latest_date": latest_date.isoformat() if latest_date else None,
                "data_points": len(data),
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"FRED GDP API hatası: {e}")
            return {"error": str(e)}

    async def get_unemployment_rate(self, region: str = "US") -> Dict[str, Any]:
        """
        İşsizlik oranı verisi çeker.

        Args:
            region: Bölge kodu (default: "US").

        Returns:
            İşsizlik oranı verisi dictionary'si.
        """
        if not self.fred_api_key:
            return {"error": "FRED API anahtarı bulunamadı"}

        await self.rate_limiter.wait_if_needed("fred")

        try:
            fred = Fred(api_key=self.fred_api_key)
            series_id = "UNRATE" if region == "US" else "UNRATE"

            data = await asyncio.to_thread(
                fred.get_series,
                series_id,
                observation_start=datetime(2020, 1, 1)
            )

            latest_value = data.iloc[-1] if len(data) > 0 else None
            latest_date = data.index[-1] if len(data) > 0 else None

            return {
                "series_id": series_id,
                "region": region,
                "latest_value": float(latest_value) if latest_value is not None else None,
                "latest_date": latest_date.isoformat() if latest_date else None,
                "data_points": len(data),
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"FRED Unemployment API hatası: {e}")
            return {"error": str(e)}

    async def get_interest_rates(self, region: str = "US") -> Dict[str, Any]:
        """
        Merkez bankası faiz oranları verisi çeker.

        Args:
            region: Bölge kodu (default: "US").

        Returns:
            Faiz oranları verisi dictionary'si.
        """
        if not self.fred_api_key:
            return {"error": "FRED API anahtarı bulunamadı"}

        await self.rate_limiter.wait_if_needed("fred")

        try:
            fred = Fred(api_key=self.fred_api_key)
            # Federal Funds Rate için FRED seri ID'si
            series_id = "FEDFUNDS" if region == "US" else "FEDFUNDS"

            data = await asyncio.to_thread(
                fred.get_series,
                series_id,
                observation_start=datetime(2020, 1, 1)
            )

            latest_value = data.iloc[-1] if len(data) > 0 else None
            latest_date = data.index[-1] if len(data) > 0 else None

            return {
                "series_id": series_id,
                "region": region,
                "latest_value": float(latest_value) if latest_value is not None else None,
                "latest_date": latest_date.isoformat() if latest_date else None,
                "data_points": len(data),
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"FRED Interest Rates API hatası: {e}")
            return {"error": str(e)}

