"""
Context Builder: Semantic Context Cloud oluşturur.

Bu modül, her sorgu için anlık olarak bir "Semantic Context Cloud"
oluşturur. Bu, statik veritabanı yerine dinamik, canlı veri toplama
stratejisidir.
"""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime

from fin_neurosim.core.intent_engine import Intent
from fin_neurosim.data_sources.market_api import MarketAPI
from fin_neurosim.data_sources.macro_api import MacroAPI
from fin_neurosim.data_sources.news_api import NewsAPI
from fin_neurosim.data_sources.rate_limiter import RateLimiter
from fin_neurosim.utils.logging import setup_logger
from fin_neurosim.utils.time_utils import calculate_freshness_score

logger = setup_logger(__name__)


class ContextBuilder:
    """
    Semantic Context Cloud oluşturucu.

    Her sorgu için, çeşitli veri kaynaklarından paralel olarak
    veri toplar ve birleştirir.
    """

    def __init__(self):
        """ContextBuilder'ı başlatır."""
        self.rate_limiter = RateLimiter()
        self.market_api = MarketAPI(self.rate_limiter)
        self.macro_api = MacroAPI(self.rate_limiter)
        self.news_api = NewsAPI(self.rate_limiter)

    async def build_context(self, intent: Intent, user_query: str) -> Dict[str, Any]:
        """
        Intent'e göre Semantic Context Cloud oluşturur.

        Args:
            intent: Analiz edilmiş kullanıcı intent'i.
            user_query: Orijinal kullanıcı sorgusu.

        Returns:
            Context Cloud dictionary'si.
        """
        logger.info(f"Context Cloud oluşturuluyor: {intent}")

        # Paralel veri toplama
        tasks = []

        # Piyasa verisi
        if intent.assets:
            for asset in intent.assets:
                if asset != "GENERAL_MARKET":
                    tasks.append(self._get_market_data(asset))

        # Makroekonomi verisi
        tasks.append(self._get_macro_data(intent.region))

        # Haber ve anomali verisi
        tasks.append(self._get_news_data(intent, user_query))

        # Tüm görevleri paralel çalıştır
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Sonuçları birleştir
        context = {
            "intent": {
                "assets": intent.assets,
                "region": intent.region,
                "analysis_type": intent.analysis_type,
                "time_horizon": intent.time_horizon,
                "keywords": intent.keywords
            },
            "market_data": [],
            "macro_data": {},
            "news_data": {},
            "timestamp": datetime.utcnow().isoformat(),
            "data_freshness": {}
        }

        # Sonuçları kategorize et
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Veri toplama hatası: {result}")
                continue

            if isinstance(result, dict):
                if "symbol" in result or "error" in result:
                    # Piyasa verisi
                    if "error" not in result:
                        context["market_data"].append(result)
                elif "series_id" in result or "region" in result:
                    # Makroekonomi verisi
                    context["macro_data"].update(result)
                elif "results" in result or "query" in result:
                    # Haber verisi
                    context["news_data"] = result

        # Veri tazeliği skorlarını hesapla
        context["data_freshness"] = self._calculate_freshness_scores(context)

        logger.info(f"Context Cloud oluşturuldu: {len(context['market_data'])} piyasa, "
                   f"{len(context['macro_data'])} makro, {len(context.get('news_data', {}).get('results', []))} haber")

        return context

    async def _get_market_data(self, symbol: str) -> Dict[str, Any]:
        """
        Piyasa verisi çeker.

        Args:
            symbol: Varlık sembolü.

        Returns:
            Piyasa verisi dictionary'si.
        """
        try:
            data = await self.market_api.get_intraday_data(symbol, interval="5min")
            return data
        except Exception as e:
            logger.error(f"Piyasa verisi çekilemedi ({symbol}): {e}")
            return {"error": str(e), "symbol": symbol}

    async def _get_macro_data(self, region: str) -> Dict[str, Any]:
        """
        Makroekonomi verisi çeker.

        Args:
            region: Bölge kodu.

        Returns:
            Makroekonomi verisi dictionary'si.
        """
        try:
            cpi, gdp, unemployment, interest = await asyncio.gather(
                self.macro_api.get_cpi_data(region),
                self.macro_api.get_gdp_data(region),
                self.macro_api.get_unemployment_rate(region),
                self.macro_api.get_interest_rates(region),
                return_exceptions=True
            )

            macro_data = {
                "region": region,
                "cpi": cpi if not isinstance(cpi, Exception) else {"error": str(cpi)},
                "gdp": gdp if not isinstance(gdp, Exception) else {"error": str(gdp)},
                "unemployment": unemployment if not isinstance(unemployment, Exception) else {"error": str(unemployment)},
                "interest_rates": interest if not isinstance(interest, Exception) else {"error": str(interest)}
            }

            return macro_data
        except Exception as e:
            logger.error(f"Makroekonomi verisi çekilemedi ({region}): {e}")
            return {"error": str(e), "region": region}

    async def _get_news_data(self, intent: Intent, user_query: str) -> Dict[str, Any]:
        """
        Haber ve anomali verisi çeker.

        Args:
            intent: Kullanıcı intent'i.
            user_query: Orijinal sorgu.

        Returns:
            Haber verisi dictionary'si.
        """
        try:
            # Anomali anahtar kelimeleri
            anomaly_keywords = [
                "crisis", "crash", "collapse", "panic", "anomaly",
                "banking stress", "market crash", "recession"
            ]

            # Kullanıcı sorgusundan ve intent'ten keyword'leri birleştir
            all_keywords = intent.keywords + anomaly_keywords

            news_data = await self.news_api.get_live_anomalies(
                keywords=all_keywords[:5],  # İlk 5 keyword
                max_results=20
            )

            return news_data
        except Exception as e:
            logger.error(f"Haber verisi çekilemedi: {e}")
            return {"error": str(e), "results": []}

    def _calculate_freshness_scores(self, context: Dict[str, Any]) -> Dict[str, float]:
        """
        Context içindeki verilerin tazelik skorlarını hesaplar.

        Args:
            context: Context Cloud dictionary'si.

        Returns:
            Tazelik skorları dictionary'si.
        """
        freshness_scores = {}

        # Piyasa verisi tazeliği
        if context.get("market_data"):
            for market_item in context["market_data"]:
                if "timestamp" in market_item:
                    try:
                        timestamp = datetime.fromisoformat(market_item["timestamp"].replace("Z", "+00:00"))
                        freshness_scores["market"] = calculate_freshness_score(timestamp)
                    except Exception:
                        freshness_scores["market"] = 0.5

        # Makroekonomi verisi tazeliği
        if context.get("macro_data"):
            macro_timestamps = []
            for key, value in context["macro_data"].items():
                if isinstance(value, dict) and "latest_date" in value:
                    try:
                        timestamp = datetime.fromisoformat(value["latest_date"])
                        macro_timestamps.append(timestamp)
                    except Exception:
                        pass
            if macro_timestamps:
                latest_macro = max(macro_timestamps)
                freshness_scores["macro"] = calculate_freshness_score(latest_macro, max_age_hours=168)  # 1 hafta

        # Haber verisi tazeliği
        if context.get("news_data", {}).get("results"):
            news_timestamps = []
            for item in context["news_data"]["results"]:
                if "published_date" in item and item["published_date"]:
                    try:
                        timestamp = datetime.fromisoformat(item["published_date"].replace("Z", "+00:00"))
                        news_timestamps.append(timestamp)
                    except Exception:
                        pass
            if news_timestamps:
                latest_news = max(news_timestamps)
                freshness_scores["news"] = calculate_freshness_score(latest_news, max_age_hours=24)

        # Genel tazelik skoru (ortalama)
        if freshness_scores:
            freshness_scores["overall"] = sum(freshness_scores.values()) / len(freshness_scores)
        else:
            freshness_scores["overall"] = 0.5

        return freshness_scores

