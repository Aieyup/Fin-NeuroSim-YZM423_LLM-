"""
Haber ve olay API modülü: Tavily, NewsAPI, GDELT entegrasyonu.

Bu modül, finansal haberler, olaylar ve sentiment verilerini çeker.
"""

import os
from typing import Dict, Any, Optional, List
from datetime import datetime

import httpx
from tavily import TavilyClient

from fin_neurosim.data_sources.rate_limiter import RateLimiter
from fin_neurosim.utils.logging import setup_logger
from fin_neurosim.utils.retry import async_retry
from fin_neurosim.utils.time_utils import calculate_freshness_score

logger = setup_logger(__name__)


class NewsAPI:
    """
    Haber ve olay verisi API istemcisi.

    Tavily, NewsAPI ve GDELT verilerine erişim sağlar.
    """

    def __init__(self, rate_limiter: Optional[RateLimiter] = None):
        """
        NewsAPI'yi başlatır.

        Args:
            rate_limiter: Rate limiter instance'ı (opsiyonel).
        """
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        self.newsapi_key = os.getenv("NEWSAPI_KEY")

        self.rate_limiter = rate_limiter or RateLimiter()
        if rate_limiter:
            self.rate_limiter.add_limit("tavily", max_calls=100, time_window_seconds=60)
            self.rate_limiter.add_limit("newsapi", max_calls=100, time_window_seconds=60)

    async def search_tavily(
        self,
        query: str,
        search_depth: str = "advanced",
        topic: str = "finance",
        max_results: int = 10
    ) -> Dict[str, Any]:
        """
        Tavily AI ile haber ve web araması yapar.

        Args:
            query: Arama sorgusu.
            search_depth: Arama derinliği ("basic" veya "advanced").
            topic: Konu filtresi (default: "finance").
            max_results: Maksimum sonuç sayısı.

        Returns:
            Arama sonuçları dictionary'si.
        """
        if not self.tavily_api_key:
            logger.warning("TAVILY_API_KEY bulunamadı, NewsAPI kullanılacak")
            return await self.search_newsapi(query, max_results)

        await self.rate_limiter.wait_if_needed("tavily")

        try:
            client = TavilyClient(api_key=self.tavily_api_key)
            response = client.search(
                query=query,
                search_depth=search_depth,
                topic=topic,
                max_results=max_results
            )

            results = []
            for item in response.get("results", []):
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "content": item.get("content", ""),
                    "published_date": item.get("published_date"),
                    "score": item.get("score", 0.0)
                })

            return {
                "query": query,
                "source": "tavily",
                "results": results,
                "count": len(results),
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Tavily API hatası: {e}")
            return {"error": str(e), "results": []}

    async def search_newsapi(
        self,
        query: str,
        max_results: int = 10
    ) -> Dict[str, Any]:
        """
        NewsAPI ile haber araması yapar.

        Args:
            query: Arama sorgusu.
            max_results: Maksimum sonuç sayısı.

        Returns:
            Arama sonuçları dictionary'si.
        """
        if not self.newsapi_key:
            logger.warning("NEWSAPI_KEY bulunamadı!")
            return {"error": "API anahtarı bulunamadı", "results": []}

        await self.rate_limiter.wait_if_needed("newsapi")

        try:
            async with httpx.AsyncClient() as client:
                url = "https://newsapi.org/v2/everything"
                params = {
                    "q": query,
                    "sortBy": "publishedAt",
                    "language": "en",
                    "pageSize": min(max_results, 100),
                    "apiKey": self.newsapi_key
                }
                response = await client.get(url, params=params, timeout=10.0)
                response.raise_for_status()
                data = response.json()

                results = []
                for article in data.get("articles", [])[:max_results]:
                    results.append({
                        "title": article.get("title", ""),
                        "url": article.get("url", ""),
                        "content": article.get("description", ""),
                        "published_date": article.get("publishedAt"),
                        "source": article.get("source", {}).get("name", "")
                    })

                return {
                    "query": query,
                    "source": "newsapi",
                    "results": results,
                    "count": len(results),
                    "timestamp": datetime.utcnow().isoformat()
                }
        except Exception as e:
            logger.error(f"NewsAPI hatası: {e}")
            return {"error": str(e), "results": []}

    async def get_live_anomalies(
        self,
        keywords: List[str],
        max_results: int = 20
    ) -> Dict[str, Any]:
        """
        Canlı anomali ve kriz sinyalleri arar.

        Args:
            keywords: Arama anahtar kelimeleri (örn: ["banking crisis", "market crash"]).
            max_results: Maksimum sonuç sayısı.

        Returns:
            Anomali sonuçları dictionary'si.
        """
        query = " OR ".join(keywords)
        query += " finance OR financial OR market OR economy"

        # Tavily öncelikli, yoksa NewsAPI
        if self.tavily_api_key:
            return await self.search_tavily(
                query,
                search_depth="advanced",
                topic="finance",
                max_results=max_results
            )
        else:
            return await self.search_newsapi(query, max_results)

