"""
Intent Engine: Kullanıcı sorgusunu analiz eder ve intent'e çevirir.

Bu modül, kullanıcının sorgusunu analiz ederek hangi varlıklar,
bölgeler ve analiz tipleri istediğini çıkarır.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from fin_neurosim.utils.logging import setup_logger

logger = setup_logger(__name__)


@dataclass
class Intent:
    """
    Kullanıcı intent'i: Sorgunun analiz edilmiş hali.

    Attributes:
        assets: Analiz edilecek varlıklar (hisse senetleri, vb.).
        region: Bölge (örn: "US", "EU", "TR").
        analysis_type: Analiz tipi (örn: "risk", "trend", "sentiment").
        time_horizon: Zaman ufku (örn: "short", "medium", "long").
        keywords: Önemli anahtar kelimeler.
    """

    assets: List[str]
    region: str
    analysis_type: str
    time_horizon: str
    keywords: List[str]


class IntentClassifier:
    """
    Intent sınıflandırıcı: Kullanıcı sorgusunu Intent nesnesine çevirir.

    Bu basit implementasyon, daha gelişmiş NLP modelleriyle
    (örn: LLM-based intent classification) değiştirilebilir.
    """

    # Varlık sembolleri için basit pattern'ler
    ASSET_PATTERNS = [
        r'\b[A-Z]{1,5}\b',  # Borsa sembolleri (TSLA, AAPL, vb.)
        r'\b\w+\s+stock\b',  # "Apple stock" gibi
        r'\b\w+\s+shares?\b',  # "Tesla shares" gibi
    ]

    # Bölge kodları
    REGIONS = {
        "US": ["usa", "united states", "america", "us"],
        "EU": ["europe", "eurozone", "eu", "european"],
        "TR": ["turkey", "türkiye", "turkiye", "tr"],
        "UK": ["united kingdom", "britain", "uk", "england"],
        "CN": ["china", "chinese", "cn"],
    }

    # Analiz tipleri
    ANALYSIS_TYPES = {
        "risk": ["risk", "danger", "crisis", "crash", "collapse", "anomaly"],
        "trend": ["trend", "direction", "forecast", "prediction", "outlook"],
        "sentiment": ["sentiment", "feeling", "mood", "emotion", "panic"],
        "technical": ["technical", "chart", "indicator", "volatility"],
    }

    @classmethod
    async def analyze(cls, user_query: str) -> Intent:
        """
        Kullanıcı sorgusunu analiz eder ve Intent nesnesi döndürür.

        Args:
            user_query: Kullanıcının sorgusu.

        Returns:
            Analiz edilmiş Intent nesnesi.
        """
        query_lower = user_query.lower()

        # Varlıkları çıkar
        assets = cls._extract_assets(user_query)

        # Bölgeyi çıkar
        region = cls._extract_region(query_lower)

        # Analiz tipini çıkar
        analysis_type = cls._extract_analysis_type(query_lower)

        # Zaman ufkunu çıkar
        time_horizon = cls._extract_time_horizon(query_lower)

        # Anahtar kelimeleri çıkar
        keywords = cls._extract_keywords(user_query)

        intent = Intent(
            assets=assets,
            region=region,
            analysis_type=analysis_type,
            time_horizon=time_horizon,
            keywords=keywords
        )

        logger.info(f"Intent analizi tamamlandı: {intent}")
        return intent

    @classmethod
    def _extract_assets(cls, query: str) -> List[str]:
        """
        Sorgudan varlık sembollerini çıkarır.

        Args:
            query: Kullanıcı sorgusu.

        Returns:
            Varlık sembolleri listesi.
        """
        import re
        assets = []

        # Borsa sembolleri için pattern
        pattern = r'\b([A-Z]{1,5})\b'
        matches = re.findall(pattern, query)
        assets.extend(matches)

        # "stock" veya "shares" içeren ifadeler
        stock_pattern = r'(\w+)\s+(?:stock|shares?)'
        stock_matches = re.findall(stock_pattern, query, re.IGNORECASE)
        assets.extend([s.capitalize() for s in stock_matches])

        # Tekrarları kaldır ve boş olanları filtrele
        assets = list(set([a for a in assets if len(a) > 1]))

        return assets if assets else ["GENERAL_MARKET"]

    @classmethod
    def _extract_region(cls, query_lower: str) -> str:
        """
        Sorgudan bölge bilgisini çıkarır.

        Args:
            query_lower: Küçük harfe çevrilmiş sorgu.

        Returns:
            Bölge kodu (default: "US").
        """
        for region_code, keywords in cls.REGIONS.items():
            if any(keyword in query_lower for keyword in keywords):
                return region_code
        return "US"  # Default

    @classmethod
    def _extract_analysis_type(cls, query_lower: str) -> str:
        """
        Sorgudan analiz tipini çıkarır.

        Args:
            query_lower: Küçük harfe çevrilmiş sorgu.

        Returns:
            Analiz tipi (default: "risk").
        """
        for analysis_type, keywords in cls.ANALYSIS_TYPES.items():
            if any(keyword in query_lower for keyword in keywords):
                return analysis_type
        return "risk"  # Default

    @classmethod
    def _extract_time_horizon(cls, query_lower: str) -> str:
        """
        Sorgudan zaman ufkunu çıkarır.

        Args:
            query_lower: Küçük harfe çevrilmiş sorgu.

        Returns:
            Zaman ufku ("short", "medium", "long", default: "medium").
        """
        if any(word in query_lower for word in ["short", "immediate", "today", "now"]):
            return "short"
        elif any(word in query_lower for word in ["long", "long-term", "years"]):
            return "long"
        else:
            return "medium"

    @classmethod
    def _extract_keywords(cls, query: str) -> List[str]:
        """
        Sorgudan önemli anahtar kelimeleri çıkarır.

        Args:
            query: Kullanıcı sorgusu.

        Returns:
            Anahtar kelimeler listesi.
        """
        # Basit keyword extraction
        # Daha gelişmiş NLP teknikleri kullanılabilir
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "is", "are", "was", "were"}
        words = query.lower().split()
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        return keywords[:10]  # İlk 10 keyword

