"""
Context Compressor: FinBERT ile semantic context sıkıştırma.

Bu modül, ham veriyi FinBERT embedding'leri kullanarak
sıkıştırır ve 2000 token limitine uygun hale getirir.
"""

from typing import Dict, Any, List, Tuple
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠️  sentence-transformers yüklü değil, basit compression kullanılacak")

from fin_neurosim.utils.logging import setup_logger

logger = setup_logger(__name__)


class ContextCompressor:
    """
    Context sıkıştırıcı: FinBERT ile semantic filtering.

    Ham veriyi alır, relevance scoring yapar ve
    en önemli parçaları seçer (Top-K).
    """

    MODEL_NAME = "ProsusAI/finbert"
    TOP_K = 8  # En fazla 8 parça seçilir
    TARGET_TOKENS = 1200  # Hedef token sayısı

    def __init__(self):
        """ContextCompressor'ı başlatır."""
        self.embedding_model = None
        self._load_embedding_model()

    def _load_embedding_model(self) -> None:
        """
        FinBERT embedding modelini yükler.

        Model CPU'da çalışır, VRAM kullanmaz.
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.warning("sentence-transformers yüklü değil, basit compression kullanılacak")
            self.embedding_model = None
            return
            
        try:
            logger.info(f"FinBERT modeli yükleniyor: {self.MODEL_NAME}")
            self.embedding_model = SentenceTransformer(self.MODEL_NAME)
            logger.info("FinBERT modeli başarıyla yüklendi")
        except Exception as e:
            logger.error(f"FinBERT modeli yüklenemedi: {e}")
            # Fallback: basit text-based compression
            self.embedding_model = None

    def compress(
        self,
        raw_data: Dict[str, Any],
        query: str,
        intent: Any
    ) -> str:
        """
        Ham veriyi sıkıştırır ve compact context üretir.

        Args:
            raw_data: Ham veri (API'lerden gelen).
            query: Kullanıcı sorgusu.
            intent: Kullanıcı intent'i.

        Returns:
            Sıkıştırılmış context string'i (≤1200 token).
        """
        logger.info("Context sıkıştırma başlatıldı")

        # Veriyi parçalara ayır
        text_chunks = self._extract_text_chunks(raw_data)

        if not text_chunks:
            logger.warning("Sıkıştırılacak metin bulunamadı")
            return self._create_fallback_context(raw_data, query)

        # Query embedding
        query_embedding = self._get_query_embedding(query, intent)

        # Relevance scoring
        scored_chunks = self._score_relevance(text_chunks, query_embedding)

        # Top-K seçimi
        selected_chunks = self._select_top_k(scored_chunks, self.TOP_K)

        # Compact context oluştur
        compact_context = self._build_compact_context(selected_chunks, raw_data)

        # Token sayısını kontrol et
        estimated_tokens = len(compact_context.split()) * 1.3  # Yaklaşık token hesabı
        if estimated_tokens > self.TARGET_TOKENS:
            logger.warning(
                f"Context hala büyük: ~{estimated_tokens:.0f} token. "
                "Ek kısaltma yapılıyor..."
            )
            compact_context = self._further_compress(compact_context, self.TARGET_TOKENS)

        logger.info(f"Context sıkıştırıldı: ~{len(compact_context.split())} kelime")
        return compact_context

    def _extract_text_chunks(self, raw_data: Dict[str, Any]) -> List[str]:
        """
        Ham veriden metin parçalarını çıkarır.

        Args:
            raw_data: Ham veri.

        Returns:
            Metin parçaları listesi.
        """
        chunks = []

        # Haber verisi
        news_data = raw_data.get("news_data", {})
        if news_data.get("results"):
            for item in news_data["results"]:
                title = item.get("title", "")
                content = item.get("content", "")
                if title or content:
                    chunks.append(f"{title}\n{content[:500]}")  # İlk 500 karakter

        # Piyasa verisi (text summary)
        market_data = raw_data.get("market_data", [])
        for item in market_data:
            if isinstance(item, dict):
                summary = f"Symbol: {item.get('symbol', 'N/A')}, "
                summary += f"Data points: {len(item.get('data', {}))}"
                chunks.append(summary)

        # Makroekonomi verisi
        macro_data = raw_data.get("macro_data", {})
        if macro_data:
            macro_summary = []
            for key, value in macro_data.items():
                if isinstance(value, dict) and "latest_value" in value:
                    macro_summary.append(
                        f"{key}: {value.get('latest_value', 'N/A')}"
                    )
            if macro_summary:
                chunks.append("\n".join(macro_summary))

        return chunks

    def _get_query_embedding(self, query: str, intent: Any) -> np.ndarray:
        """
        Query için embedding oluşturur.

        Args:
            query: Kullanıcı sorgusu.
            intent: Kullanıcı intent'i.

        Returns:
            Query embedding vector.
        """
        # Query'yi zenginleştir
        enriched_query = f"{query} "
        enriched_query += f"Analysis type: {intent.analysis_type}. "
        enriched_query += f"Assets: {', '.join(intent.assets)}. "
        enriched_query += f"Region: {intent.region}."

        if self.embedding_model:
            try:
                embedding = self.embedding_model.encode(enriched_query)
                return embedding
            except Exception as e:
                logger.warning(f"Embedding oluşturulamadı: {e}")

        # Fallback: sıfır vektör
        return np.zeros(768)  # FinBERT embedding boyutu

    def _score_relevance(
        self,
        text_chunks: List[str],
        query_embedding: np.ndarray
    ) -> List[Tuple[str, float]]:
        """
        Metin parçalarını relevance skoruna göre sıralar.

        Args:
            text_chunks: Metin parçaları.
            query_embedding: Query embedding.

        Returns:
            (chunk, score) tuple'ları listesi.
        """
        if not self.embedding_model or len(text_chunks) == 0:
            # Fallback: basit scoring
            return [(chunk, 0.5) for chunk in text_chunks]

        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            # Fallback: basit scoring
            return [(chunk, 0.5) for chunk in text_chunks]

        try:
            # Chunk embedding'leri
            chunk_embeddings = self.embedding_model.encode(text_chunks)

            # Cosine similarity
            query_embedding_2d = query_embedding.reshape(1, -1)
            similarities = cosine_similarity(query_embedding_2d, chunk_embeddings)[0]

            # Score'ları tuple'lara çevir
            scored_chunks = list(zip(text_chunks, similarities))
            # Score'a göre sırala (yüksekten düşüğe)
            scored_chunks.sort(key=lambda x: x[1], reverse=True)

            return scored_chunks

        except Exception as e:
            logger.warning(f"Relevance scoring hatası: {e}")
            # Fallback: eşit skor
            return [(chunk, 0.5) for chunk in text_chunks]

    def _select_top_k(
        self,
        scored_chunks: List[Tuple[str, float]],
        k: int
    ) -> List[str]:
        """
        Top-K parçayı seçer.

        Args:
            scored_chunks: Skorlanmış parçalar.
            k: Seçilecek parça sayısı.

        Returns:
            Seçilmiş parçalar listesi.
        """
        selected = [chunk for chunk, score in scored_chunks[:k]]
        logger.info(f"Top-{k} parça seçildi (skorlar: {[f'{s:.3f}' for _, s in scored_chunks[:k]]})")
        return selected

    def _build_compact_context(
        self,
        selected_chunks: List[str],
        raw_data: Dict[str, Any]
    ) -> str:
        """
        Seçilmiş parçalardan compact context oluşturur.

        Args:
            selected_chunks: Seçilmiş metin parçaları.
            raw_data: Ham veri (metadata için).

        Returns:
            Compact context string'i.
        """
        context_parts = []

        # Başlık
        context_parts.append("=== COMPACT CONTEXT (FinBERT Filtered) ===\n")

        # Seçilmiş parçalar
        for i, chunk in enumerate(selected_chunks, 1):
            context_parts.append(f"[Chunk {i}]\n{chunk}\n")

        # Metadata (kısa)
        context_parts.append("\n[METADATA]")
        context_parts.append(f"Data freshness: {raw_data.get('data_freshness', {}).get('overall', 0.5):.2f}")
        context_parts.append(f"Timestamp: {raw_data.get('timestamp', 'N/A')}")

        return "\n".join(context_parts)

    def _further_compress(self, context: str, target_tokens: int) -> str:
        """
        Context'i daha da kısaltır.

        Args:
            context: Orijinal context.
            target_tokens: Hedef token sayısı.

        Returns:
            Daha da kısaltılmış context.
        """
        # Basit kelime sayısına göre kısaltma
        words = context.split()
        target_words = int(target_tokens / 1.3)  # Token -> kelime yaklaşımı

        if len(words) <= target_words:
            return context

        # İlk N kelimeyi al
        truncated_words = words[:target_words]
        return " ".join(truncated_words) + "..."

    def _create_fallback_context(
        self,
        raw_data: Dict[str, Any],
        query: str
    ) -> str:
        """
        Fallback context oluşturur (sıkıştırma başarısız olursa).

        Args:
            raw_data: Ham veri.
            query: Kullanıcı sorgusu.

        Returns:
            Basit context string'i.
        """
        context = f"Query: {query}\n\n"
        context += "Data summary:\n"
        context += f"- News items: {len(raw_data.get('news_data', {}).get('results', []))}\n"
        context += f"- Market data points: {len(raw_data.get('market_data', []))}\n"
        context += f"- Macro indicators: {len(raw_data.get('macro_data', {}))}\n"
        return context

