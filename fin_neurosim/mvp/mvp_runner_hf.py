"""
MVP Runner (HuggingFace): Basitleştirilmiş sürüm - HF uyumlu.

MVP, tam sistemin "ruhunu" koruyarak en az parça ile çalışır.
HuggingFace modelleri ile sequential loading kullanır.
"""

import asyncio
from typing import Dict, Any

from fin_neurosim.core.intent_engine import IntentClassifier
from fin_neurosim.core.context_builder import ContextBuilder
from fin_neurosim.core.context_compressor import ContextCompressor
from fin_neurosim.core.confidence_engine import ConfidenceEngine
from fin_neurosim.llm.model_loader import ModelLoader
from fin_neurosim.llm.stage1_mistral import Stage1Mistral
from fin_neurosim.llm.stage2_llama import Stage2Llama
from fin_neurosim.schemas.final_report import FinalReport
from fin_neurosim.utils.logging import setup_logger

logger = setup_logger(__name__)


class MVPRunnerHF:
    """
    MVP Runner (HuggingFace): Basitleştirilmiş sistem sürümü.

    HuggingFace modelleri ile VRAM-safe çalışır.
    """

    def __init__(self):
        """MVPRunnerHF'ı başlatır."""
        self.intent_classifier = IntentClassifier()
        self.context_builder = ContextBuilder()
        self.context_compressor = ContextCompressor()
        self.confidence_engine = ConfidenceEngine()
        self.model_loader = ModelLoader()
        self.stage1_mistral = Stage1Mistral(self.model_loader)
        self.stage2_llama = Stage2Llama(self.model_loader)

    async def run(self, user_query: str) -> FinalReport:
        """
        MVP akışını çalıştırır (HuggingFace modelleri ile).

        Args:
            user_query: Kullanıcı sorgusu.

        Returns:
            Nihai rapor.
        """
        logger.info(f"MVP (HF) çalıştırılıyor: {user_query}")

        try:
            # 1. Intent Analysis (CPU)
            intent = await self.intent_classifier.analyze(user_query)
            logger.info(f"Intent: {intent.analysis_type} - {intent.assets}")

            # 2. Context Builder (Tavily + Alpha Vantage)
            raw_data = await self.context_builder.build_context(intent, user_query)
            logger.info("Context oluşturuldu")

            # 3. Context Compression (FinBERT)
            compact_context = self.context_compressor.compress(
                raw_data,
                user_query,
                intent
            )
            logger.info(f"Context sıkıştırıldı: ~{len(compact_context.split())} kelime")

            # 4. Stage-1: Minority Focus (Mistral-7B)
            logger.info("Stage-1 başlatılıyor (Mistral-7B)...")
            stage1_report = await self.stage1_mistral.analyze(compact_context, intent)
            logger.info(f"Stage-1: {stage1_report.risk_level}")

            # Modeli kaldır
            self.model_loader.unload_model()

            # 5. Basit confidence hesaplama
            confidence_map = self.confidence_engine.compute_dynamic_confidence(
                [stage1_report],
                freshness_factor=0.95,
                context_freshness=raw_data.get("data_freshness", {})
            )

            # 6. Stage-2: Synthesis (Llama-3-8B)
            logger.info("Stage-2 başlatılıyor (Llama-3-8B)...")
            final_report = await self.stage2_llama.synthesize(
                stage1_report,
                confidence_map,
                compact_context[:500],
                user_query
            )

            # Modeli kaldır
            self.model_loader.unload_model()

            logger.info(f"MVP (HF) tamamlandı: Risk seviyesi = {final_report.final_risk_level}")
            return final_report

        except Exception as e:
            logger.error(f"MVP (HF) hatası: {e}", exc_info=True)
            self.model_loader.unload_model()
            raise

