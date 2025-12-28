"""
Orchestrator (HuggingFace): Revize edilmiş ana koordinatör.

Bu modül, HuggingFace modelleri ile sequential loading
kullanarak VRAM-safe çalışır.
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


class FinNeuroSimOrchestratorHF:
    """
    Fin-NeuroSim 2.0 HuggingFace Orkestratör.

    Sequential model loading ile VRAM-safe çalışır.
    """

    def __init__(self):
        """Orchestrator'ı başlatır."""
        self.intent_classifier = IntentClassifier()
        self.context_builder = ContextBuilder()
        self.context_compressor = ContextCompressor()
        self.confidence_engine = ConfidenceEngine()
        self.model_loader = ModelLoader()
        self.stage1_mistral = Stage1Mistral(self.model_loader)
        self.stage2_llama = Stage2Llama(self.model_loader)

    async def process_query(self, user_query: str) -> FinalReport:
        """
        Kullanıcı sorgusunu işler ve nihai raporu döndürür.

        VRAM-safe sequential loading kullanır.

        Args:
            user_query: Kullanıcının sorgusu.

        Returns:
            Nihai risk analizi raporu.
        """
        logger.info(f"Sorgu işleniyor (HF): {user_query}")

        try:
            # 1. Intent Decomposition (CPU - GPU kullanmaz)
            intent = await self.intent_classifier.analyze(user_query)
            logger.info(f"Intent analizi tamamlandı: {intent.analysis_type} - {intent.assets}")

            # 2. Live Web & API Harvesting (CPU / Async)
            raw_data = await self.context_builder.build_context(intent, user_query)
            logger.info("Ham veri toplandı")

            # 3. Semantic Context Compression (FinBERT - CPU)
            compact_context = self.context_compressor.compress(
                raw_data,
                user_query,
                intent
            )
            logger.info(f"Context sıkıştırıldı: ~{len(compact_context.split())} kelime")

            # 4. Stage-1: Minority Focus (Mistral-7B)
            logger.info("Stage-1 başlatılıyor (Mistral-7B)...")
            stage1_report = await self.stage1_mistral.analyze(compact_context, intent)
            logger.info(f"Stage-1 tamamlandı: Risk seviyesi = {stage1_report.risk_level}")

            # Stage-1 bittiğinde modeli kaldır
            logger.info("Mistral-7B bellekten kaldırılıyor...")
            self.model_loader.unload_model()

            # 5. Confidence & Reliability Scoring (CPU)
            confidence_map = self.confidence_engine.compute_dynamic_confidence(
                [stage1_report],
                freshness_factor=0.95,
                context_freshness=raw_data.get("data_freshness", {})
            )
            logger.info(f"Güven skorları hesaplandı: {confidence_map}")

            # 6. Stage-2: Synthesis & Decision (Llama-3-8B)
            logger.info("Stage-2 başlatılıyor (Llama-3-8B)...")
            final_report = await self.stage2_llama.synthesize(
                stage1_report,
                confidence_map,
                compact_context[:500],  # Kısa context
                user_query
            )
            logger.info(f"Stage-2 tamamlandı: Risk seviyesi = {final_report.final_risk_level}")

            # Stage-2 bittiğinde modeli kaldır
            logger.info("Llama-3-8B bellekten kaldırılıyor...")
            self.model_loader.unload_model()

            logger.info("Tüm işlem tamamlandı")
            return final_report

        except Exception as e:
            logger.error(f"Orchestrator hatası: {e}", exc_info=True)
            # Hata durumunda modeli temizle
            self.model_loader.unload_model()
            raise

