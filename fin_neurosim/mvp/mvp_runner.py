"""
MVP Runner: Basitleştirilmiş sürüm.

MVP, tam sistemin "ruhunu" koruyarak en az parça ile çalışır.
MVP kapsamı:
- Intent Decomposition
- Live News + Market API
- 2 Agent: RiskAgent, SynthesisAgent
- Basit confidence
- Tek çıktı JSON
"""

import asyncio
from typing import Dict, Any

from fin_neurosim.core.intent_engine import IntentClassifier
from fin_neurosim.core.context_builder import ContextBuilder
from fin_neurosim.core.confidence_engine import ConfidenceEngine
from fin_neurosim.agents.risk_agent import RiskAgent
from fin_neurosim.agents.synthesis_agent import SynthesisAgent
from fin_neurosim.schemas.final_report import FinalReport
from fin_neurosim.utils.logging import setup_logger

logger = setup_logger(__name__)


class MVPRunner:
    """
    MVP Runner: Basitleştirilmiş sistem sürümü.

    Tam sistemin temel özelliklerini içerir ancak
    daha az ajan ve basitleştirilmiş akış kullanır.
    """

    def __init__(self):
        """MVPRunner'ı başlatır."""
        self.intent_classifier = IntentClassifier()
        self.context_builder = ContextBuilder()
        self.confidence_engine = ConfidenceEngine()
        self.risk_agent = RiskAgent()
        self.synthesis_agent = SynthesisAgent()

    async def run(self, user_query: str) -> FinalReport:
        """
        MVP akışını çalıştırır.

        Args:
            user_query: Kullanıcı sorgusu.

        Returns:
            Nihai rapor.
        """
        logger.info(f"MVP çalıştırılıyor: {user_query}")

        # 1. Intent Analysis
        intent = await self.intent_classifier.analyze(user_query)
        logger.info(f"Intent: {intent.analysis_type} - {intent.assets}")

        # 2. Context Builder (Tavily + Alpha Vantage)
        context = await self.context_builder.build_context(intent, user_query)
        logger.info("Context oluşturuldu")

        # 3. RiskAgent (Minority Focus)
        risk_output = await self.risk_agent.analyze(context, intent)
        logger.info(f"RiskAgent: {risk_output.risk_level}")

        # 4. Basit confidence hesaplama
        confidence_map = self.confidence_engine.compute_dynamic_confidence(
            [risk_output],
            freshness_factor=0.95,
            context_freshness=context.get("data_freshness", {})
        )

        # 5. SynthesisAgent (Sadece RiskAgent çıktısı ile)
        final_report = await self.synthesis_agent.synthesize(
            [risk_output],
            confidence_map,
            context,
            user_query
        )

        logger.info(f"MVP tamamlandı: Risk seviyesi = {final_report.final_risk_level}")
        return final_report

