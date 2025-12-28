"""
Orchestrator: Ana async event-loop ve sistem koordinatörü.

Bu modül, tüm sistemin koordinasyonunu sağlar ve
kullanıcı sorgusundan nihai rapora kadar tüm süreci yönetir.
"""

import asyncio
from typing import Dict, Any, Optional

from fin_neurosim.core.intent_engine import IntentClassifier
from fin_neurosim.core.context_builder import ContextBuilder
from fin_neurosim.core.confidence_engine import ConfidenceEngine
from fin_neurosim.core.circuit_breaker import CircuitBreaker
from fin_neurosim.schemas.final_report import FinalReport
from fin_neurosim.utils.logging import setup_logger

logger = setup_logger(__name__)


class FinNeuroSimOrchestrator:
    """
    Fin-NeuroSim 2.0 Ana Orkestratör.

    Kullanıcı sorgusunu alır, tüm ajanları koordine eder
    ve nihai raporu üretir.
    """

    def __init__(self):
        """Orchestrator'ı başlatır."""
        self.intent_classifier = IntentClassifier()
        self.context_builder = ContextBuilder()
        self.confidence_engine = ConfidenceEngine()
        self.circuit_breaker = CircuitBreaker()

        # Ajanlar lazy import ile yüklenecek
        self.agents = None

    async def process_query(self, user_query: str) -> FinalReport:
        """
        Kullanıcı sorgusunu işler ve nihai raporu döndürür.

        Args:
            user_query: Kullanıcının sorgusu.

        Returns:
            Nihai risk analizi raporu.
        """
        logger.info(f"Sorgu işleniyor: {user_query}")

        # 1. Intent Decomposition
        intent = await self.intent_classifier.analyze(user_query)
        logger.info(f"Intent analizi tamamlandı: {intent.analysis_type} - {intent.assets}")

        # 2. Semantic Context Cloud oluştur
        context = await self.context_builder.build_context(intent, user_query)
        logger.info("Context Cloud oluşturuldu")

        # 3. Ajanları yükle (lazy import)
        if self.agents is None:
            self.agents = await self._load_agents()

        # 4. Stage-1: Minority-Focused Agents (Paralel)
        agent_outputs = await self._run_stage1_agents(context, intent)
        logger.info(f"Stage-1 tamamlandı: {len(agent_outputs)} ajan çıktısı")

        # 5. Circuit Breaker kontrolü
        circuit_event = self.circuit_breaker.check_contradictions(agent_outputs)
        if circuit_event:
            logger.warning(f"Circuit Breaker aktif: {circuit_event.event}")
            # Hakem ajan çalıştır
            arbiter_recommendation = self.circuit_breaker.get_arbiter_recommendation(
                circuit_event, agent_outputs
            )
            # Hakem ajan çıktısını ekle
            # (Bu kısım ajan implementasyonuna bağlı)

        # 6. Dynamic Confidence Computation
        confidence_map = self.confidence_engine.compute_dynamic_confidence(
            agent_outputs,
            freshness_factor=0.95,
            context_freshness=context.get("data_freshness", {})
        )
        logger.info(f"Güven skorları hesaplandı: {confidence_map}")

        # 7. Stage-2: Bayesian Synthesis
        final_report = await self._run_stage2_synthesis(
            agent_outputs,
            confidence_map,
            context,
            user_query
        )
        logger.info(f"Nihai rapor üretildi: Risk seviyesi = {final_report.final_risk_level}")

        return final_report

    async def _load_agents(self) -> Dict[str, Any]:
        """
        Ajanları lazy import ile yükler.

        Returns:
            Ajan dictionary'si.
        """
        from fin_neurosim.agents.risk_agent import RiskAgent
        from fin_neurosim.agents.macro_agent import MacroAgent
        from fin_neurosim.agents.technical_agent import TechnicalAgent
        from fin_neurosim.agents.sentiment_agent import SentimentAgent
        from fin_neurosim.agents.synthesis_agent import SynthesisAgent

        agents = {
            "RiskAgent": RiskAgent(),
            "MacroAgent": MacroAgent(),
            "TechnicalAgent": TechnicalAgent(),
            "SentimentAgent": SentimentAgent(),
            "SynthesisAgent": SynthesisAgent()
        }

        logger.info(f"{len(agents)} ajan yüklendi")
        return agents

    async def _run_stage1_agents(
        self,
        context: Dict[str, Any],
        intent: Any
    ) -> list:
        """
        Stage-1 ajanlarını paralel çalıştırır.

        Args:
            context: Context Cloud.
            intent: Kullanıcı intent'i.

        Returns:
            Ajan çıktıları listesi.
        """
        if self.agents is None:
            self.agents = await self._load_agents()

        # Stage-1 ajanları (SynthesisAgent hariç)
        stage1_agents = [
            self.agents["RiskAgent"],
            self.agents["MacroAgent"],
            self.agents["TechnicalAgent"],
            self.agents["SentimentAgent"]
        ]

        # Paralel çalıştır
        tasks = [
            agent.analyze(context, intent)
            for agent in stage1_agents
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Hataları filtrele
        agent_outputs = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Ajan hatası: {result}")
            else:
                agent_outputs.append(result)

        return agent_outputs

    async def _run_stage2_synthesis(
        self,
        agent_outputs: list,
        confidence_map: Dict[str, float],
        context: Dict[str, Any],
        user_query: str
    ) -> FinalReport:
        """
        Stage-2 synthesis ajanını çalıştırır.

        Args:
            agent_outputs: Stage-1 ajan çıktıları.
            confidence_map: Güven skorları.
            context: Context Cloud.
            user_query: Orijinal kullanıcı sorgusu.

        Returns:
            Nihai rapor.
        """
        if self.agents is None:
            self.agents = await self._load_agents()

        synthesis_agent = self.agents["SynthesisAgent"]

        final_report = await synthesis_agent.synthesize(
            agent_outputs,
            confidence_map,
            context,
            user_query
        )

        return final_report

