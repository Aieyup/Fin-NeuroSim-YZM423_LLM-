"""
Circuit Breaker: Çelişki tespiti ve fail-safe mekanizması.

Bu modül, ajanlar arasında çelişki tespit edildiğinde
sistemi güvenli moda alır ve hakem ajan atar.
"""

from typing import List, Dict, Any, Optional
from enum import Enum

from fin_neurosim.schemas.agent_output import AgentOutput, CircuitBreakerEvent
from fin_neurosim.utils.logging import setup_logger

logger = setup_logger(__name__)


class ContradictionType(Enum):
    """Çelişki tipleri."""

    RISK_TECHNICAL = "risk_technical"  # RiskAgent vs TechnicalAgent
    MACRO_SENTIMENT = "macro_sentiment"  # MacroAgent vs SentimentAgent
    MULTIPLE = "multiple"  # Birden fazla çelişki


class CircuitBreaker:
    """
    Circuit Breaker: Çelişki tespiti ve yönetimi.

    Ajanlar arasında çelişki tespit edildiğinde,
    sistemi güvenli moda alır ve hakem ajan atar.
    """

    # Çelişki kuralları: (ajan1, ajan2) -> hakem ajan
    CONTRADICTION_RULES = {
        ("RiskAgent", "TechnicalAgent"): "MacroAgent",
        ("TechnicalAgent", "RiskAgent"): "MacroAgent",
        ("MacroAgent", "SentimentAgent"): "RiskAgent",
        ("SentimentAgent", "MacroAgent"): "RiskAgent",
    }

    # Risk seviyesi eşleştirmesi
    RISK_LEVELS = {
        "critical": 4,
        "high": 3,
        "medium": 2,
        "low": 1
    }

    def check_contradictions(
        self,
        agent_outputs: List[AgentOutput]
    ) -> Optional[CircuitBreakerEvent]:
        """
        Ajan çıktıları arasında çelişki kontrolü yapar.

        Args:
            agent_outputs: Ajan çıktıları listesi.

        Returns:
            Çelişki tespit edildiyse CircuitBreakerEvent, yoksa None.
        """
        if len(agent_outputs) < 2:
            return None  # Çelişki için en az 2 ajan gerekli

        # Ajan çıktılarını dictionary'ye çevir
        agent_dict = {output.agent_id: output for output in agent_outputs}

        # Çelişki kurallarını kontrol et
        contradictions = []
        arbiter = None

        for (agent1, agent2), arbiter_agent in self.CONTRADICTION_RULES.items():
            if agent1 in agent_dict and agent2 in agent_dict:
                output1 = agent_dict[agent1]
                output2 = agent_dict[agent2]

                if self._is_contradictory(output1, output2):
                    contradictions.append((agent1, agent2))
                    if arbiter is None:
                        arbiter = arbiter_agent

        if contradictions:
            conflicting_agents = []
            for agent1, agent2 in contradictions:
                conflicting_agents.extend([agent1, agent2])
            conflicting_agents = list(set(conflicting_agents))

            event = CircuitBreakerEvent(
                event="CONTRADICTION_DETECTED",
                conflicting_agents=conflicting_agents,
                arbiter=arbiter or "SynthesisAgent",
                action="REWEIGHT_AND_REEVALUATE",
                details={
                    "contradiction_count": len(contradictions),
                    "contradiction_types": [c[0] + "_vs_" + c[1] for c in contradictions]
                }
            )

            logger.warning(
                f"Çelişki tespit edildi: {conflicting_agents}. "
                f"Hakem: {arbiter}"
            )

            return event

        return None

    def _is_contradictory(
        self,
        output1: AgentOutput,
        output2: AgentOutput
    ) -> bool:
        """
        İki ajan çıktısının çelişkili olup olmadığını kontrol eder.

        Args:
            output1: İlk ajan çıktısı.
            output2: İkinci ajan çıktısı.

        Returns:
            Çelişkili ise True, değilse False.
        """
        # Risk seviyelerini sayısal değerlere çevir
        risk1 = self.RISK_LEVELS.get(output1.risk_level.lower(), 2)
        risk2 = self.RISK_LEVELS.get(output2.risk_level.lower(), 2)

        # Risk seviyeleri arasındaki fark çok büyükse çelişki var
        risk_diff = abs(risk1 - risk2)

        # Örnek: Biri "low" (1), diğeri "critical" (4) -> çelişki
        if risk_diff >= 3:
            return True

        # Key drivers'da zıt kelimeler varsa çelişki
        drivers1 = " ".join(output1.key_drivers).lower()
        drivers2 = " ".join(output2.key_drivers).lower()

        # Zıt kelime çiftleri
        opposite_pairs = [
            ("bullish", "bearish"),
            ("rise", "fall"),
            ("growth", "decline"),
            ("positive", "negative"),
            ("buy", "sell"),
            ("crash", "rally")
        ]

        for word1, word2 in opposite_pairs:
            if (word1 in drivers1 and word2 in drivers2) or \
               (word2 in drivers1 and word1 in drivers2):
                return True

        return False

    def get_arbiter_recommendation(
        self,
        event: CircuitBreakerEvent,
        agent_outputs: List[AgentOutput]
    ) -> Dict[str, Any]:
        """
        Hakem ajan için öneri üretir.

        Args:
            event: Circuit breaker event'i.
            agent_outputs: Tüm ajan çıktıları.

        Returns:
            Hakem ajan için öneri dictionary'si.
        """
        conflicting_ids = event.conflicting_agents
        conflicting_outputs = [
            output for output in agent_outputs
            if output.agent_id in conflicting_ids
        ]

        recommendation = {
            "arbiter": event.arbiter,
            "conflicting_agents": conflicting_ids,
            "conflicting_outputs": [
                {
                    "agent_id": output.agent_id,
                    "risk_level": output.risk_level,
                    "confidence": output.confidence,
                    "key_drivers": output.key_drivers
                }
                for output in conflicting_outputs
            ],
            "instruction": (
                f"{event.arbiter}, aşağıdaki çelişkili ajan çıktılarını "
                f"değerlendir ve nihai kararı ver. Çelişki: "
                f"{', '.join(conflicting_ids)}"
            )
        }

        return recommendation

