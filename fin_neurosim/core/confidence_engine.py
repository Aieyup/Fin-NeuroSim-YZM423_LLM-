"""
Confidence Engine: Dinamik güven skorları hesaplar.

Bu modül, ajan çıktılarının güvenilirliğini veri tazeliği,
kaynak güvenilirliği ve ajanlar arası uyum gibi faktörlere
göre hesaplar.
"""

from typing import Dict, List, Any
from datetime import datetime

from fin_neurosim.schemas.agent_output import AgentOutput, SourceSummary
from fin_neurosim.utils.logging import setup_logger
from fin_neurosim.utils.time_utils import calculate_freshness_score

logger = setup_logger(__name__)


class ConfidenceEngine:
    """
    Dinamik güven skoru hesaplayıcı.

    Her ajan çıktısı için güven skoru hesaplar ve
    ajanlar arası uyumu değerlendirir.
    """

    # Kaynak güvenilirlik skorları (statik, genişletilebilir)
    SOURCE_RELIABILITY = {
        "Reuters": 0.95,
        "Bloomberg": 0.95,
        "Financial Times": 0.90,
        "Wall Street Journal": 0.90,
        "FRED": 0.98,
        "Alpha Vantage": 0.85,
        "Tavily": 0.80,
        "NewsAPI": 0.75,
        "default": 0.70
    }

    def compute_dynamic_confidence(
        self,
        agent_outputs: List[AgentOutput],
        freshness_factor: float = 0.95,
        context_freshness: Dict[str, float] = None
    ) -> Dict[str, float]:
        """
        Ajan çıktıları için dinamik güven skorları hesaplar.

        Args:
            agent_outputs: Ajan çıktıları listesi.
            freshness_factor: Tazelik faktörü (default: 0.95).
            context_freshness: Context verilerinin tazelik skorları.

        Returns:
            Her ajan için güven skorları dictionary'si.
        """
        confidence_map = {}

        for agent_output in agent_outputs:
            agent_id = agent_output.agent_id

            # Temel güven skoru (ajanın kendi confidence'ı)
            base_confidence = agent_output.confidence

            # Kaynak güvenilirliği
            source_reliability = self._calculate_source_reliability(agent_output.source_summary)

            # Veri tazeliği
            data_freshness = self._calculate_data_freshness(
                agent_output,
                context_freshness or {}
            )

            # Nihai güven skoru
            # Formül: base * source_reliability * data_freshness * freshness_factor
            final_confidence = (
                base_confidence *
                source_reliability *
                data_freshness *
                freshness_factor
            )

            confidence_map[agent_id] = max(0.0, min(1.0, final_confidence))

            logger.debug(
                f"{agent_id} güven skoru: {final_confidence:.3f} "
                f"(base={base_confidence:.3f}, source={source_reliability:.3f}, "
                f"freshness={data_freshness:.3f})"
            )

        return confidence_map

    def _calculate_source_reliability(self, source_summary: List[SourceSummary]) -> float:
        """
        Kaynak güvenilirliği skorunu hesaplar.

        Args:
            source_summary: Kaynak özeti listesi.

        Returns:
            Kaynak güvenilirliği skoru (0-1 arası).
        """
        if not source_summary:
            return self.SOURCE_RELIABILITY["default"]

        total_reliability = 0.0
        total_weight = 0.0

        for source in source_summary:
            source_name = source.source
            reliability = self.SOURCE_RELIABILITY.get(
                source_name,
                self.SOURCE_RELIABILITY["default"]
            )

            # Kaynağın kendi reliability skoru ile çarp
            weighted_reliability = reliability * source.reliability
            weight = source.freshness  # Tazelik ağırlık olarak kullanılır

            total_reliability += weighted_reliability * weight
            total_weight += weight

        if total_weight > 0:
            return total_reliability / total_weight
        else:
            return self.SOURCE_RELIABILITY["default"]

    def _calculate_data_freshness(
        self,
        agent_output: AgentOutput,
        context_freshness: Dict[str, float]
    ) -> float:
        """
        Veri tazeliği skorunu hesaplar.

        Args:
            agent_output: Ajan çıktısı.
            context_freshness: Context tazelik skorları.

        Returns:
            Veri tazeliği skoru (0-1 arası).
        """
        # Ajan çıktısındaki source summary'den tazelik skorları
        if agent_output.source_summary:
            freshness_scores = [s.freshness for s in agent_output.source_summary]
            agent_freshness = sum(freshness_scores) / len(freshness_scores)
        else:
            agent_freshness = 0.5

        # Context tazeliği
        context_fresh = context_freshness.get("overall", 0.5)

        # İkisinin ortalaması
        combined_freshness = (agent_freshness + context_fresh) / 2.0

        return max(0.0, min(1.0, combined_freshness))

    def calculate_inter_agent_agreement(
        self,
        agent_outputs: List[AgentOutput]
    ) -> float:
        """
        Ajanlar arası uyum skorunu hesaplar.

        Args:
            agent_outputs: Ajan çıktıları listesi.

        Returns:
            Uyum skoru (0-1 arası). Yüksek skor = yüksek uyum.
        """
        if len(agent_outputs) < 2:
            return 1.0  # Tek ajan varsa uyum %100

        # Risk seviyelerini sayısal değerlere çevir
        risk_levels = {
            "critical": 4,
            "high": 3,
            "medium": 2,
            "low": 1
        }

        risk_values = []
        for output in agent_outputs:
            risk_level = output.risk_level.lower()
            risk_value = risk_levels.get(risk_level, 2)  # Default: medium
            risk_values.append(risk_value)

        # Standart sapma hesapla (düşük sapma = yüksek uyum)
        if len(risk_values) > 1:
            mean = sum(risk_values) / len(risk_values)
            variance = sum((x - mean) ** 2 for x in risk_values) / len(risk_values)
            std_dev = variance ** 0.5

            # Standart sapmayı 0-1 arası skora çevir (max std_dev = 2)
            agreement = 1.0 - min(1.0, std_dev / 2.0)
        else:
            agreement = 1.0

        return max(0.0, min(1.0, agreement))

