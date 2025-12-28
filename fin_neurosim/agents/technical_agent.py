"""
Technical Agent: Stage-1 Teknik Analiz Ajanı.

Bu ajan, piyasa verilerini teknik analiz açısından
değerlendirir ve volatilite, trend ve momentum analizi yapar.
"""

import json
from typing import Dict, Any
from pathlib import Path

from fin_neurosim.agents.base_agent import BaseAgent
from fin_neurosim.schemas.agent_output import AgentOutput, SourceSummary
from fin_neurosim.utils.logging import setup_logger

logger = setup_logger(__name__)


class TechnicalAgent(BaseAgent):
    """
    Technical Agent: Teknik analiz ajanı.

    Volatilite, trend ve momentum göstergelerini analiz eder.
    """

    def __init__(self, llm_provider: str = "openai"):
        """
        TechnicalAgent'ı başlatır.

        Args:
            llm_provider: LLM sağlayıcısı.
        """
        super().__init__("TechnicalAgent", llm_provider)
        self.prompt_template = self._load_prompt_template()

    def _load_prompt_template(self) -> str:
        """
        Prompt şablonunu yükler.

        Returns:
            Prompt şablonu string'i.
        """
        prompt_path = Path(__file__).parent.parent / "prompts" / "stage1_technical.md"
        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logger.warning(f"Prompt dosyası yüklenemedi: {e}")
            return self._get_default_prompt()

    def _get_default_prompt(self) -> str:
        """
        Varsayılan prompt döndürür.

        Returns:
            Varsayılan prompt.
        """
        return """
[PERSONA]: 20+ yıl deneyimli teknik analiz uzmanı.
[TASK]: Piyasa verilerini teknik analiz açısından değerlendir.
[OUTPUT]: JSON formatında teknik sinyal ve risk seviyesi.
"""

    async def analyze(
        self,
        context: Dict[str, Any],
        intent: Any
    ) -> AgentOutput:
        """
        Context ve intent'e göre teknik analiz yapar.

        Args:
            context: Semantic Context Cloud.
            intent: Kullanıcı intent'i.

        Returns:
            AgentOutput nesnesi.
        """
        logger.info("TechnicalAgent analiz başlatıldı")

        # Prompt'u hazırla
        prompt = self._build_prompt(context, intent)

        # LLM'i çağır
        response = await self._call_llm(
            prompt=prompt,
            system_prompt="Sen bir teknik analiz uzmanısın.",
            temperature=0.4
        )

        # LLM yanıtını parse et
        analysis_result = self._parse_llm_response(response)

        # Source summary oluştur
        source_summary = self._create_source_summary_from_context(context)

        # AgentOutput oluştur
        agent_output = AgentOutput(
            agent_id=self.agent_id,
            signal_type="technical",
            risk_level=analysis_result.get("risk_level", "medium"),
            confidence=analysis_result.get("confidence", 0.5),
            key_drivers=analysis_result.get("key_drivers", []),
            source_summary=source_summary,
            reasoning=analysis_result.get("reasoning", response),
            raw_data=analysis_result
        )

        logger.info(f"TechnicalAgent analiz tamamlandı: Risk seviyesi = {agent_output.risk_level}")
        return agent_output

    def _build_prompt(self, context: Dict[str, Any], intent: Any) -> str:
        """
        Context ve intent'ten prompt oluşturur.

        Args:
            context: Context Cloud.
            intent: Kullanıcı intent'i.

        Returns:
            Hazırlanmış prompt.
        """
        market_data = context.get("market_data", {})

        prompt = self.prompt_template.replace("{market_data}", str(market_data))
        prompt = prompt.replace("{vix_data}", str(context.get("vix_data", {})))
        prompt = prompt.replace("{assets}", ", ".join(intent.assets))
        prompt = prompt.replace("{time_horizon}", intent.time_horizon)

        return prompt

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """
        LLM yanıtını parse eder.

        Args:
            response: LLM yanıtı.

        Returns:
            Parse edilmiş analiz sonucu.
        """
        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                result = json.loads(json_str)
            else:
                result = self._parse_text_response(response)
        except Exception as e:
            logger.warning(f"LLM yanıtı parse edilemedi: {e}")
            result = self._parse_text_response(response)

        return result

    def _parse_text_response(self, response: str) -> Dict[str, Any]:
        """
        Metin yanıtını parse eder.

        Args:
            response: LLM yanıtı.

        Returns:
            Parse edilmiş sonuç.
        """
        risk_level = "medium"
        if any(word in response.lower() for word in ["crash", "collapse", "critical"]):
            risk_level = "critical"
        elif any(word in response.lower() for word in ["high", "volatile", "uncertain"]):
            risk_level = "high"
        elif any(word in response.lower() for word in ["low", "stable", "bullish"]):
            risk_level = "low"

        key_drivers = []
        if "volatility" in response.lower():
            key_drivers.append("Yüksek volatilite")
        if "trend" in response.lower():
            key_drivers.append("Trend değişimi")
        if "momentum" in response.lower():
            key_drivers.append("Momentum kaybı")

        return {
            "risk_level": risk_level,
            "confidence": 0.6,
            "key_drivers": key_drivers,
            "reasoning": response
        }

    def _create_source_summary_from_context(self, context: Dict[str, Any]) -> list[SourceSummary]:
        """
        Context'ten source summary oluşturur.

        Args:
            context: Context Cloud.

        Returns:
            SourceSummary listesi.
        """
        sources = ["Alpha Vantage"]
        freshness_scores = {
            "Alpha Vantage": context.get("data_freshness", {}).get("market", 0.5)
        }

        return self._create_source_summary(sources, freshness_scores)

