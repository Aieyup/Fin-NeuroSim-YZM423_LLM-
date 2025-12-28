"""
Macro Agent: Stage-1 Makroekonomi Analiz Ajanı.

Bu ajan, makroekonomik göstergeleri analiz eder ve
sistemik riskleri tespit eder.
"""

import json
from typing import Dict, Any
from pathlib import Path

from fin_neurosim.agents.base_agent import BaseAgent
from fin_neurosim.schemas.agent_output import AgentOutput, SourceSummary
from fin_neurosim.utils.logging import setup_logger

logger = setup_logger(__name__)


class MacroAgent(BaseAgent):
    """
    Macro Agent: Makroekonomi analiz ajanı.

    CPI, GDP, işsizlik ve faiz oranlarını analiz eder.
    """

    def __init__(self, llm_provider: str = "openai"):
        """
        MacroAgent'ı başlatır.

        Args:
            llm_provider: LLM sağlayıcısı.
        """
        super().__init__("MacroAgent", llm_provider)
        self.prompt_template = self._load_prompt_template()

    def _load_prompt_template(self) -> str:
        """
        Prompt şablonunu yükler.

        Returns:
            Prompt şablonu string'i.
        """
        prompt_path = Path(__file__).parent.parent / "prompts" / "stage1_macro.md"
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
[PERSONA]: 15+ yıl deneyimli makroekonomist ve sistemik risk uzmanı.
[TASK]: Makroekonomik verileri analiz et ve sistemik risk göstergelerini tespit et.
[OUTPUT]: JSON formatında sistemik risk seviyesi ve analiz.
"""

    async def analyze(
        self,
        context: Dict[str, Any],
        intent: Any
    ) -> AgentOutput:
        """
        Context ve intent'e göre makroekonomi analizi yapar.

        Args:
            context: Semantic Context Cloud.
            intent: Kullanıcı intent'i.

        Returns:
            AgentOutput nesnesi.
        """
        logger.info("MacroAgent analiz başlatıldı")

        # Prompt'u hazırla
        prompt = self._build_prompt(context, intent)

        # LLM'i çağır
        response = await self._call_llm(
            prompt=prompt,
            system_prompt="Sen bir makroekonomist ve sistemik risk uzmanısın.",
            temperature=0.4
        )

        # LLM yanıtını parse et
        analysis_result = self._parse_llm_response(response)

        # Source summary oluştur
        source_summary = self._create_source_summary_from_context(context)

        # AgentOutput oluştur
        agent_output = AgentOutput(
            agent_id=self.agent_id,
            signal_type="macro",
            risk_level=analysis_result.get("systemic_risk_level", "medium"),
            confidence=analysis_result.get("confidence", 0.5),
            key_drivers=analysis_result.get("key_drivers", []),
            source_summary=source_summary,
            reasoning=analysis_result.get("reasoning", response),
            raw_data=analysis_result
        )

        logger.info(f"MacroAgent analiz tamamlandı: Risk seviyesi = {agent_output.risk_level}")
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
        macro_data = context.get("macro_data", {})

        prompt = self.prompt_template.replace("{macro_data}", str(macro_data))
        prompt = prompt.replace("{region}", intent.region)
        prompt = prompt.replace("{time_horizon}", intent.time_horizon)
        prompt = prompt.replace("{market_data}", str(context.get("market_data", {})))

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
        if any(word in response.lower() for word in ["critical", "crisis", "recession"]):
            risk_level = "critical"
        elif any(word in response.lower() for word in ["high", "concern", "warning"]):
            risk_level = "high"
        elif any(word in response.lower() for word in ["low", "stable", "healthy"]):
            risk_level = "low"

        key_drivers = []
        if "inflation" in response.lower():
            key_drivers.append("Enflasyon endişesi")
        if "unemployment" in response.lower():
            key_drivers.append("İşsizlik trendi")
        if "interest" in response.lower():
            key_drivers.append("Faiz oranı değişimi")

        return {
            "systemic_risk_level": risk_level,
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
        sources = ["FRED"]
        freshness_scores = {
            "FRED": context.get("data_freshness", {}).get("macro", 0.5)
        }

        return self._create_source_summary(sources, freshness_scores)

