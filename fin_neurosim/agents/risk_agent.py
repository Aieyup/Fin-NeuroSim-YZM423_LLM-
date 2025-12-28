"""
Risk Agent: Stage-1 Crisis Detector.

Bu ajan, nadir görülen ancak yüksek etkili kriz sinyallerini
tespit etmek için özel olarak tasarlanmıştır.
"""

import json
from typing import Dict, Any
from pathlib import Path

from fin_neurosim.agents.base_agent import BaseAgent
from fin_neurosim.schemas.agent_output import AgentOutput, SourceSummary
from fin_neurosim.utils.logging import setup_logger
from fin_neurosim.utils.time_utils import calculate_freshness_score

logger = setup_logger(__name__)


class RiskAgent(BaseAgent):
    """
    Risk Agent: Kriz ve anomali tespit ajanı.

    Minority class sinyallerine odaklanır ve
    Black Swan olaylarını erken tespit etmeye çalışır.
    """

    def __init__(self, llm_provider: str = "openai"):
        """
        RiskAgent'ı başlatır.

        Args:
            llm_provider: LLM sağlayıcısı.
        """
        super().__init__("RiskAgent", llm_provider)
        self.prompt_template = self._load_prompt_template()

    def _load_prompt_template(self) -> str:
        """
        Prompt şablonunu yükler.

        Returns:
            Prompt şablonu string'i.
        """
        prompt_path = Path(__file__).parent.parent / "prompts" / "stage1_risk.md"
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
[PERSONA]: 20+ yıl deneyimli, "Black Swan" olayları konusunda uzmanlaşmış kıdemli risk analisti.
[TONE]: Şüpheci, detaycı, uyarıcı ve aşırı rasyonel.
[TASK]: Aşağıdaki veriler içinde piyasanın "gürültü" olarak gördüğü ancak %1 olasılıkla gerçekleşebilecek sistemik kriz sinyallerini tespit et.
[OUTPUT]: JSON formatında; tespit edilen anomaliler, benzer tarihsel krizler ve "Anomali Güven Skoru" (0-1).
"""

    async def analyze(
        self,
        context: Dict[str, Any],
        intent: Any
    ) -> AgentOutput:
        """
        Context ve intent'e göre risk analizi yapar.

        Args:
            context: Semantic Context Cloud.
            intent: Kullanıcı intent'i.

        Returns:
            AgentOutput nesnesi.
        """
        logger.info("RiskAgent analiz başlatıldı")

        # Prompt'u hazırla
        prompt = self._build_prompt(context, intent)

        # LLM'i çağır
        response = await self._call_llm(
            prompt=prompt,
            system_prompt="Sen bir kriz tespit uzmanısın. Minority class sinyallerine özel dikkat göster.",
            temperature=0.3  # Düşük temperature = daha deterministik
        )

        # LLM yanıtını parse et
        analysis_result = self._parse_llm_response(response)

        # Source summary oluştur
        source_summary = self._create_source_summary_from_context(context)

        # AgentOutput oluştur
        agent_output = AgentOutput(
            agent_id=self.agent_id,
            signal_type="anomaly",
            risk_level=analysis_result.get("risk_level", "medium"),
            confidence=analysis_result.get("anomaly_confidence", 0.5),
            key_drivers=analysis_result.get("key_drivers", []),
            source_summary=source_summary,
            reasoning=analysis_result.get("reasoning", response),
            raw_data=analysis_result
        )

        logger.info(f"RiskAgent analiz tamamlandı: Risk seviyesi = {agent_output.risk_level}")
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
        # Haber verisini 3 kez tekrarla (positional bias)
        news_data = context.get("news_data", {})
        news_repeated = "\n".join([str(news_data)] * 3)

        # Prompt'u doldur
        prompt = self.prompt_template.replace("{live_context}", news_repeated)
        prompt = prompt.replace("{historical_patterns}", "2008 Financial Crisis, 2020 COVID-19 Crash")
        prompt = prompt.replace("{market_data}", str(context.get("market_data", {})))
        prompt = prompt.replace("{macro_data}", str(context.get("macro_data", {})))

        # Ek context bilgisi ekle
        prompt += f"\n\n[EK CONTEXT]:\n"
        prompt += f"- Analiz Tipi: {intent.analysis_type}\n"
        prompt += f"- Varlıklar: {', '.join(intent.assets)}\n"
        prompt += f"- Bölge: {intent.region}\n"
        prompt += f"- Zaman Ufku: {intent.time_horizon}\n"

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
            # JSON bloğunu bul
            json_start = response.find("{")
            json_end = response.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                result = json.loads(json_str)
            else:
                # JSON bulunamadıysa, basit parsing
                result = self._parse_text_response(response)
        except Exception as e:
            logger.warning(f"LLM yanıtı parse edilemedi: {e}")
            result = self._parse_text_response(response)

        return result

    def _parse_text_response(self, response: str) -> Dict[str, Any]:
        """
        Metin yanıtını parse eder (JSON bulunamazsa).

        Args:
            response: LLM yanıtı.

        Returns:
            Parse edilmiş sonuç.
        """
        # Basit keyword-based parsing
        risk_level = "medium"
        if any(word in response.lower() for word in ["critical", "crisis", "crash", "collapse"]):
            risk_level = "critical"
        elif any(word in response.lower() for word in ["high", "danger", "warning"]):
            risk_level = "high"
        elif any(word in response.lower() for word in ["low", "stable", "safe"]):
            risk_level = "low"

        # Key drivers'ı çıkar
        key_drivers = []
        if "anomaly" in response.lower():
            key_drivers.append("Anomali tespit edildi")
        if "crisis" in response.lower():
            key_drivers.append("Kriz sinyali")

        return {
            "risk_level": risk_level,
            "anomaly_confidence": 0.6,
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
        sources = []
        freshness_scores = {}

        # Haber kaynakları
        news_data = context.get("news_data", {})
        if news_data.get("results"):
            source_name = news_data.get("source", "NewsAPI")
            sources.append(source_name)
            # Haber tazeliği
            if news_data.get("results"):
                freshness_scores[source_name] = context.get("data_freshness", {}).get("news", 0.5)

        # Piyasa verisi
        if context.get("market_data"):
            sources.append("Alpha Vantage")
            freshness_scores["Alpha Vantage"] = context.get("data_freshness", {}).get("market", 0.5)

        # Makro verisi
        if context.get("macro_data"):
            sources.append("FRED")
            freshness_scores["FRED"] = context.get("data_freshness", {}).get("macro", 0.5)

        return self._create_source_summary(sources, freshness_scores)

