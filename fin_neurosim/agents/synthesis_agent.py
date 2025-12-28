"""
Synthesis Agent: Stage-2 Decision Maker.

Bu ajan, tüm Stage-1 ajanlarının çıktılarını sentezler
ve nihai risk kararını verir.
"""

import json
from typing import Dict, Any, List
from pathlib import Path
from datetime import datetime

from fin_neurosim.agents.base_agent import BaseAgent
from fin_neurosim.schemas.agent_output import AgentOutput
from fin_neurosim.schemas.final_report import FinalReport, ActionPlan
from fin_neurosim.utils.logging import setup_logger

logger = setup_logger(__name__)


class SynthesisAgent(BaseAgent):
    """
    Synthesis Agent: Nihai karar verici ajan.

    Tüm ajan çıktılarını Bayesyen ağırlıklandırma ile sentezler.
    """

    def __init__(self, llm_provider: str = "openai"):
        """
        SynthesisAgent'ı başlatır.

        Args:
            llm_provider: LLM sağlayıcısı.
        """
        super().__init__("SynthesisAgent", llm_provider)
        self.prompt_template = self._load_prompt_template()

    def _load_prompt_template(self) -> str:
        """
        Prompt şablonunu yükler.

        Returns:
            Prompt şablonu string'i.
        """
        prompt_path = Path(__file__).parent.parent / "prompts" / "stage2_synthesis.md"
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
[PERSONA]: Çok disiplinli finansal stratejist ve baş ekonomist.
[TASK]: Farklı uzmanlardan gelen raporları sentezle ve nihai kararı ver.
[OUTPUT]: JSON formatında nihai risk seviyesi, stratejik gerekçe ve aksiyon planı.
"""

    async def synthesize(
        self,
        agent_outputs: List[AgentOutput],
        confidence_map: Dict[str, float],
        context: Dict[str, Any],
        user_query: str
    ) -> FinalReport:
        """
        Ajan çıktılarını sentezler ve nihai raporu üretir.

        Args:
            agent_outputs: Stage-1 ajan çıktıları.
            confidence_map: Güven skorları.
            context: Context Cloud.
            user_query: Orijinal kullanıcı sorgusu.

        Returns:
            FinalReport nesnesi.
        """
        logger.info("SynthesisAgent sentez başlatıldı")

        # Prompt'u hazırla
        prompt = self._build_prompt(agent_outputs, confidence_map, context, user_query)

        # LLM'i çağır
        response = await self._call_llm(
            prompt=prompt,
            system_prompt="Sen bir finansal stratejist ve baş ekonomistsin. Tüm perspektifleri dikkate al ve net bir karar ver.",
            temperature=0.5
        )

        # LLM yanıtını parse et
        synthesis_result = self._parse_llm_response(response)

        # FinalReport oluştur
        final_report = FinalReport(
            query=user_query,
            final_risk_level=synthesis_result.get("final_risk_level", "medium"),
            overall_confidence=synthesis_result.get("overall_confidence", 0.5),
            strategic_rationale=synthesis_result.get("strategic_rationale", response),
            action_plan=self._parse_action_plan(synthesis_result.get("action_plan", [])),
            agent_contributions=agent_outputs,
            data_freshness_score=context.get("data_freshness", {}).get("overall", 0.5),
            warnings=synthesis_result.get("warnings", [])
        )

        logger.info(f"SynthesisAgent sentez tamamlandı: Risk seviyesi = {final_report.final_risk_level}")
        return final_report

    def _build_prompt(
        self,
        agent_outputs: List[AgentOutput],
        confidence_map: Dict[str, float],
        context: Dict[str, Any],
        user_query: str
    ) -> str:
        """
        Sentez prompt'unu oluşturur.

        Args:
            agent_outputs: Ajan çıktıları.
            confidence_map: Güven skorları.
            context: Context Cloud.
            user_query: Kullanıcı sorgusu.

        Returns:
            Hazırlanmış prompt.
        """
        # Ajan çıktılarını formatla
        agent_outputs_str = "\n".join([
            f"- {output.agent_id}: Risk={output.risk_level}, "
            f"Confidence={output.confidence}, "
            f"Drivers={', '.join(output.key_drivers[:3])}"
            for output in agent_outputs
        ])

        # Güven skorlarını formatla
        confidence_str = "\n".join([
            f"- {agent_id}: {score:.3f}"
            for agent_id, score in confidence_map.items()
        ])

        prompt = self.prompt_template.replace("{agent_outputs}", agent_outputs_str)
        prompt = prompt.replace("{confidence_metrics}", confidence_str)
        prompt = prompt.replace("{context}", str(context)[:1000])  # İlk 1000 karakter
        prompt = prompt.replace("{user_query}", user_query)

        return prompt

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """
        LLM yanıtını parse eder.

        Args:
            response: LLM yanıtı.

        Returns:
            Parse edilmiş sentez sonucu.
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
        if any(word in response.lower() for word in ["critical", "crisis", "urgent"]):
            risk_level = "critical"
        elif any(word in response.lower() for word in ["high", "warning", "concern"]):
            risk_level = "high"
        elif any(word in response.lower() for word in ["low", "stable", "safe"]):
            risk_level = "low"

        return {
            "final_risk_level": risk_level,
            "overall_confidence": 0.6,
            "strategic_rationale": response,
            "action_plan": [],
            "warnings": []
        }

    def _parse_action_plan(self, action_plan_data: List[Dict[str, Any]]) -> List[ActionPlan]:
        """
        Aksiyon planını parse eder.

        Args:
            action_plan_data: Aksiyon planı verisi.

        Returns:
            ActionPlan listesi.
        """
        action_plans = []

        for item in action_plan_data:
            if isinstance(item, dict):
                action_plans.append(
                    ActionPlan(
                        priority=item.get("priority", "medium"),
                        action=item.get("action", ""),
                        rationale=item.get("rationale", "")
                    )
                )
            elif isinstance(item, str):
                # Basit string ise, varsayılan priority ile ekle
                action_plans.append(
                    ActionPlan(
                        priority="medium",
                        action=item,
                        rationale="LLM tarafından önerildi"
                    )
                )

        return action_plans

