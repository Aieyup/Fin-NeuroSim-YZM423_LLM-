"""
Base Agent: Tüm ajanlar için temel sınıf.

Bu modül, tüm ajanların ortak özelliklerini ve
standart interface'ini tanımlar.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import os

from fin_neurosim.schemas.agent_output import AgentOutput, SourceSummary
from fin_neurosim.utils.logging import setup_logger

logger = setup_logger(__name__)


class BaseAgent(ABC):
    """
    Tüm ajanlar için temel sınıf.

    Her ajan bu sınıftan türetilmeli ve
    analyze() metodunu implement etmelidir.
    """

    def __init__(self, agent_id: str, llm_provider: str = "openai"):
        """
        BaseAgent'ı başlatır.

        Args:
            agent_id: Ajan tanımlayıcısı (örn: "RiskAgent").
            llm_provider: LLM sağlayıcısı ("openai" veya "anthropic").
        """
        self.agent_id = agent_id
        self.llm_provider = llm_provider
        self.api_key = self._get_api_key()

    def _get_api_key(self) -> Optional[str]:
        """
        LLM API anahtarını alır.

        Returns:
            API anahtarı veya None.
        """
        if self.llm_provider == "openai":
            return os.getenv("OPENAI_API_KEY")
        elif self.llm_provider == "anthropic":
            return os.getenv("ANTHROPIC_API_KEY")
        else:
            return None

    @abstractmethod
    async def analyze(
        self,
        context: Dict[str, Any],
        intent: Any
    ) -> AgentOutput:
        """
        Context ve intent'e göre analiz yapar.

        Args:
            context: Semantic Context Cloud.
            intent: Kullanıcı intent'i.

        Returns:
            AgentOutput nesnesi.
        """
        pass

    async def _call_llm(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7
    ) -> str:
        """
        LLM'i çağırır (OpenAI veya Anthropic).

        Args:
            prompt: Kullanıcı prompt'u.
            system_prompt: Sistem prompt'u (opsiyonel).
            temperature: Temperature parametresi.

        Returns:
            LLM yanıtı (string).
        """
        if not self.api_key:
            logger.error(f"{self.agent_id}: API anahtarı bulunamadı!")
            return "API anahtarı bulunamadı."

        if self.llm_provider == "openai":
            return await self._call_openai(prompt, system_prompt, temperature)
        elif self.llm_provider == "anthropic":
            return await self._call_anthropic(prompt, system_prompt, temperature)
        else:
            logger.error(f"Bilinmeyen LLM sağlayıcısı: {self.llm_provider}")
            return "LLM sağlayıcısı bulunamadı."

    async def _call_openai(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7
    ) -> str:
        """
        OpenAI API'yi çağırır.

        Args:
            prompt: Kullanıcı prompt'u.
            system_prompt: Sistem prompt'u.
            temperature: Temperature.

        Returns:
            LLM yanıtı.
        """
        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=self.api_key)

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = await client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                temperature=temperature
            )

            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API hatası: {e}")
            return f"LLM hatası: {str(e)}"

    async def _call_anthropic(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7
    ) -> str:
        """
        Anthropic API'yi çağırır.

        Args:
            prompt: Kullanıcı prompt'u.
            system_prompt: Sistem prompt'u.
            temperature: Temperature.

        Returns:
            LLM yanıtı.
        """
        try:
            from anthropic import AsyncAnthropic

            client = AsyncAnthropic(api_key=self.api_key)

            system_message = system_prompt or "Sen bir finansal analiz uzmanısın."

            message = await client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=4096,
                system=system_message,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature
            )

            return message.content[0].text
        except Exception as e:
            logger.error(f"Anthropic API hatası: {e}")
            return f"LLM hatası: {str(e)}"

    def _create_source_summary(
        self,
        sources: list,
        freshness_scores: Dict[str, float]
    ) -> list[SourceSummary]:
        """
        Source summary listesi oluşturur.

        Args:
            sources: Kaynak adları listesi.
            freshness_scores: Kaynak tazelik skorları.

        Returns:
            SourceSummary listesi.
        """
        source_summaries = []
        for source in sources:
            freshness = freshness_scores.get(source, 0.5)
            reliability = self._get_source_reliability(source)

            source_summaries.append(
                SourceSummary(
                    source=source,
                    freshness=freshness,
                    reliability=reliability
                )
            )

        return source_summaries

    def _get_source_reliability(self, source: str) -> float:
        """
        Kaynak güvenilirlik skorunu döndürür.

        Args:
            source: Kaynak adı.

        Returns:
            Güvenilirlik skoru (0-1 arası).
        """
        reliability_map = {
            "Reuters": 0.95,
            "Bloomberg": 0.95,
            "Financial Times": 0.90,
            "Wall Street Journal": 0.90,
            "FRED": 0.98,
            "Alpha Vantage": 0.85,
            "Tavily": 0.80,
            "NewsAPI": 0.75,
        }

        return reliability_map.get(source, 0.70)

