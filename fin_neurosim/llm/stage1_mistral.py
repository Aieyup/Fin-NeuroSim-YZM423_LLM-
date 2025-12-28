"""
Stage-1 Mistral-7B: Minority-focused risk analizi.

Bu modül, Mistral-7B modelini kullanarak
Stage-1 risk analizini gerçekleştirir.
"""

import json
import torch
from typing import Dict, Any, Optional
from pathlib import Path

from fin_neurosim.llm.model_loader import ModelLoader
from fin_neurosim.schemas.agent_output import AgentOutput, SourceSummary
from fin_neurosim.utils.logging import setup_logger

logger = setup_logger(__name__)


class Stage1Mistral:
    """
    Stage-1 Mistral-7B analiz modülü.

    Minority-focused risk analizi yapar.
    """

    MODEL_NAME = "mistralai/Mistral-7B-v0.3"
    MAX_TOKENS = 300
    TEMPERATURE = 0.3  # Düşük temperature = daha deterministik

    def __init__(self, model_loader: ModelLoader):
        """
        Stage1Mistral'ı başlatır.

        Args:
            model_loader: ModelLoader instance'ı.
        """
        self.model_loader = model_loader
        self.prompt_template = self._load_prompt_template()

    def _load_prompt_template(self) -> str:
        """
        Prompt şablonunu yükler.

        Returns:
            Prompt şablonu string'i.
        """
        prompt_path = Path(__file__).parent.parent / "prompts" / "stage1_minority.md"
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
        compact_context: str,
        intent: Any
    ) -> AgentOutput:
        """
        Compact context ile Stage-1 analizi yapar.

        Args:
            compact_context: Sıkıştırılmış context (≤1200 token).
            intent: Kullanıcı intent'i.

        Returns:
            AgentOutput nesnesi.
        """
        logger.info("Stage-1 Mistral analizi başlatıldı")

        # Model yüklü değilse yükle
        if not self.model_loader.is_model_loaded(self.MODEL_NAME):
            logger.info(f"Model yükleniyor: {self.MODEL_NAME}")
            self.model_loader.load_model(self.MODEL_NAME)

        model, tokenizer = self.model_loader.get_current_model()

        # Prompt'u hazırla
        prompt = self._build_prompt(compact_context, intent)

        # Token sayısını kontrol et
        tokens = tokenizer.encode(prompt, return_tensors="pt")
        token_count = tokens.shape[1]

        # Token limiti: 2000 (güvenli limit için 1900 kullan)
        max_tokens = 1900  # Buffer için 100 token bırak
        
        if token_count > max_tokens:
            logger.warning(
                f"Token sayısı limiti aşıyor: {token_count} > {max_tokens}. "
                "Context kısaltılıyor..."
            )
            # Context'i kısalt
            prompt = self._truncate_prompt(prompt, tokenizer, max_tokens=max_tokens)
            # Yeniden say
            tokens = tokenizer.encode(prompt, return_tensors="pt")
            token_count = tokens.shape[1]

        logger.info(f"Prompt token sayısı: {token_count} (limit: {max_tokens})")

        try:
            # Model inference
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.MAX_TOKENS,
                    temperature=self.TEMPERATURE,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )

            # Decode
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Prompt'u çıkar, sadece yanıtı al
            response = response[len(prompt):].strip()

            # LLM yanıtını parse et
            analysis_result = self._parse_llm_response(response)

            # AgentOutput oluştur
            agent_output = AgentOutput(
                agent_id="Stage1Mistral",
                signal_type="anomaly",
                risk_level=analysis_result.get("risk_level", "medium"),
                confidence=analysis_result.get("anomaly_confidence", 0.5),
                key_drivers=analysis_result.get("key_drivers", []),
                source_summary=[],  # Context compressor'dan gelecek
                reasoning=analysis_result.get("reasoning", response),
                raw_data=analysis_result
            )

            logger.info(f"Stage-1 analiz tamamlandı: Risk seviyesi = {agent_output.risk_level}")
            return agent_output

        except Exception as e:
            logger.error(f"Stage-1 analiz hatası: {e}")
            raise

    def _build_prompt(self, compact_context: str, intent: Any) -> str:
        """
        Prompt'u oluşturur.

        Args:
            compact_context: Sıkıştırılmış context.
            intent: Kullanıcı intent'i.

        Returns:
            Hazırlanmış prompt.
        """
        # Virtual agents (lens'ler) prompt içinde
        prompt = f"""
{self.prompt_template}

[CONTEXT]:
{compact_context}

[VIRTUAL AGENTS - ÇOKLU LENS ANALİZİ]:

1. [Risk Lens]: Anomali ve kriz sinyallerini tespit et.
2. [Macro Lens]: Makroekonomik sistemik riskleri değerlendir.
3. [Sentiment Lens]: Piyasa duygusundaki aşırılıkları yakala.
4. [Technical Lens]: Volatilite ve yapısal göstergeleri analiz et.

[INTENT]:
- Analiz Tipi: {intent.analysis_type}
- Varlıklar: {', '.join(intent.assets)}
- Bölge: {intent.region}
- Zaman Ufku: {intent.time_horizon}

[OUTPUT FORMAT]:
JSON formatında şu alanları içer:
- "risk_level": "critical" | "high" | "medium" | "low"
- "anomaly_confidence": 0-1 arası skor
- "key_drivers": Ana risk faktörleri listesi
- "reasoning": Detaylı mantıksal gerekçe
- "virtual_agents_output": Her lens'in çıktısı
"""

        return prompt

    def _truncate_prompt(
        self,
        prompt: str,
        tokenizer: Any,
        max_tokens: int = 2000
    ) -> str:
        """
        Prompt'u token limitine göre kısaltır.

        Args:
            prompt: Orijinal prompt.
            tokenizer: Tokenizer.
            max_tokens: Maksimum token sayısı.

        Returns:
            Kısaltılmış prompt.
        """
        tokens = tokenizer.encode(prompt)
        if len(tokens) <= max_tokens:
            return prompt

        # Context kısmını kısalt
        # Prompt başlangıcını ve bitişini koru
        context_start = prompt.find("[CONTEXT]:")
        if context_start == -1:
            # Context bulunamadı, genel kısaltma
            truncated_tokens = tokens[:max_tokens]
            return tokenizer.decode(truncated_tokens, skip_special_tokens=True)

        # Context kısmını bul ve kısalt
        before_context = prompt[:context_start]
        context_part = prompt[context_start:]

        before_tokens = tokenizer.encode(before_context)
        context_tokens = tokenizer.encode(context_part)

        # Context'i kısalt
        available_tokens = max_tokens - len(before_tokens) - 100  # Buffer
        if available_tokens > 0:
            truncated_context_tokens = context_tokens[:available_tokens]
            truncated_context = tokenizer.decode(
                truncated_context_tokens,
                skip_special_tokens=True
            )
            return before_context + truncated_context

        # Çok kısaysa, sadece başlangıç kısmını al
        truncated_tokens = tokens[:max_tokens]
        return tokenizer.decode(truncated_tokens, skip_special_tokens=True)

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
        risk_level = "medium"
        if any(word in response.lower() for word in ["critical", "crisis", "crash", "collapse"]):
            risk_level = "critical"
        elif any(word in response.lower() for word in ["high", "danger", "warning"]):
            risk_level = "high"
        elif any(word in response.lower() for word in ["low", "stable", "safe"]):
            risk_level = "low"

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

