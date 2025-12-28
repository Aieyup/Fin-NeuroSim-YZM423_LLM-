"""
Stage-2 Llama-3-8B: Bayesian synthesis ve karar üretimi.

Bu modül, Llama-3-8B modelini kullanarak
Stage-2 sentez ve nihai kararı üretir.
"""

import json
import torch
from typing import Dict, Any, List
from pathlib import Path

from fin_neurosim.llm.model_loader import ModelLoader
from fin_neurosim.schemas.agent_output import AgentOutput
from fin_neurosim.schemas.final_report import FinalReport, ActionPlan
from fin_neurosim.utils.logging import setup_logger

logger = setup_logger(__name__)


class Stage2Llama:
    """
    Stage-2 Llama-3-8B sentez modülü.

    Bayesian synthesis ve nihai karar üretimi yapar.
    """

    # Alternatif modeller (öncelik sırasına göre)
    # Not: Llama-3 modelleri gated olabilir, bu yüzden açık modeller öncelikli
    MODEL_NAMES = [
        "mistralai/Mistral-7B-Instruct-v0.2",   # İlk tercih (açık, güvenilir)
        "mistralai/Mistral-7B-v0.3",            # Fallback 1 (Stage-1 ile aynı)
        "microsoft/Phi-3-medium-4k-instruct",   # Fallback 2 (küçük, hızlı)
        "meta-llama/Meta-Llama-3-8B-Instruct",  # Fallback 3 (gated olabilir)
    ]
    MODEL_NAME = MODEL_NAMES[0]  # Varsayılan
    MAX_TOKENS = 500
    TEMPERATURE = 0.5  # Orta temperature = dengeli reasoning

    def __init__(self, model_loader: ModelLoader):
        """
        Stage2Llama'ı başlatır.

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
        stage1_report: AgentOutput,
        confidence_metrics: Dict[str, float],
        compact_context: str,
        user_query: str
    ) -> FinalReport:
        """
        Stage-1 raporunu sentezler ve nihai raporu üretir.

        Args:
            stage1_report: Stage-1 ajan çıktısı.
            confidence_metrics: Güven skorları.
            compact_context: Sıkıştırılmış context (kısa).
            user_query: Orijinal kullanıcı sorgusu.

        Returns:
            FinalReport nesnesi.
        """
        # Stage-1 risk seviyesini sakla (fallback için)
        self._last_stage1_risk = stage1_report.risk_level
        
        logger.info("Stage-2 Llama sentez başlatıldı")
        
        # Stage-1 risk seviyesini sakla (fallback için)
        self._last_stage1_risk = stage1_report.risk_level

        # Model yüklü değilse yükle (fallback ile)
        if not self.model_loader.is_model_loaded():
            model_loaded = False
            for model_name in self.MODEL_NAMES:
                try:
                    logger.info(f"Model yükleniyor: {model_name}")
                    self.model_loader.load_model(model_name)
                    self.MODEL_NAME = model_name  # Başarılı modeli kaydet
                    model_loaded = True
                    break
                except Exception as e:
                    logger.warning(f"Model yüklenemedi ({model_name}): {e}")
                    continue
            
            if not model_loaded:
                raise RuntimeError(
                    f"Hiçbir model yüklenemedi. Denenen modeller: {self.MODEL_NAMES}"
                )

        model_result = self.model_loader.get_current_model()
        if model_result is None:
            raise RuntimeError("Model yüklenemedi!")
        model, tokenizer = model_result

        # Prompt'u hazırla
        prompt = self._build_prompt(
            stage1_report,
            confidence_metrics,
            compact_context,
            user_query
        )

        # Token sayısını kontrol et
        tokens = tokenizer.encode(prompt, return_tensors="pt")
        token_count = tokens.shape[1]

        # Token limiti: 2000 (güvenli limit için 1900 kullan)
        max_tokens = 1900  # Buffer için 100 token bırak
        
        if token_count > max_tokens:
            logger.warning(
                f"Token sayısı limiti aşıyor: {token_count} > {max_tokens}. "
                "Prompt kısaltılıyor..."
            )
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
            synthesis_result = self._parse_llm_response(response)

            # FinalReport oluştur
            final_report = FinalReport(
                query=user_query,
                final_risk_level=synthesis_result.get("final_risk_level", "medium"),
                overall_confidence=synthesis_result.get("overall_confidence", 0.5),
                strategic_rationale=synthesis_result.get("strategic_rationale", response),
                action_plan=self._parse_action_plan(synthesis_result.get("action_plan", [])),
                agent_contributions=[stage1_report],
                data_freshness_score=confidence_metrics.get("data_freshness", 0.5),
                warnings=synthesis_result.get("warnings", [])
            )

            logger.info(f"Stage-2 sentez tamamlandı: Risk seviyesi = {final_report.final_risk_level}")
            return final_report

        except Exception as e:
            logger.error(f"Stage-2 sentez hatası: {e}")
            raise

    def _build_prompt(
        self,
        stage1_report: AgentOutput,
        confidence_metrics: Dict[str, float],
        compact_context: str,
        user_query: str
    ) -> str:
        """
        Sentez prompt'unu oluşturur.

        Args:
            stage1_report: Stage-1 raporu.
            confidence_metrics: Güven skorları.
            compact_context: Sıkıştırılmış context.
            user_query: Kullanıcı sorgusu.

        Returns:
            Hazırlanmış prompt.
        """
        prompt = f"""
{self.prompt_template}

[STAGE-1 RAPORU]:
- Risk Seviyesi: {stage1_report.risk_level}
- Güven Skoru: {stage1_report.confidence}
- Ana Faktörler: {', '.join(stage1_report.key_drivers[:5])}
- Gerekçe: {stage1_report.reasoning[:500]}

[GÜVEN METRİKLERİ]:
{json.dumps(confidence_metrics, indent=2)}

[CONTEXT (KISA)]:
{compact_context[:500]}

[KULLANICI SORGUSU]:
{user_query}

[BAYESYEN AĞIRLIKLANDIRMA KURALLARI]:
1. Stage-1 uyarıları yüksek ağırlık alır (0.3-0.7, güven skoruna göre)
2. Güven skorları ağırlıklandırmada kullanılır
3. Minority vs Majority dengesi kurulmalı

[OUTPUT FORMAT]:
JSON formatında şu alanları içer:
- "final_risk_level": "critical" | "high" | "medium" | "low"
- "overall_confidence": 0-1 arası skor
- "strategic_rationale": En az 300 kelime detaylı gerekçe
- "action_plan": Aksiyon planı listesi (her biri: priority, action, rationale)
- "warnings": Uyarılar listesi
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
        truncated_tokens = tokens[:max_tokens]
        return tokenizer.decode(truncated_tokens, skip_special_tokens=True)

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

        # Eğer boşsa veya None ise, varsayılan aksiyon planı oluştur
        if not action_plan_data:
            logger.warning("Aksiyon planı boş, varsayılan plan oluşturuluyor...")
            # Stage-1 raporuna göre varsayılan aksiyon planı
            risk_level = getattr(self, '_last_stage1_risk', 'medium')
            if risk_level in ['critical', 'high']:
                action_plans.append(
                    ActionPlan(
                        priority="high",
                        action="Piyasayı yakından izleyin ve risk yönetimi protokollerini aktifleştirin",
                        rationale="Yüksek risk seviyesi tespit edildi, dikkatli olunmalı"
                    )
                )
            return action_plans

        for item in action_plan_data:
            if isinstance(item, dict):
                # Boş değerleri kontrol et
                action = item.get("action", "").strip()
                rationale = item.get("rationale", "").strip()
                priority = item.get("priority", "medium").strip().lower()
                
                if action:  # Sadece action varsa ekle
                    action_plans.append(
                        ActionPlan(
                            priority=priority if priority in ['immediate', 'high', 'medium', 'low'] else 'medium',
                            action=action,
                            rationale=rationale if rationale else "LLM tarafından önerildi"
                        )
                    )
            elif isinstance(item, str) and item.strip():
                action_plans.append(
                    ActionPlan(
                        priority="medium",
                        action=item.strip(),
                        rationale="LLM tarafından önerildi"
                    )
                )

        # Eğer hala boşsa, en azından bir varsayılan ekle
        if not action_plans:
            logger.warning("Aksiyon planı parse edilemedi, varsayılan plan oluşturuluyor...")
            action_plans.append(
                ActionPlan(
                    priority="medium",
                    action="Risk analizi tamamlandı. Piyasa koşullarını düzenli olarak izleyin.",
                    rationale="Detaylı aksiyon planı üretilemedi, genel öneri"
                )
            )

        return action_plans

