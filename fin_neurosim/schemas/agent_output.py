"""
Ajan çıktı şemaları: Standart mesaj formatı ve validasyon.

Bu modül, tüm ajanların çıktılarını standartlaştırmak için Pydantic şemalarını içerir.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class SourceSummary(BaseModel):
    """
    Kaynak özeti: Veri kaynağının güvenilirlik ve tazelik bilgileri.

    Attributes:
        source: Kaynak adı (örn: "Reuters", "FRED", "Alpha Vantage").
        freshness: Veri tazeliği skoru (0-1 arası).
        reliability: Kaynak güvenilirlik skoru (0-1 arası).
    """

    source: str = Field(..., description="Kaynak adı")
    freshness: float = Field(..., ge=0.0, le=1.0, description="Veri tazeliği skoru")
    reliability: float = Field(..., ge=0.0, le=1.0, description="Kaynak güvenilirlik skoru")


class AgentOutput(BaseModel):
    """
    Standart ajan çıktı formatı.

    Tüm ajanlar bu formatta çıktı üretmelidir. Bu, ajanlar arası
    iletişimi standartlaştırır ve orchestrator'ın çıktıları işlemesini kolaylaştırır.

    Attributes:
        agent_id: Ajan tanımlayıcısı (örn: "RiskAgent", "MacroAgent").
        timestamp: Çıktının üretildiği zaman damgası (ISO 8601 formatında).
        signal_type: Sinyal tipi (örn: "anomaly", "trend", "sentiment").
        risk_level: Risk seviyesi ("critical", "high", "medium", "low").
        confidence: Ajanın çıktısına olan güven skoru (0-1 arası).
        key_drivers: Risk veya sinyalin ana nedenleri (liste).
        source_summary: Kullanılan veri kaynaklarının özeti.
        reasoning: Ajanın mantıksal gerekçesi (metin).
        raw_data: Ham veri (opsiyonel, debug için).
    """

    agent_id: str = Field(..., description="Ajan tanımlayıcısı")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Zaman damgası")
    signal_type: str = Field(..., description="Sinyal tipi")
    risk_level: str = Field(..., description="Risk seviyesi")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Güven skoru")
    key_drivers: List[str] = Field(default_factory=list, description="Ana nedenler")
    source_summary: List[SourceSummary] = Field(default_factory=list, description="Kaynak özeti")
    reasoning: str = Field(..., description="Mantıksal gerekçe")
    raw_data: Optional[Dict[str, Any]] = Field(default=None, description="Ham veri (opsiyonel)")

    class Config:
        """Pydantic yapılandırması."""

        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class CircuitBreakerEvent(BaseModel):
    """
    Circuit Breaker olay şeması.

    Ajanlar arasında çelişki tespit edildiğinde bu format kullanılır.

    Attributes:
        event: Olay tipi (örn: "CONTRADICTION_DETECTED").
        conflicting_agents: Çelişen ajanların ID'leri.
        arbiter: Hakem ajan ID'si.
        action: Alınacak aksiyon (örn: "REWEIGHT_AND_REEVALUATE").
        details: Ek detaylar (opsiyonel).
    """

    event: str = Field(..., description="Olay tipi")
    conflicting_agents: List[str] = Field(..., description="Çelişen ajanlar")
    arbiter: str = Field(..., description="Hakem ajan")
    action: str = Field(..., description="Alınacak aksiyon")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Ek detaylar")

