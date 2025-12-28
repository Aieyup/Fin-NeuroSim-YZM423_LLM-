"""
Nihai rapor şemaları: Sistem çıktısı formatı.

Bu modül, orchestrator'ın ürettiği nihai raporun yapısını tanımlar.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

from fin_neurosim.schemas.agent_output import AgentOutput


class ActionPlan(BaseModel):
    """
    Aksiyon planı: Sistemin önerdiği spesifik aksiyonlar.

    Attributes:
        priority: Öncelik seviyesi ("immediate", "high", "medium", "low").
        action: Aksiyon açıklaması.
        rationale: Aksiyonun gerekçesi.
    """

    priority: str = Field(..., description="Öncelik seviyesi")
    action: str = Field(..., description="Aksiyon açıklaması")
    rationale: str = Field(..., description="Aksiyonun gerekçesi")


class FinalReport(BaseModel):
    """
    Nihai risk analizi raporu.

    Bu rapor, tüm ajanların çıktılarının sentezlenmesi sonucu üretilir
    ve kullanıcıya sunulur.

    Attributes:
        query: Kullanıcının orijinal sorgusu.
        timestamp: Raporun üretildiği zaman damgası.
        final_risk_level: Nihai risk seviyesi ("critical", "high", "medium", "low").
        overall_confidence: Genel güven skoru (0-1 arası).
        strategic_rationale: Stratejik gerekçe (en az 300 kelime).
        action_plan: Spesifik aksiyon planı (liste).
        agent_contributions: Her ajanın katkısı (detaylı).
        data_freshness_score: Veri tazeliği skoru (0-1 arası).
        warnings: Uyarılar ve sınırlamalar (liste).
    """

    query: str = Field(..., description="Kullanıcı sorgusu")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Zaman damgası")
    final_risk_level: str = Field(..., description="Nihai risk seviyesi")
    overall_confidence: float = Field(..., ge=0.0, le=1.0, description="Genel güven skoru")
    strategic_rationale: str = Field(..., min_length=300, description="Stratejik gerekçe")
    action_plan: List[ActionPlan] = Field(default_factory=list, description="Aksiyon planı")
    agent_contributions: List[AgentOutput] = Field(default_factory=list, description="Ajan katkıları")
    data_freshness_score: float = Field(..., ge=0.0, le=1.0, description="Veri tazeliği skoru")
    warnings: List[str] = Field(default_factory=list, description="Uyarılar ve sınırlamalar")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Ek metadata")

    class Config:
        """Pydantic yapılandırması."""

        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

