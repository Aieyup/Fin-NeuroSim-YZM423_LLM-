"""
Model Loader: Sequential model loading ve VRAM yönetimi.

Bu modül, HuggingFace modellerini 4-bit quantization ile
yükler ve VRAM'i verimli kullanır.
"""

import gc
import os
import torch
from typing import Optional, Dict, Any
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

from fin_neurosim.utils.logging import setup_logger

logger = setup_logger(__name__)


class ModelLoader:
    """
    HuggingFace model yükleyici ve yöneticisi.

    Sequential loading sağlar ve VRAM'i optimize eder.
    """

    def __init__(self):
        """ModelLoader'ı başlatır."""
        self.current_model = None
        self.current_tokenizer = None
        self.current_model_name = None

    def load_model(
        self,
        model_name: str,
        device_map: str = "auto",
        max_memory: Optional[Dict[int, str]] = None,
        use_auth_token: Optional[str] = None
    ) -> tuple:
        """
        Model ve tokenizer'ı yükler (4-bit quantization ile).

        Args:
            model_name: HuggingFace model adı.
            device_map: Cihaz haritası (default: "auto").
            max_memory: Maksimum bellek sınırları (opsiyonel).
            use_auth_token: HuggingFace token (opsiyonel, env'den alınır).

        Returns:
            (model, tokenizer) tuple'ı.
        """
        # Eğer başka bir model yüklüyse, önce temizle
        if self.current_model is not None:
            logger.info(f"Önceki model temizleniyor: {self.current_model_name}")
            self.unload_model()

        logger.info(f"Model yükleniyor: {model_name}")

        # 4-bit quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16
        )

        try:
            # HuggingFace token'ı al
            if use_auth_token is None:
                use_auth_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
            
            # Tokenizer'ı yükle
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    token=use_auth_token
                )
            except Exception as e:
                # Bazı modeller için tokenizer farklı olabilir
                logger.warning(f"Tokenizer yükleme hatası: {e}, alternatif deneniyor...")
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    token=use_auth_token,
                    use_fast=False  # Fast tokenizer yerine normal tokenizer
                )
            
            # EOS token yoksa ekle
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Model'i yükle (4-bit)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map=device_map,
                max_memory=max_memory,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                token=use_auth_token
            )

            # Model'i eval moduna al
            model.eval()

            self.current_model = model
            self.current_tokenizer = tokenizer
            self.current_model_name = model_name

            logger.info(f"Model başarıyla yüklendi: {model_name}")

            return (model, tokenizer)

        except Exception as e:
            logger.error(f"Model yükleme hatası: {e}")
            raise

    def unload_model(self) -> None:
        """
        Mevcut modeli ve tokenizer'ı bellekten kaldırır.

        CUDA cache'i de temizler.
        """
        if self.current_model is not None:
            logger.info("Model bellekten kaldırılıyor...")

            # Model'i CPU'ya taşı
            if hasattr(self.current_model, 'cpu'):
                self.current_model = self.current_model.cpu()

            # Model ve tokenizer'ı sil
            del self.current_model
            del self.current_tokenizer

            self.current_model = None
            self.current_tokenizer = None
            self.current_model_name = None

            # Garbage collection
            gc.collect()

            # CUDA cache temizle
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("CUDA cache temizlendi")

            logger.info("Model başarıyla kaldırıldı")

    def get_current_model(self) -> Optional[tuple]:
        """
        Mevcut yüklü model ve tokenizer'ı döndürür.

        Returns:
            (model, tokenizer) tuple'ı veya None.
        """
        if self.current_model is not None and self.current_tokenizer is not None:
            return (self.current_model, self.current_tokenizer)
        return None

    def is_model_loaded(self, model_name: Optional[str] = None) -> bool:
        """
        Belirtilen modelin yüklü olup olmadığını kontrol eder.

        Args:
            model_name: Kontrol edilecek model adı (opsiyonel).

        Returns:
            Model yüklüyse True, değilse False.
        """
        if model_name is None:
            return self.current_model is not None
        return self.current_model_name == model_name

