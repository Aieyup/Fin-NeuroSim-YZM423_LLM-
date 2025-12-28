# Google Colab Kurulum Rehberi

Bu rehber, Fin-NeuroSim 2.0 projesini Google Colab ortamında çalıştırmak için adım adım talimatlar içerir.

## 📋 Ön Gereksinimler

1. **Google Colab hesabı** (ücretsiz)
2. **Google Drive** (proje dosyalarını saklamak için)
3. **HuggingFace hesabı** (modellere erişim için)
4. **API anahtarları:**
   - Tavily API key
   - Alpha Vantage API key (opsiyonel)
   - FRED API key (opsiyonel)

## 🚀 Kurulum Adımları

### 1. Proje Dosyalarını Google Drive'a Yükleyin

1. Google Drive'ınızı açın
2. `MyDrive/LLM_Proje/` klasörü oluşturun
3. Tüm `fin_neurosim/` klasörünü buraya yükleyin
4. Yapı şöyle olmalı:
   ```
   /content/drive/MyDrive/LLM_Proje/
   └── fin_neurosim/
       ├── core/
       ├── llm/
       ├── agents/
       ├── data_sources/
       ├── prompts/
       ├── schemas/
       ├── utils/
       └── mvp/
   ```

### 2. Colab Notebook'u Oluşturun

Yeni bir Colab notebook oluşturun ve aşağıdaki hücreleri sırayla çalıştırın:

#### Hücre 1: Gerekli Kütüphaneleri İçe Aktar

```python
import asyncio
import os
import sys
import nest_asyncio
import torch
from pathlib import Path

# Colab'da async için gerekli
nest_asyncio.apply()

# CUDA kontrolü
print(f"CUDA Mevcut: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
```

#### Hücre 2: Google Drive'ı Bağla

```python
from google.colab import drive
drive.mount('/content/drive')

# Proje path'i
base_path = '/content/drive/MyDrive/LLM_Proje/fin_neurosim'
print(f"Proje path: {base_path}")
```

#### Hücre 3: HuggingFace Token Ayarla

```python
from huggingface_hub import login

# Token'ı ayarla
hf_token = os.getenv('HF_TOKEN')
if not hf_token:
    hf_token = input("HuggingFace token'ınızı girin: ")
    os.environ['HF_TOKEN'] = hf_token

login(token=hf_token)
```

#### Hücre 4: Bağımlılıkları Yükle

```python
!pip install -q transformers>=4.35.0 torch>=2.0.0 bitsandbytes>=0.41.0 accelerate>=0.24.0
!pip install -q sentence-transformers>=2.2.0 scikit-learn>=1.3.0
!pip install -q httpx aiohttp requests pydantic pydantic-settings
!pip install -q alpha-vantage fredapi tavily-python python-dateutil nest-asyncio

print("✅ Bağımlılıklar yüklendi")
```

#### Hücre 5: Proje Yapısını Kontrol Et

```python
# Klasörleri oluştur
folders = [
    "core", "llm", "agents", "data_sources",
    "prompts", "schemas", "utils", "mvp"
]

for folder in folders:
    folder_path = Path(base_path) / folder
    folder_path.mkdir(parents=True, exist_ok=True)
    print(f"✅ {folder_path}")

# __init__.py dosyalarını oluştur
init_files = [
    f"{base_path}/__init__.py",
    f"{base_path}/core/__init__.py",
    f"{base_path}/llm/__init__.py",
    f"{base_path}/agents/__init__.py",
    f"{base_path}/data_sources/__init__.py",
    f"{base_path}/prompts/__init__.py",
    f"{base_path}/schemas/__init__.py",
    f"{base_path}/utils/__init__.py",
    f"{base_path}/mvp/__init__.py",
]

for init_file in init_files:
    path = Path(init_file)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('"""Package init file."""\n', encoding='utf-8')
        print(f"✅ Oluşturuldu: {init_file}")

# Path'i sys.path'e ekle
if base_path not in sys.path:
    sys.path.insert(0, base_path)
    print(f"✅ Path eklendi: {base_path}")
```

#### Hücre 6: API Anahtarlarını Ayarla

```python
# API anahtarlarını ayarla
os.environ['TAVILY_API_KEY'] = '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'
os.environ['ALPHA_VANTAGE_API_KEY'] = 'your_alpha_vantage_key_here'  # Kendi anahtarınızı ekleyin
os.environ['FRED_API_KEY'] = 'your_fred_key_here'  # Kendi anahtarınızı ekleyin

print("✅ API anahtarları ayarlandı")
```

#### Hücre 7: Projeyi İçe Aktar ve Çalıştır

```python
# Projeyi içe aktar
try:
    from fin_neurosim.core.orchestrator_hf import FinNeuroSimOrchestratorHF
    print("✅ Fin-NeuroSim 2.0 başarıyla yüklendi")
except ImportError as e:
    print(f"❌ Import hatası: {e}")
    print(f"\nLütfen proje dosyalarının doğru yerde olduğundan emin olun.")
    print(f"Beklenen path: {base_path}")

# Async fonksiyon
async def run_analysis():
    """Risk analizi çalıştırır."""
    try:
        print("\n🚀 Orchestrator başlatılıyor...")
        orchestrator = FinNeuroSimOrchestratorHF()
        
        print("📊 Analiz başlatılıyor...")
        result = await orchestrator.process_query(
            "TSLA hissesi için risk analizi yap"
        )
        
        return result
    except Exception as e:
        print(f"❌ Hata: {e}")
        import traceback
        traceback.print_exc()
        raise

# Çalıştır
result = asyncio.run(run_analysis())

# Sonuçları göster
print("\n" + "="*80)
print("📊 RİSK ANALİZİ SONUÇLARI")
print("="*80)
print(f"\nSorgu: {result.query}")
print(f"Risk Seviyesi: {result.final_risk_level.upper()}")
print(f"Güven Skoru: {result.overall_confidence:.2%}")
print(f"\nStratejik Gerekçe:\n{result.strategic_rationale[:500]}...")
print(f"\nAksiyon Planı:")
for i, action in enumerate(result.action_plan[:3], 1):
    print(f"  {i}. [{action.priority.upper()}] {action.action}")
```

## ⚠️ Önemli Notlar

### GPU Kullanımı

- Colab'da GPU kullanmak için: **Runtime → Change runtime type → GPU**
- T4 GPU genellikle yeterlidir (16GB VRAM)
- Model yükleme sırasında VRAM kullanımını izleyin

### Model İndirme

- İlk çalıştırmada modeller HuggingFace'den indirilecek
- İndirme süresi internet hızınıza bağlıdır
- Modeller cache'lenecek, sonraki çalıştırmalarda daha hızlı olacak

### Hata Ayıklama

Eğer import hatası alırsanız:

1. Proje dosyalarının doğru yerde olduğundan emin olun
2. `__init__.py` dosyalarının mevcut olduğunu kontrol edin
3. Path'in doğru eklendiğini kontrol edin

### API Anahtarları

- Tavily API key zorunludur
- Alpha Vantage ve FRED API key'leri opsiyoneldir
- API key'lerinizi güvenli tutun, asla commit etmeyin

## 🔧 Sorun Giderme

### CUDA Hatası

```python
# CUDA cache'i temizle
import torch
torch.cuda.empty_cache()
```

### Import Hatası

```python
# Path'i tekrar kontrol et
import sys
print(sys.path)
print(f"Base path var mı: {base_path in sys.path}")
```

### Model Yükleme Hatası

```python
# HuggingFace token'ı kontrol et
import os
print(f"HF_TOKEN var mı: {'HF_TOKEN' in os.environ}")
```

## 📚 Ek Kaynaklar

- [HuggingFace Model Hub](https://huggingface.co/models)
- [Google Colab Dokümantasyonu](https://colab.research.google.com/)