# Fin-NeuroSim 2.0

**Web-Native, Asynchronous, Minority-Aware Financial Intelligence System**  
**HuggingFace Uyumlu (16GB VRAM, 4-bit Quantization)**
---

## Özet (Abstract)

**Fin-NeuroSim 2.0**, finansal piyasalarda nadir görülen ancak yüksek etki potansiyeline sahip kriz ve anomali sinyallerini erken aşamada tespit etmek amacıyla geliştirilmiş, web-tabanlı, asenkron, çok ajanlı bir finansal zeka sistemidir. Sistem, geleneksel makine öğrenmesi modellerinin çoğunluk sınıfı (normal piyasa koşulları) lehine azınlık sınıfı sinyallerini (kriz göstergeleri) bastırdığı **sınıf dengesizliği (class-imbalanced) finansal veri** problemini ele almaktadır.

**Problem Tanımı:** Finansal piyasalarda ciddi bir sınıf dengesizliği görülmektedir; kriz olayları toplam piyasa gözlemlerinin küçük bir kısmını temsil eder. Bu tür veriler üzerinde eğitilen geleneksel modeller genellikle erken uyarı sinyallerini tespit edemez ve bu durum önemli finansal kayıplara yol açar. Büyük dil modellerini (LLM) bu görev için fine-tuning yapmak hesaplama açısından maliyetlidir ve yeni olaylar ortaya çıktıkça sürekli yeniden eğitim gerektirir.

**Çözüm Yaklaşımı:** Bu proje, ValizadehAslani ve arkadaşları (2022) tarafından önerilen metodolojiden ilham alarak **In-Context Learning (ICL)** yoluyla **İki Aşamalı Fine-Tuning Simülasyonu** uygular. Geleneksel fine-tuning yerine, sistem iki aşamalı fine-tuning sürecini simüle etmek için stratejik prompt mühendisliği ve bağlamsal yeniden ağırlıklandırma kullanır: (1) **Aşama-1: Azınlık Odaklı Bağlamsal Yeniden Ağırlıklandırma** (pozisyonel önyargı ve tekrar ağırlıklandırması kullanarak) ve (2) **Aşama-2: Bayesci Sentez** (dinamik güven skoru tabanlı ağırlıklandırma ile).

**Ana Katkılar:**
- ICL kullanarak sıfır maliyetli fine-tuning simülasyonu, pahalı model yeniden eğitimine olan ihtiyacı ortadan kaldırır
- Pozisyonel önyargı ve tekrar ağırlıklandırma stratejileriyle azınlık odaklı analiz
- Web-tabanlı veri toplama yoluyla sıfır günlük (zero-day) olaylara gerçek zamanlı adaptasyon
- Sıralı model yükleme ile VRAM-verimli mimari (16GB GPU uyumlu)
- Her aşamada detaylı gerekçelendirme ile açıklanabilir karar verme

---

## 1. Giriş ve Motivasyon

### 1.1 Problem Tanımı

Finansal piyasalarda aşırı sınıf dengesizliği görülmektedir. Normal piyasa koşulları veri dağılımına hakimken, nadir kriz olayları—piyasa çöküşleri, ani volatilite artışları veya sistemik risk sinyalleri gibi—küçük bir azınlık sınıfını temsil eder. Bu tür dengesiz veri setleri üzerinde eğitilen geleneksel makine öğrenmesi modelleri genellikle:

- **Azınlık sınıfı sinyallerini bastırır** ve çoğunluk desenlerini tercih eder
- **Potansiyel krizlerin erken uyarı işaretlerini tespit edemez**
- **Yeni piyasa koşullarına adapte olmak için pahalı fine-tuning gerektirir**
- **Sıfır günlük (zero-day) olaylara gerçek zamanlı adaptasyon sağlayamaz**

Bu zorluk, finansal risk analizinde özellikle kritiktir; bir kriz sinyalini kaçırmak felaket boyutunda kayıplara yol açabilirken, yanlış pozitifler yanlış negatiflerden daha kabul edilebilirdir.

### 1.2 Motivasyon

Bu projenin motivasyonu üç temel gözlemden kaynaklanmaktadır:

1. **Fine-Tuning Sınırlamaları:** Finansal analiz için büyük dil modellerinin geleneksel fine-tuning'i hesaplama açısından maliyetlidir, etiketli veri setleri gerektirir ve yeniden eğitim olmadan yeni olaylara adapte olamaz.

2. **Azınlık Sınıfı Bastırılması:** Dengesiz veriler üzerinde eğitilen standart modeller doğal olarak çoğunluk desenlerine yönelir ve bu durum onları kriz tespiti için etkisiz hale getirir.

3. **Gerçek Zamanlı Adaptasyon İhtiyacı:** Finansal piyasalar hızla evrilir ve sıfır günlük olaylar, model yeniden eğitim döngülerini beklemeden anında analiz gerektirir.

### 1.3 Çözüm Yaklaşımı

**Fin-NeuroSim 2.0**, bu zorlukları **In-Context Learning (ICL)** kullanarak **İki Aşamalı Fine-Tuning Simülasyonu** yaklaşımıyla ele alır. Sistem:

- Model ağırlıklarını değiştirmeden **fine-tuning davranışını simüle eder**
- Stratejik bağlam mühendisliği yoluyla **azınlık sınıfı sinyallerini önceliklendirir**
- Web-tabanlı veri toplama yoluyla **yeni olaylara gerçek zamanlı adapte olur**
- Tüketici seviyesi donanımda (16GB VRAM) **verimli çalışır**

### 1.4 In-Context Learning Avantajları

ICL yaklaşımı, geleneksel fine-tuning'e göre birkaç avantaj sunar:

- **Sıfır Fine-Tuning Maliyeti:** Model ağırlık güncellemeleri gerekmez, hesaplama yükü ortadan kalkar
- **Sorgu Bazlı Özelleştirme:** Her analiz belirli piyasa koşullarına göre uyarlanabilir
- **Anında Adaptasyon:** Yeni olaylar yeniden eğitim olmadan anında dahil edilebilir
- **Açıklanabilir Kararlar:** Tüm gerekçelendirme prompt bağlamında görünür, tam şeffaflık sağlar
- **Kaynak Verimliliği:** Önceden eğitilmiş modellerle çalışır, yalnızca çıkarım yetenekleri gerektirir

---

## 2. İlgili Çalışmalar ve Teorik Temel

### 2.1 İki Aşamalı Fine-Tuning Metodolojisi

Bu proje, ValizadehAslani ve arkadaşları (2022) tarafından sınıf dengesizliği olan verilerden öğrenme için önerilen **İki Aşamalı Fine-Tuning** metodolojisinden ilham almıştır. Orijinal metodoloji şunlardan oluşur:

**Aşama-1: Sınıf Dengeli Yeniden Ağırlıklandırma Kaybı**
- Fine-tuning sırasında kayıp fonksiyonuna sınıf dengeli yeniden ağırlıklandırma uygular
- Eğitim sırasında azınlık sınıfı örneklerine daha yüksek ağırlık verir
- Çoğunluk ve azınlık sınıfları arasındaki öğrenme sürecini dengelemeyi amaçlar

**Aşama-2: Standart Fine-Tuning**
- Yeniden ağırlıklandırılmış model üzerinde standart fine-tuning gerçekleştirir
- Modelin sınıflar arası ayrım yapma yeteneğini geliştirir
- Genel sınıflandırma performansını iyileştirir

**Bizim Adaptasyonumuz:**
Model ağırlıklarını fine-tuning yoluyla değiştirmek yerine, bu iki aşamalı süreci **In-Context Learning** kullanarak simüle ediyoruz:

- **Aşama-1 Simülasyonu:** Sınıf dengeli yeniden ağırlıklandırmayı simüle etmek için pozisyonel önyargı (anomali verilerini bağlam başına yerleştirme) ve tekrar ağırlıklandırması (kritik sinyalleri 3x tekrarlama) uygularız
- **Aşama-2 Simülasyonu:** İyileştirme aşamasını simüle etmek için dinamik güven skoru tabanlı ağırlıklandırma ile Bayesci sentez kullanırız

Bu adaptasyon, fine-tuning'in hesaplama maliyeti olmadan benzer faydaları elde etmemize olanak tanır.

### 2.2 Sınıf Dengesizliği Öğrenme Literatürü

Sınıf dengesizliği öğrenme, makine öğrenmesinde iyi çalışılmış bir problemdir. Yaygın yaklaşımlar şunları içerir:

- **Yeniden örnekleme teknikleri** (azınlığı fazla örnekleme, çoğunluğu az örnekleme)
- **Maliyet duyarlı öğrenme** (azınlık sınıfı yanlış sınıflandırmasına daha yüksek maliyet atama)
- **Topluluk yöntemleri** (farklı önyargılara sahip çoklu modelleri birleştirme)
- **Sentetik veri üretimi** (SMOTE, ADASYN)

Bizim yaklaşımımız, veri seviyesi manipülasyonu yerine **bağlamsal yeniden ağırlıklandırma**ya prompt seviyesinde odaklanarak farklılaşır ve bu durum onu LLM tabanlı analiz için uygun hale getirir.

### 2.3 In-Context Learning (ICL)

In-Context Learning (Brown ve ark., 2020), büyük dil modellerinin model ağırlıklarını güncellemeden prompt bağlamında sağlanan örneklere dayalı olarak davranışlarını uyarlama yeteneğini ifade eder. Temel özellikler:

- **Az örnekli öğrenme:** Modeller bağlamda az sayıda örnekten öğrenebilir
- **Görev adaptasyonu:** Davranış sorgu bazında özelleştirilebilir
- **Sıfır örnekli genelleme:** Eğitim sırasında görülmeyen yeni görevleri ele alabilir

Sistemimiz, ICL'yi fine-tuning davranışını simüle etmek için kullanır ve model ağırlık güncellemeleri olmadan azınlık odaklı analize olanak tanır.



## 3. Metodoloji

### 3.1 ICL ile İki Aşamalı Fine-Tuning Simülasyonu

Sistem, ValizadehAslani ve arkadaşları (2022) tarafından önerilen iki aşamalı fine-tuning sürecini, gerçek model ağırlık güncellemeleri yerine **In-Context Learning** kullanarak simüle eder. Bu yaklaşım, model güncellemeleri için sıfır hesaplama yükü korurken fine-tuning'e benzer faydalar sağlar.

### 3.2 Aşama-1: Azınlık Odaklı Bağlamsal Yeniden Ağırlıklandırma

**Aşama-1**, orijinal metodolojideki "Sınıf Dengeli Yeniden Ağırlıklandırma Kaybı" aşamasına karşılık gelir. Bunu stratejik bağlam mühendisliği yoluyla simüle ediyoruz:

#### 3.2.1 Pozisyonel Önyargı

- **Anomali veri yerleşimi:** Azınlık sınıfı sinyalleri (anomaliler, kriz göstergeleri) **bağlamın başına** yerleştirilir
- **Gerekçe:** LLM'ler yakınlık önyargısı sergiler ve bağlamın başındaki bilgilere daha fazla ağırlık verir
- **Uygulama:** Risk sinyalleri, volatilite artışları ve anomali tespitleri bağlam sıralamasında önceliklendirilir

#### 3.2.2 Tekrar Ağırlıklandırması

- **Kritik sinyal amplifikasyonu:** Önemli azınlık sınıfı sinyalleri bağlamda **3 kez tekrarlanır**
- **Gerekçe:** Tekrar, transformer mimarilerinde dikkat ağırlığını artırır
- **Uygulama:** Yüksek risk göstergeleri, kriz uyarıları ve anomali desenleri açıkça tekrarlanır

#### 3.2.3 Sanal Ajanlar Mimarisi

Farklı perspektifler için ayrı modeller eğitmek yerine, prompt mühendisliği ile uygulanan **sanal ajanlar** kullanıyoruz:

- **Risk Lens:** Anomali tespiti, volatilite artışları ve olağandışı piyasa desenlerine odaklanır
- **Makro Lens:** Sistemik risk, makroekonomik göstergeler ve politika etkilerini analiz eder
- **Sentiment Lens:** Piyasa duygusu, haber tonu ve yatırımcı psikolojisini değerlendirir
- **Teknik Lens:** Teknik göstergeler, trend desenleri ve volatilite metriklerini inceler

**Model:** `mistralai/Mistral-7B-v0.3` (temel model, 4-bit quantization)
- **Neden temel model?** Temel modeller, instruction-tuning önyargıları olmadan daha esnek prompt kontrolü sağlar
- **Neden Mistral-7B?** Gerçek zamanlı analiz için kapasite ve çıkarım hızı arasında optimal denge

### 3.3 Aşama-2: Bayesci Sentez

**Aşama-2**, "Standart Fine-Tuning" iyileştirme aşamasına karşılık gelir. Bunu güven skoru tabanlı sentez yoluyla simüle ediyoruz:

#### 3.3.1 Dinamik Ağırlıklandırma

- **Güven skoru tabanlı toplama:** Aşama-1 ajan çıktıları güven skorlarına göre ağırlıklandırılır
- **Ağırlık aralığı:** 0.3 ile 0.7 arası, güvene göre dinamik olarak ayarlanır
- **Azınlık önceliklendirmesi:** Aşama-1 uyarıları orta düzeyde güvenle bile daha yüksek ağırlık alır

#### 3.3.2 Azınlık vs Çoğunluk Dengesi

- **Uyarılara yönelik önyargı:** Aşama-1 azınlık sınıfı sinyalleri (kriz uyarıları) sentezde daha yüksek ağırlık alır
- **Gerekçe:** Yanlış negatifler (bir krizi kaçırma) yanlış pozitiflerden daha maliyetlidir
- **Uygulama:** Güven > 0.4 olan risk sinyalleri 0.6+ ağırlıklandırılırken, normal sinyaller > 0.7 güven gerektirir

#### 3.3.3 Stratejik Gerekçelendirme

- **Minimum 300 kelime:** Aşama-2 çıktısı detaylı stratejik gerekçe içerir (minimum 300 kelime)
- **Açıklanabilir kararlar:** Her risk değerlendirmesi kapsamlı gerekçelendirme içerir
- **Uygulanabilir içgörüler:** Sentez, öncelik seviyeleriyle somut aksiyon planları üretir

**Model:** `mistralai/Mistral-7B-Instruct-v0.2` (instruction-tuned, 4-bit quantization)
- **Neden instruction-tuned?** Sentez, karar verme ve yapılandırılmış çıktı üretiminde daha iyidir
- **Neden aynı model ailesi?** Tutarlılığı sağlar ve aşamalar arası bağlam kaymasını azaltır

### 3.4 Orijinal Metodoloji ile Bağlantı

| Orijinal İki Aşamalı Fine-Tuning | Bizim ICL Simülasyonumuz |
|-------------------------------|-------------------|
| Aşama-1: Sınıf dengeli yeniden ağırlıklandırma kaybı | Aşama-1: Pozisyonel önyargı + tekrar ağırlıklandırması |
| Aşama-2: Standart fine-tuning | Aşama-2: Dinamik ağırlıklandırma ile Bayesci sentez |
| Model ağırlık güncellemeleri | Bağlam mühendisliği |
| Eğitim verisi gerekli | Gerçek zamanlı web verisi |
| Pahalı hesaplama | Sıfır fine-tuning maliyeti |



## 4. Model Seçimi ve Gerekçeleri

### 4.1 Aşama-1 Modeli: Mistral-7B-v0.3

**Seçim:** `mistralai/Mistral-7B-v0.3` (temel model, 4-bit quantization)

**Teknik Gerekçe:**

1. **Temel Model Avantajı:**
   - **Esnek prompt kontrolü:** Temel modeller (instruction-tuning olmadan) promptlar aracılığıyla davranışın daha doğrudan manipülasyonuna izin verir
   - **Instruction önyargısı yok:** Instruction-tuned modeller, azınlık odaklı yeniden ağırlıklandırma stratejimizle çakışabilecek önceden öğrenilmiş desenlere sahiptir
   - **Sanal ajanlar için daha iyi:** Birden fazla "lens" perspektifi, belirli instruction formatları için ayarlanmamış temel modellerle daha iyi çalışır

2. **7B Parametre Boyutu:**
   - **Yeterli kapasite:** 7B parametre, finansal analiz için yeterli akıl yürütme yeteneği sağlar
   - **Hızlı çıkarım:** 13B+ modellerden daha küçük, gerçek zamanlı analiz sağlar (aşama başına 30-60 saniye)
   - **VRAM verimliliği:** 4-bit quantization ile 8GB VRAM'e rahatça sığar

3. **Mistral Mimarisi:**
   - **Güçlü performans:** Mistral modelleri akıl yürütme görevlerinde mükemmel performans gösterir
   - **Verimli dikkat:** Sliding Window Attention (SWA) hesaplama maliyetini azaltır
   - **Açık kaynak:** Tamamen açık kaynak, lisans kısıtlaması yok

4. **4-bit Quantization:**
   - **VRAM kısıtı:** 16GB GPU'larda (T4, tüketici seviyesi) çalışmayı mümkün kılar
   - **Minimal kalite kaybı:** 4-bit quantization, çıkarım için FP16 performansının >%95'ini korur
   - **bitsandbytes entegrasyonu:** Kanıtlanmış güvenilirliğe sahip endüstri standardı quantization kütüphanesi

### 4.2 Aşama-2 Modeli: Mistral-7B-Instruct-v0.2

**Seçim:** `mistralai/Mistral-7B-Instruct-v0.2` (instruction-tuned, 4-bit quantization)

**Teknik Gerekçe:**

1. **Instruction-Tuned Avantajı:**
   - **Daha iyi sentez:** Instruction-tuned modeller, çoklu girdileri tutarlı çıktılara birleştirmede mükemmeldir
   - **Yapılandırılmış çıktı:** Format gereksinimlerini (JSON, yapılandırılmış gerekçelendirme) takip etmede daha iyidir
   - **Karar verme:** Instruction-following üzerinde önceden eğitilmiş, karar verme ve açıklamada daha iyidir

2. **Aynı Model Ailesi:**
   - **Tutarlılık:** Her iki aşama için Mistral kullanmak tutarlı tokenization ve embedding alanı sağlar
   - **Azaltılmış bağlam kayması:** Benzer mimariler aşamalar arası potansiyel tutarsızlıkları azaltır
   - **Birleşik ekosistem:** Bakım ve hata ayıklama daha kolay

3. **Yedekleme Mekanizması:**
   - **Birincil:** `mistralai/Mistral-7B-Instruct-v0.2`
   - **Yedek 1:** `mistralai/Mistral-7B-v0.3` (instruction modeli mevcut değilse)
   - **Yedek 2:** `microsoft/Phi-3-medium-4k-instruct` (alternatif mimari)
   - **Gerekçe:** Birincil model mevcut olmasa bile sistem güvenilirliğini sağlar

### 4.3 Bağlam Sıkıştırma Modeli: FinBERT

**Seçim:** `ProsusAI/finbert` (FP32, CPU tabanlı)

**Teknik Gerekçe:**

1. **Finansal Alan Uzmanlaşması:**
   - **Finansal metin üzerinde önceden eğitilmiş:** FinBERT özellikle finansal haberler, raporlar ve piyasa verileri üzerinde eğitilmiştir
   - **Alan embedding'leri:** Finansal kavramlar için daha ilgili semantik temsiller sağlar
   - **Daha iyi ilgili skorlama:** Finansal terminoloji genel amaçlı modellerden daha iyi anlaşılır

2. **CPU Tabanlı İşlem:**
   - **Sıfır VRAM kullanımı:** CPU'da çalışır, LLM çıkarımı için GPU belleğini serbest bırakır
   - **Paralel işleme:** GPU model yükleme ile eşzamanlı çalışabilir
   - **Kaynak verimliliği:** Küçük model (~110M parametre) CPU'da verimli çalışır

3. **Semantik Sıkıştırma:**
   - **İlgili skorlama:** FinBERT embedding'leri finansal verilerin doğru ilgili skorlamasını sağlar
   - **Top-K seçimi:** Bağlam için en önemli bilgi parçalarını belirler
   - **Token optimizasyonu:** Kritik bilgileri korurken bağlamı ~3000 token'dan ~1200 token'a düşürür

### 4.4 Model Yükleme Stratejisi

**Sıralı Yükleme:**
- Aynı anda VRAM'de yalnızca bir model yüklü
- Aşama-2 modeli yüklenmeden önce Aşama-1 modeli kaldırılır
- Aşamalar arası otomatik CUDA önbellek temizleme
- Model kaldırma sonrası garbage collection

**Gerekçe:** Model yükü dahil olmak üzere VRAM kapasitesini asla aşmayarak 16GB GPU'larda çalışmayı mümkün kılar.


## 5. Veri Kaynakları ve Mantığı

### 5.1 Veri Kaynağı Genel Bakış

Sistem, finansal risk analizinde her biri belirli bir amaç için kullanılan üç birincil veri kaynağı kullanır:

| Veri Kaynağı | Amaç | Neden Kullanılıyor | Öncelik |
|------------|---------|----------|----------|
| **Tavily AI** | Haber ve anomali tespiti | Sıfır günlük olay tespiti, gerçek zamanlı haberler | Zorunlu |
| **Alpha Vantage** | Piyasa verisi | Fiyat, hacim, volatilite metrikleri | Opsiyonel |
| **FRED API** | Makroekonomik göstergeler | Sistemik risk değerlendirmesi | Opsiyonel |




Sistem, detaylı ve formatlanmış bir rapor üretir. Rapor şu bölümleri içerir:

### Rapor Bölümleri

1. **Risk Özeti**: Risk seviyesi, güven skoru, veri tazeliği
2. **Stratejik Gerekçe**: Tam analiz (kesilmeden, paragraflar halinde)
3. **Aksiyon Planı**: Öncelik sırasıyla aksiyonlar ve gerekçeleri
4. **Ajan Katkıları**: Her ajanın detaylı analizi
5. **Uyarılar**: Sistem sınırlamaları ve uyarılar

### Örnek Çıktı

```
====================================================================================================
                          📊 FIN-NEUROSIM 2.0 - RİSK ANALİZİ RAPORU
====================================================================================================

📅 Rapor Tarihi: 2025-12-28 13:06:19
🔍 Analiz Sorgusu: TSLA hissesi için risk analizi yap

----------------------------------------------------------------------------------------------------
                                 🎯 RİSK ÖZETİ
----------------------------------------------------------------------------------------------------

🟠 Risk Seviyesi: HIGH
📊 Güven Skoru: 60.0%
📈 Veri Tazeliği: 50.0%

                              📝 STRATEJİK GEREKÇE VE ANALİZ
----------------------------------------------------------------------------------------------------

   [Tam stratejik gerekçe, kesilmeden, paragraflar halinde...]
   [90 karakter genişliğinde, okunaklı formatlanmış metin]

                                 🎯 AKSİYON PLANI
----------------------------------------------------------------------------------------------------

🟠 Aksiyon 1: [HIGH]
   📌 Piyasayı yakından izleyin ve risk yönetimi protokollerini aktifleştirin
   💡 Gerekçe: Yüksek risk seviyesi tespit edildi, dikkatli olunmalı

🟡 Aksiyon 2: [MEDIUM]
   📌 [Aksiyon açıklaması]
   💡 Gerekçe: [Gerekçe açıklaması]

                        🤖 AJAN KATKILARI VE ANALİZLER
----------------------------------------------------------------------------------------------------

📊 Stage1Mistral:
   Risk Seviyesi: MEDIUM
   Güven Skoru: 60.0%
   Ana Faktörler:
      • Anomali tespit edildi
      • Kriz sinyali
      • [Diğer faktörler...]
   Özet: [Ajan reasoning özeti, 300 karakter]
```

## 6. Avantajlar

### 6.1 Sıfır Fine-Tuning Maliyeti

**Geleneksel Yaklaşım:**
- Fine-tuning GPU saatleri gerektirir (pahalı)
- Yeni olaylar için sürekli yeniden eğitim gerekir
- Model depolama ve sürümleme yükü
- Dağıtım karmaşıklığı

**Bizim ICL Yaklaşımımız:**
- **Sıfır model ağırlık güncellemesi:** Fine-tuning gerekmez
- **Sorgu bazlı özelleştirme:** Her analiz belirli koşullara göre uyarlanır
- **Yeniden eğitim döngüsü yok:** Yeni olaylar anında dahil edilir
- **Daha düşük operasyonel maliyet:** Yalnızca çıkarım gerekir, eğitim altyapısı gerekmez

**Maliyet Karşılaştırması:**
- Fine-tuning: Model güncellemesi başına ~$50-200 (bulut GPU saatleri)
- ICL: $0 (önceden eğitilmiş modeller kullanır, yalnızca çıkarım)

### 6.2 Gerçek Zamanlı Adaptasyon

**Web-Tabanlı Veri Toplama:**
- **Sıfır günlük olaylar:** Sistem son dakika haberlerine anında adapte olur
- **Veri gecikmesi yok:** Gerçek zamanlı API entegrasyonu (Tavily, Alpha Vantage)
- **Dinamik bağlam:** Her sorgu mevcut verilerden taze bağlam oluşturur
- **Piyasa duyarlılığı:** Olaylar gelişirken analiz edebilir

**Örnek:** Saat 14:00'te gerçekleşen bir piyasa çöküşü, model yeniden eğitimini beklemeden saat 14:05'te en son haberler ve verilerle analiz edilebilir.

### 6.3 VRAM-Verimli Mimari

**Sıralı Model Yükleme:**
- Aynı anda VRAM'de yalnızca bir model
- Aşamalar arası otomatik önbellek temizleme
- 4-bit quantization bellek ayak izini %75 azaltır

**Donanım Gereksinimleri:**
- **Minimum:** 16GB VRAM GPU (T4, tüketici seviyesi)
- **Çoklu GPU gerekmez:** Tek GPU yeterli
- **Tüketici donanımı uyumlu:** Oyun GPU'larında çalışır (RTX 3090, RTX 4090)

**Karşılaştırma:**
- Geleneksel yaklaşım: Genellikle 40GB+ VRAM (A100) veya çoklu GPU kurulumu gerektirir
- Bizim yaklaşımımız: 16GB VRAM yeterli, araştırmacılar ve uygulayıcılar için erişilebilir

### 6.4 Açıklanabilir Karar Verme

**Tam Şeffaflık:**
- **Aşama-1 gerekçelendirme:** Her sanal ajan detaylı analiz sağlar
- **Aşama-2 sentez:** Kapsamlı stratejik gerekçe (300+ kelime)
- **Güven skorları:** Her değerlendirme güven metrikleri içerir
- **Aksiyon planları:** Öncelik seviyeleriyle somut öneriler

**Denetim İzleme:**
- Tüm gerekçelendirme prompt bağlamında görünür
- "Kara kutu" kararlar yok
- Düzenleyici uyumluluk: Kararlar açıklanabilir ve denetlenebilir
- Kullanıcı güveni: Kullanıcılar belirli risk seviyelerinin neden atandığını anlar

### 6.5 Azınlık Odaklı Analiz

**Açık Azınlık Sınıfı Odaklanması:**
- **Pozisyonel önyargı:** Anomali verisi bağlamda önceliklendirilir
- **Tekrar ağırlıklandırması:** Kritik sinyaller 3x amplifiye edilir
- **Uyarılara yönelik önyargı:** Sistem ihtiyatlı tarafta hata yapar
- **Erken tespit:** Nadir ancak yüksek etkili olayları yakalamak için tasarlanmıştır

**Risk Felsefesi:**
- **Yanlış negatifler maliyetli:** Bir krizi kaçırmak yanlış alarmdan daha kötüdür
- **Azınlık sinyalleri korunur:** Sistem açıkça azınlık sınıfı bilgisini korur
- **Kriz-öncelikli yaklaşım:** Black Swan olaylarını tespit etmek için optimize edilmiştir

---

## 7. Kullanım Senaryoları ve Uygulamalar

### 7.1 Bireysel Hisse Senedi Risk Analizi

**Kullanım Senaryosu:** Belirli bir hisse senedi için risk seviyesini analiz etme (örn. TSLA, AAPL)

**Girdi:** "TSLA hisse risk analizi"

**Sistem Davranışı:**
- Son TSLA haberleri, anomaliler, piyasa verilerini toplar
- Risk, Makro, Sentiment, Teknik lens'ler aracılığıyla analiz eder
- Risk seviyesine sentezler (DÜŞÜK/ORTA/YÜKSEK/KRİTİK)
- Uygulanabilir öneriler sağlar

**Çıktı:**
- Güven skoru ile risk seviyesi
- Temel risk faktörleri (volatilite, haber duygusu, teknik göstergeler)
- Aksiyon planı (acil/yüksek/orta/düşük öncelikli aksiyonlar)

### 7.2 Piyasa Geneli Kriz Sinyali Tespiti

**Kullanım Senaryosu:** Piyasa geneli krizlerin erken uyarı sinyallerini tespit etme

**Girdi:** "ABD piyasası kriz sinyalleri"

**Sistem Davranışı:**
- Son dakika haberleri, sistemik risk göstergelerini arar
- Makroekonomik verileri analiz eder (GSYİH, enflasyon, işsizlik)
- Piyasa duygusu ve volatiliteyi değerlendirir
- Potansiyel kriz tetikleyicilerini belirler

**Çıktı:**
- Kriz olasılığı değerlendirmesi
- Temel sistemik risk faktörleri
- Önerilen savunma aksiyonları

### 7.3 Sektörel Risk Değerlendirmesi

**Kullanım Senaryosu:** Belirli bir sektör için risk değerlendirmesi (örn. teknoloji, enerji)

**Girdi:** "Teknoloji sektörü risk değerlendirmesi"

**Sistem Davranışı:**
- Sektöre özel haber ve verileri toplar
- Sektör trendlerini, düzenleyici değişiklikleri, piyasa koşullarını analiz eder
- Daha geniş piyasa göstergeleriyle karşılaştırır
- Sektöre özel zayıflıkları belirler

**Çıktı:**
- Piyasaya göre sektör risk seviyesi
- Sektöre özel risk faktörleri
- Sektör rotasyon önerileri

### 7.4 Portföy Risk Yönetimi

**Kullanım Senaryosu:** Genel portföy riskini değerlendirme ve ayarlama önerileri

**Girdi:** "Teknoloji ağırlıklı portföy için risk analizi"

**Sistem Davranışı:**
- Bireysel holding'leri analiz eder
- Portföy konsantrasyon riskini değerlendirir
- Holding'ler arası korelasyonu değerlendirir
- Çeşitlendirme fırsatlarını belirler

**Çıktı:**
- Portföy seviyesi risk değerlendirmesi
- Konsantrasyon riski uyarıları
- Çeşitlendirme önerileri
- Yeniden dengeleme önerileri

### 7.5 Erken Uyarı Sistemi

**Kullanım Senaryosu:** Erken kriz tespiti için sürekli izleme

**Girdi:** Zamanlanmış sorgular (örn. günlük piyasa taraması)

**Sistem Davranışı:**
- Otomatik günlük/haftalık risk taramaları
- Zaman içinde risk seviyelerindeki değişiklikleri takip eder
- Gelişmekte olan desenleri belirler
- Önemli risk artışlarında uyarı verir

**Çıktı:**
- Risk trend analizi
- Önemli değişiklikler için uyarı bildirimleri
- Tarihsel risk seviyesi takibi

---

## 8. Gelecek Çalışmalar ve Araştırma Yönleri

### 8.1 Model İyileştirmeleri

#### 8.1.1 Daha Büyük Model Entegrasyonu

**Potansiyel Modeller:**
- **Llama-3-70B:** Önemli ölçüde daha büyük kapasite, daha iyi akıl yürütme
- **Mixtral-8x7B:** Mixture-of-Experts, verimli çıkarım
- **GPT-4 sınıfı modeller:** API erişimi mevcut olursa

**Gereksinimler:**
- **VRAM:** 40GB+ (A100, H100) veya çoklu GPU kurulumu
- **Altyapı:** Üretim için bulut bilişim (AWS, GCP)
- **Maliyet:** Daha yüksek çıkarım maliyeti, ancak potansiyel olarak daha iyi doğruluk

**Faydalar:**
- Karmaşık finansal senaryolarda daha iyi akıl yürütme
- Daha nüanslı azınlık sınıfı tespiti
- İyileştirilmiş sentez kalitesi

#### 8.1.2 Fine-Tuning Entegrasyonu

**Yaklaşım:**
- **LoRA (Low-Rank Adaptation):** Minimal parametrelerle verimli fine-tuning
- **QLoRA:** VRAM verimliliği için 4-bit quantize edilmiş LoRA
- **Alan-spesifik fine-tuning:** Finansal kriz veri setleri üzerinde eğitim

**Hibrit Strateji:**
- ICL (mevcut yaklaşım) ile fine-tuned modelleri birleştirme
- Temel anlama için fine-tuned modeller, gerçek zamanlı adaptasyon için ICL kullanma
- A/B testi: Yalnızca ICL vs ICL+fine-tuning performansını karşılaştırma

**Araştırma Soruları:**
- Fine-tuning azınlık sınıfı tespitini iyileştirir mi?
- ICL ve fine-tuning arasındaki optimal denge nedir?
- Modeller ne sıklıkla yeniden eğitilmelidir?

#### 8.1.3 Çok Modlu Analiz

**Uzantılar:**
- **Grafik analizi:** Teknik grafik desenlerini dahil etme (mum grafiği, destek/direnç)
- **Haber görselleri:** Haber makalelerindeki grafik ve çizelgeleri analiz etme
- **Sosyal medya:** Twitter/Reddit duygusu ile görsel analizi entegre etme

**Teknik Gereksinimler:**
- Görü-dil modelleri (örn. GPT-4V, LLaVA)
- Görüntü ön işleme pipeline'ı
- Çok modlu bağlam sıkıştırma

### 8.2 Veri Kaynağı Genişletmeleri

#### 8.2.1 GDELT Entegrasyonu

**Global Database of Events, Language, and Tone:**
- **Kapsam:** Küresel olaylar, haberler, sosyal medya bahisleri
- **Gerçek zamanlı:** Her 15 dakikada bir güncelleme
- **Kullanım durumu:** Küresel kriz tespiti, jeopolitik risk

**Entegrasyon Zorlukları:**
- Büyük veri hacmi verimli filtreleme gerektirir
- API rate limit'leri ve maliyet düşünceleri
- Finansal olaylar için ilgili skorlama

#### 8.2.2 IMF/Dünya Bankası API'leri

**Amaç:** Yetkili makroekonomik veri

**Veri Türleri:**
- GSYİH tahminleri
- Enflasyon projeksiyonları
- Politika duyuruları
- Ülke risk değerlendirmeleri

**Faydalar:**
- Piyasa tabanlı göstergelerden daha güvenilir
- İleriye dönük veri (tahminler)
- Küresel kapsam

#### 8.2.3 Blockchain ve Kripto Verisi

**Amaç:** Kripto para piyasası analizi

**Veri Kaynakları:**
- On-chain metrikler (işlem hacmi, cüzdan aktivitesi)
- Borsa verisi (Binance, Coinbase API'leri)
- DeFi protokolleri (borç verme oranları, likidite havuzları)

**Kullanım Durumları:**
- Kripto piyasası risk değerlendirmesi
- Geleneksel piyasalarla korelasyon
- Gelişmekte olan varlık sınıfı analizi

#### 8.2.4 Sosyal Medya Duygusu

**Kaynaklar:**
- Twitter/X API (rate limit düşünceleriyle)
- Reddit (r/wallstreetbets, r/investing)
- StockTwits

**Zorluklar:**
- API erişim kısıtlamaları
- Gürültü filtreleme (bot tespiti)
- Gerçek zamanlı işleme gereksinimleri

**Potansiyel:**
- Erken duygu değişimi tespiti
- Perakende yatırımcı duygu takibi
- Viral olay etki değerlendirmesi

### 8.3 Metodoloji İyileştirmeleri

#### 8.3.1 Gerçek Fine-Tuning Karşılaştırma Çalışması

**Araştırma Hedefi:** ICL simülasyonu vs gerçek fine-tuning karşılaştırması

**Metodoloji:**
- Finansal kriz veri setleri üzerinde fine-tuned modeller eğitme
- Performans karşılaştırması: ICL vs fine-tuned vs hibrit
- Ölçüm: Doğruluk, azınlık sınıfı geri çağırma, çıkarım hızı, maliyet

**Beklenen Sonuçlar:**
- ICL etkinliğini fine-tuning'e göre nicelleştirme
- Fine-tuning'in gerekli olduğu senaryoları belirleme
- Hibrit stratejiler geliştirme

#### 8.3.2 A/B Test Çerçevesi

**Amaç:** Farklı yaklaşımların sistematik değerlendirmesi

**Test Değişkenleri:**
- Model seçimi (Mistral vs Llama vs Phi)
- Prompt mühendisliği stratejileri
- Bağlam sıkıştırma yöntemleri
- Ağırlıklandırma şemaları

**Metrikler:**
- Azınlık sınıfı tespit oranı
- Yanlış pozitif/negatif oranları
- Kullanıcı memnuniyeti
- Yanıt süresi

#### 8.3.3 Topluluk Yöntemleri

**Yaklaşım:** Geliştirilmiş doğruluk için çoklu modelleri birleştirme

**Stratejiler:**
- **Model çeşitliliği:** Farklı mimariler kullanma (Mistral, Llama, Phi)
- **Oylama mekanizmaları:** Çoğunluk oyu veya ağırlıklı ortalama
- **Güven tabanlı seçim:** En yüksek güven skorlu model çıktısını kullanma

**Zorluklar:**
- Artan hesaplama maliyeti
- VRAM yönetimi (çoklu modeller)
- Gecikme düşünceleri

### 8.4 Kaynak İhtiyaçları

#### 8.4.1 Donanım Yükseltmeleri

**Mevcut:** 16GB VRAM (T4, tüketici GPU)

**Gelecek Seçenekleri:**
- **A100 (40GB):** Daha büyük modeller için destek (Llama-70B)
- **H100 (80GB):** Yeni nesil modeller için gelecek garantisi
- **Çoklu GPU:** Paralel işleme, model parçalama

**Maliyet Düşünceleri:**
- Bulut bilişim: Kullandıkça öde (AWS, GCP, Azure)
- Şirket içi: Daha yüksek başlangıç maliyeti, daha düşük uzun vadeli maliyet
- Hibrit: Bulutta geliştirme, şirket içi üretim

#### 8.4.2 Premium API Abonelikleri

**Mevcut:** Rate limit'li ücretsiz tier API'ler

**Yükseltme Seçenekleri:**
- **Alpha Vantage Premium:** Daha yüksek rate limit'ler, daha fazla veri
- **GDELT Premium:** Tam veritabanı erişimi
- **Haber API'leri:** Premium haber kaynakları (Bloomberg, Reuters)

**Faydalar:**
- Daha güvenilir veri erişimi
- Daha yüksek veri kalitesi
- Azaltılmış rate limiting sorunları

#### 8.4.3 Veritabanı Altyapısı

**Amaç:** Tarihsel veri önbellekleme ve analiz

**Bileşenler:**
- **Zaman serisi veritabanı:** Piyasa verisi için InfluxDB, TimescaleDB
- **Belge deposu:** Haber makaleleri ve raporlar için MongoDB
- **Vektör veritabanı:** Semantik arama için Pinecone, Weaviate

**Kullanım Durumları:**
- Tarihsel risk seviyesi takibi
- Zaman içinde desen tanıma
- Performans geri testi

#### 8.4.4 Bulut Bilişim Dağıtımı

**Üretim Gereksinimleri:**
- **Ölçeklenebilirlik:** Çoklu eşzamanlı sorguları işleme
- **Güvenilirlik:** %99.9 çalışma süresi
- **İzleme:** Loglama, uyarı, performans takibi
- **Güvenlik:** API anahtarı yönetimi, veri şifreleme

**Platform Seçenekleri:**
- **AWS:** EC2 (GPU örnekleri), SageMaker
- **GCP:** Compute Engine, Vertex AI
- **Azure:** Virtual Machines, Azure ML

**Mimari:**
- Load balancer → API gateway → Worker node'ları (GPU)
- Talebe göre otomatik ölçeklendirme
- Konteynerleştirme (Docker, Kubernetes)

### 8.5 Araştırma Soruları

1. **ICL vs Fine-Tuning:** ICL hangi koşullarda fine-tuning performansına eşit veya onu aşar?

2. **Azınlık Sınıfı Tespiti:** Finansal risk analizinde yanlış pozitifler ve yanlış negatifler arasındaki optimal denge nedir?

3. **Gerçek Zamanlı Adaptasyon:** Sistem, fine-tuned modellere kıyasla sıfır günlük olaylara ne kadar hızlı adapte olabilir?

4. **Maliyet-Etkinlik:** ICL ve fine-tuning yaklaşımları arasındaki toplam sahip olma maliyeti (TCO) karşılaştırması nedir?

5. **Açıklanabilirlik:** ICL açıklanabilirliği fine-tuned model yorumlanabilirliğiyle nasıl karşılaştırılır?

---

## 9. Referanslar

### 9.1 Birincil Referans

**ValizadehAslani, T., et al. (2022).** "Two-Stage Fine-Tuning: A Novel Strategy for Learning Class-Imbalanced Data." *[Journal/Conference]*.  
*Bu makale, sınıf dengesizliği olan veri setlerini ele almak için iki aşamalı fine-tuning metodolojisini tanıtır ve ICL tabanlı simülasyon yaklaşımımızın teorik temelini oluşturur.*

### 9.2 In-Context Learning Literatürü

**Brown, T., et al. (2020).** "Language Models are Few-Shot Learners." *Advances in Neural Information Processing Systems (NeurIPS)*, 33, 1877-1901.  
*Büyük dil modellerinin in-context learning yeteneklerini tanıtan temel makale.*

**Dong, Q., et al. (2022).** "A Survey on In-Context Learning." *arXiv preprint arXiv:2301.00234*.  
*In-context learning yöntemleri ve uygulamalarının kapsamlı araştırması.*

**Liu, P., et al. (2021).** "Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in Natural Language Processing." *arXiv preprint arXiv:2107.13586*.  
*Prompt mühendisliği teknikleri ve uygulamalarının araştırması.*

### 9.3 Sınıf Dengesizliği Öğrenme

**He, H., & Garcia, E. A. (2009).** "Learning from Imbalanced Data." *IEEE Transactions on Knowledge and Data Engineering*, 21(9), 1263-1284.  
*Sınıf dengesizliği öğrenme teknikleri üzerine temel araştırma.*

**Chawla, N. V., et al. (2002).** "SMOTE: Synthetic Minority Over-sampling Technique." *Journal of Artificial Intelligence Research*, 16, 321-357.  
*Dengesiz veri setlerini ele almak için SMOTE algoritmasının tanıtımı.*

**Johnson, J. M., & Khoshgoftaar, T. M. (2019).** "Survey on Deep Learning with Class Imbalance." *Journal of Big Data*, 6(1), 1-54.  
*Sınıf dengesizliği ile derin öğrenme yaklaşımlarının araştırması.*

### 9.4 Finansal NLP ve Alan Modelleri

**Araci, D. (2019).** "FinBERT: Financial Sentiment Analysis with Pre-trained Language Models." *arXiv preprint arXiv:1908.10063*.  
*Bağlam sıkıştırma için sistemimizde kullanılan finansal alan-spesifik BERT modeli FinBERT'in tanıtımı.*

**Yang, L., et al. (2020).** "Financial Sentiment Analysis: Techniques and Applications." *ACM Computing Surveys*, 53(4), 1-36.  
*Finansal duygu analizi yöntemlerinin araştırması.*

### 9.5 Bayesci Toplama ve Topluluk Yöntemleri

**Dietterich, T. G. (2000).** "Ensemble Methods in Machine Learning." *International Workshop on Multiple Classifier Systems*, 1-15.  
*Topluluk öğrenme yöntemleri üzerine temel çalışma.*

**Kuncheva, L. I. (2004).** "Combining Pattern Classifiers: Methods and Algorithms." *John Wiley & Sons*.  
*Bayesci yaklaşımlar dahil olmak üzere sınıflandırıcı birleştirme yöntemlerinin kapsamlı rehberi.*

### 9.6 Model Quantization ve Verimlilik

**Dettmers, T., et al. (2022).** "QLoRA: Efficient Finetuning of Quantized LLMs." *arXiv preprint arXiv:2305.14314*.  
*Quantize edilmiş modellerin verimli fine-tuning'i için QLoRA yönteminin tanıtımı.*

**Frantar, E., et al. (2022).** "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers." *arXiv preprint arXiv:2210.17323*.  
*Büyük dil modelleri için eğitim sonrası quantization yöntemi.*



## 📝 Lisans

Bu proje akademik araştırma amaçlıdır. Bu proje [GNU GENEL KAMU LİSANSI](LICENSE) altında lisanslanmıştır.

---
## **TR**: Proje geliştirme ve işbirliği için:
- E-posta: [eyup.tp@hotmail.com](mailto:eyup.tp@hotmail.com)

---

## 📞 Destek

Sorularınız için:
- Proje dokümantasyonunu inceleyin
- `COLAB_SETUP.md` dosyasına bakın
- GitHub Issues kullanın 

---