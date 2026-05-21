# Custom Engine Coding Agent — Design Spec

## Overview

Custom Engine Coding Agent, NVIDIA DeepStream Coding Agent yapısını referans alarak, Stream Engine üzerindeki Custom Engine yapısı için AI coding assistant'larına yönelik skill seti ve agent yapısı sağlar. Kullanıcı, skiller aracılığıyla agent'a "custom engine üzerinde YOLOv8s çalıştır, obje sayısını frame üzerine bas" gibi talimatlar verdiğinde, agent nereye ne yazacağını bilerek doğrudan çalışan kod üretir.

## Constraints

- **Docker volume**: Kullanıcının erişebileceği tek klasör `custom_engine/`'dir. Tüm üretilecek modüller bu klasör içine yazılır.
- **Sabit yapı**: `CustomEngine` sınıf adı, `custom_engine.py` dosya adı, `__call__` ve `set_data` fonksiyonları değiştirilemez.
- **Processor Pipeline**: Yeni özellikler `BaseProcessor` interface'ini uygulayan ayrı `.py` dosyaları olarak eklenir, `__call__` içinden pipeline olarak çalıştırılır.
- **GPU/CPU**: Maksimum performans hedeflenir. GPU varsa kullanılır (ONNX Runtime GPU / TensorRT), yoksa CPU'ya fallback edilir.
- **Skill mekanizması**: DeepStream Coding Agent ile aynı yapı — SKILL.md + references/ + .claude-plugin/. Kullanıcı skilleri opencode/Claude Code gibi araçlara yükler.
- **custom_engine yolu**: Varsayılan olarak `./custom_engine/` (workspace root'a göre). İleride değişebilir.

## Architecture: Processor Pipeline

Her modül `BaseProcessor` abstract sınıfını uygular. `CustomEngine.__call__` processors listesini sırayla çalıştırır, metadata'yı biriktirir ve isteğe bağlı `message_buffer`'a gönderir.

### base_processor.py

```python
from abc import ABC, abstractmethod

class BaseProcessor(ABC):
    def __init__(self, **kwargs):
        self.config = kwargs
    
    @abstractmethod
    def process(self, frame, metadata=None):
        if metadata is None:
            metadata = {}
        pass
    
    @property
    def name(self):
        return self.__class__.__name__
```

**Kontrat:**
- `process(frame, metadata)` → `(frame, metadata)` döndürür
- Frame BGR formatında gelir ve BGR olarak döndürülür
- Metadata dict olarak birikir, her processor kendi alanını ekler/günceller
- GPU kullanımı opsiyonel, processor kendi içinde handle eder

### custom_engine.py (güncellenmiş)

```python
from queue import Queue
from logging import Logger
from .base_processor import BaseProcessor

class CustomEngine:
    def __init__(self, logger, camera_id, message_buffer) -> None:
        self.camera_id = camera_id
        self.logger = logger
        self.is_active = True
        self.message_buffer = message_buffer
        self.processors = []
    
    def add_processor(self, processor):
        if isinstance(processor, BaseProcessor):
            self.processors.append(processor)
        else:
            raise TypeError(f"{processor} must be BaseProcessor instance")
    
    def __call__(self, iframe):
        frame = iframe.copy()
        metadata = {}
        for proc in self.processors:
            frame, metadata = proc.process(frame, metadata)
        if self.message_buffer and metadata:
            self.message_buffer.put(metadata)
        return frame
    
    def set_data(self, **kwargs):
        self.logger.info(f"KWARGS     ------->  {kwargs}")
        data = kwargs
```

**Değişiklikler (orijinale göre):**
- `is_active` varsayılan olarak `True` (orijinalde `False` idi)
- `processors` listesi eklendi
- `add_processor()` metodu eklendi
- `__call__` pipeline olarak çalışır, orijinaldeki BGR↔RGB channel swap kodu kaldırıldı. Bu işlem gerekiyorsa `ChannelSwapProcessor` gibi bir processor olarak eklenebilir.
- Metadata biriktirme ve `message_buffer` gönderme mantığı eklendi

## Project Structure

```
custom-engine-coding-agent/
├── custom_engine/                          # Docker volume
│   ├── custom_engine.py                    # CustomEngine + pipeline
│   └── base_processor.py                   # BaseProcessor abstract sınıfı
├── skills/
│   ├── custom-engine-dev/                  # Ana skill: pipeline kuralları, API referansları
│   │   ├── SKILL.md
│   │   ├── .claude-plugin/
│   │   │   └── plugin.json
│   │   ├── evals/
│   │   │   └── evals.json
│   │   └── references/
│   │       ├── base_processor_api.md
│   │       ├── custom_engine_api.md
│   │       ├── pipeline_patterns.md
│   │       └── troubleshooting.md
│   ├── custom-engine-yolo/                 # YOLO Detection skill
│   │   ├── SKILL.md
│   │   └── references/
│   │       ├── yolo_models.md
│   │       ├── inference_config.md
│   │       └── drawing_annotations.md
│   ├── custom-engine-ocr/                  # OCR skill
│   │   ├── SKILL.md
│   │   └── references/
│   │       ├── ocr_engines.md
│   │       └── text_processing.md
│   └── custom-engine-tracking/            # Object Tracking skill
│       ├── SKILL.md
│       └── references/
│           ├── tracker_types.md
│           └── tracking_patterns.md
├── example_prompts/
│   ├── yolo_object_count.md
│   ├── ocr_text_detection.md
│   └── yolo_with_tracking.md
└── README.md
```

## Skills

### custom-engine-dev (Ana Skill)

Tüm skillerin temel kurallarını ve API referanslarını içerir. Agent bu skill sayesinde CustomEngine yapısını, BaseProcessor interface'ini ve pipeline pattern'lerini bilir.

**SKILL.md kuralları:**
- Her zaman `BaseProcessor` interface'ini kullan, doğrudan `__call__` içine logic yazma
- Yeni processor'lar `custom_engine/` içine ayrı `.py` dosyası olarak yazılır
- `CustomEngine`, `__init__`, `__call__`, `set_data` sabit kalır
- Processor'lar `add_processor()` ile eklenir
- Metadata dict formatında birikir, `message_buffer` üzerinden isteğe bağlı gönderilir
- GPU varsa ONNX Runtime GPU / TensorRT kullan, yoksa CPU fallback
- Frame BGR formatında gelir, processor'lar BGR'de çalışmalı

**references/:**

| Dosya | İçerik |
|-------|--------|
| base_processor_api.md | BaseProcessor interface, process() kontratı, metadata formatı |
| custom_engine_api.md | CustomEngine sınıf detayları, add_processor, __call__ akışı |
| pipeline_patterns.md | Processor zincirleme örnekleri, sıralama kuralları, yaygın patternler |
| troubleshooting.md | Import hataları, GPU/CPU fallback, frame format sorunları |

**evals/:** DeepStream referanslı evals.json formatında test caseleri.

### custom-engine-yolo

YOLO tabanlı object detection için skill. Ultralytics ailesi modeller desteklenir.

**SKILL.md kuralları:**
- Ultralytics tabanlı modeller (YOLOv8/v11/v10/v26)
- Model dosyası yoksa processor içinde otomatik indirme mantığı
- GPU: `onnxruntime-gpu` ile çalıştır, CPU: `onnxruntime` fallback
- Tespit sonuçları metadata'ya `{"detections": [...], "object_count": N}` olarak eklenir
- Opsiyonel: bbox/label frame üzerine çizilir

**references/:**

| Dosya | İçerik |
|-------|--------|
| yolo_models.md | Desteklenen modeller, indirme, ONNX export |
| inference_config.md | GPU/CPU seçimi, ONNX Runtime konfigürasyonu, performans |
| drawing_annotations.md | Bbox, label, count çizimi, OpenCV annotasyonları |

### custom-engine-ocr

OCR (Optical Character Recognition) için skill. Tesseract ve EasyOCR desteği.

**SKILL.md kuralları:**
- Tesseract ve EasyOCR desteği
- ROI bazlı veya tam frame OCR
- Sonuçlar metadata'ya `{"ocr_text": "...", "ocr_regions": [...]}` olarak eklenir
- Dil konfigürasyonu parametrik

**references/:**

| Dosya | İçerik |
|-------|--------|
| ocr_engines.md | Tesseract/EasyOCR kurulumu, dil paketleri, GPU desteği |
| text_processing.md | OCR sonucu işleme, confidence filtering, metadata formatı |

### custom-engine-tracking

Object Tracking için skill. YOLO sonrası tracking zinciri.

**SKILL.md kuralları:**
- YOLO sonrası tracking zinciri
- IOU, ByteTrack, SORT desteği
- Track ID'ler metadata'ya `{"tracks": [...], "track_count": N}` olarak eklenir
- YOLO + Tracking birlikte çalışacak şekilde zincirlenir

**references/:**

| Dosya | İçerik |
|-------|--------|
| tracker_types.md | Tracker çeşitleri, konfigürasyon, performans karşılaştırması |
| tracking_patterns.md | YOLO → Tracking zinciri, ID atama, geçiş patternleri |

## Example Prompts

| Dosya | Senaryo |
|-------|---------|
| yolo_object_count.md | Custom engine üzerinde YOLOv8s çalıştır, obje sayısını frame üzerine bas |
| ocr_text_detection.md | Frame üzerinde OCR çalıştır, tespit edilen metinleri çiz |
| yolo_with_tracking.md | YOLO tespit + tracking zinciri kur, her objeye ID ata |

## Metadata Format

Her processor kendi alanını metadata dict'ine ekler. Ortak alanlar:

```python
{
    "camera_id": "cam_01",
    "timestamp": 1715779200,
    "detections": [          # YOLO skill
        {"bbox": [x1,y1,x2,y2], "class": "person", "confidence": 0.92, "track_id": 3}
    ],
    "object_count": 5,      # YOLO skill
    "ocr_text": "...",       # OCR skill
    "ocr_regions": [...],   # OCR skill
    "tracks": [...],        # Tracking skill
    "track_count": 3         # Tracking skill
}
```

## Future Considerations

- **Custom Drawing skill**: Frame üzerine text, rectangle, circle, polygon çizimi (şimdilik basit, ileride ayrı skill)
- **custom_engine yolu konfigürasyonu**: Farklı volume mount lokasyonları için path konfigürasyonu
- **TensorRT entegrasyonu**: ONNX Runtime yanı sıra doğrudan TensorRT engine çalıştırma
