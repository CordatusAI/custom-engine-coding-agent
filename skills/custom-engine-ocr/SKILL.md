---
name: custom-engine-ocr
description: OCR (Optical Character Recognition) processor for Custom Engine. Use when adding text reading, license plate recognition, or document scanning to a Custom Engine pipeline.
owner: OpenZeka
service: custom-engine
version: 1.0.0
---

# Custom Engine OCR Skill

When this skill is active, **read the relevant reference documents** before generating code.

## Overview

Creates an `OCRReader` processor that runs OCR on each frame or on specific ROIs. Supports Tesseract and EasyOCR backends.

## Pipeline Position

OCR should come **after detection processors** if reading from detected regions (e.g., license plates):

```
Input → [YOLODetector] → OCRReader → [DrawAnnotations] → Output
```

If running OCR on the full frame (no detection), it can be the first processor.

## Critical Rules

1. **Always extend BaseProcessor**: Create `custom_engine/ocr_reader.py` extending `BaseProcessor`
2. **Backend selection**: EasyOCR supports GPU, Tesseract is CPU-only. Prefer EasyOCR when GPU is available
3. **ROI source**: If `roi_source` is set, read detections from metadata to crop regions before OCR
4. **Metadata format**: Always output:
   - `ocr_text`: concatenated string of all detected text
   - `ocr_regions`: list of dicts with `text`, `bbox`, `confidence`
5. **Language config**: Default `"eng"`, configurable for multilingual support
6. **BGR convention**: Convert to RGB before passing to OCR engines
7. **Confidence filtering**: Filter results below minimum confidence threshold

## Processor Template

```python
from base_processor import BaseProcessor


class OCRReader(BaseProcessor):
    def __init__(self, engine="easyocr", language="en", confidence=0.5,
                 roi_source=None, **kwargs):
        super().__init__(engine=engine, language=language,
                         confidence=confidence, roi_source=roi_source, **kwargs)
        self.engine_name = engine
        self.language = language
        self.confidence = confidence
        self.roi_source = roi_source
        self.reader = self._init_reader()

    def _init_reader(self):
        if self.engine_name == "easyocr":
            import easyocr
            gpu = self._check_gpu()
            return easyocr.Reader([self.language], gpu=gpu)
        else:
            import pytesseract
            return pytesseract

    def _check_gpu(self):
        try:
            import onnxruntime as ort
            return "CUDAExecutionProvider" in ort.get_available_providers()
        except ImportError:
            return False

    def process(self, frame, metadata=None):
        if metadata is None:
            metadata = {}
        import cv2
        regions = []
        if self.roi_source and self.roi_source in metadata:
            rois = metadata[self.roi_source]
            for det in rois:
                x1, y1, x2, y2 = det["bbox"]
                roi = frame[y1:y2, x1:x2]
                text, conf = self._read_text(roi)
                if conf >= self.confidence:
                    regions.append({"text": text, "bbox": det["bbox"], "confidence": conf})
        else:
            results = self._read_full(frame)
            for text, bbox, conf in results:
                if conf >= self.confidence:
                    regions.append({"text": text, "bbox": bbox, "confidence": conf})
        metadata["ocr_text"] = " ".join(r["text"] for r in regions)
        metadata["ocr_regions"] = regions
        return frame, metadata

    def _read_text(self, image):
        if self.engine_name == "easyocr":
            results = self.reader.readtext(image)
            if results:
                best = max(results, key=lambda x: x[2])
                return best[1], best[2]
            return "", 0.0
        else:
            import cv2
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            text = self.reader.image_to_string(rgb)
            return text.strip(), 1.0

    def _read_full(self, frame):
        if self.engine_name == "easyocr":
            import cv2
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return self.reader.readtext(rgb)
        else:
            import cv2
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            text = self.reader.image_to_string(rgb)
            return [(text, [0, 0, 0, 0], 1.0)] if text.strip() else []
```

## Reference Documents

| Document | Use When |
|----------|----------|
| [references/ocr_engines.md](references/ocr_engines.md) | Tesseract/EasyOCR setup, language packs, GPU support |
| [references/text_processing.md](references/text_processing.md) | OCR result processing, confidence filtering, metadata format |
