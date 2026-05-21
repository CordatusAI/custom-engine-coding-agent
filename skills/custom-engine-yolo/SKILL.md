---
name: custom-engine-yolo
description: YOLO object detection processor for Custom Engine. Use when adding YOLOv8/v11/v10/v26 object detection, bounding box drawing, or object counting to a Custom Engine pipeline.
owner: OpenZeka
service: custom-engine
version: 1.0.0
---

# Custom Engine YOLO Detection Skill

When this skill is active, **read the relevant reference documents** before generating code. Do NOT rely on memory.

## Overview

Creates a `YOLODetector` processor that runs YOLO object detection on each frame. Supports Ultralytics YOLO family models with ONNX Runtime inference (GPU-first, CPU-fallback).

## Pipeline Position

YOLO detection should be one of the **first processors** in the pipeline, as other processors (tracking, OCR, annotation) may depend on its metadata.

```
Input → YOLODetector → [Tracker] → [OCR] → [DrawAnnotations] → Output
```

## Critical Rules

1. **Always extend BaseProcessor**: Create `custom_engine/yolo_detector.py` with a class extending `BaseProcessor`
2. **GPU-first inference**: Use `onnxruntime` with `CUDAExecutionProvider` if available, fall back to `CPUExecutionProvider`
3. **Model auto-download**: If the ONNX model file does not exist, download it automatically (Ultralytics hub or direct URL)
4. **Metadata format**: Always output:
   - `yolo_detections`: list of dicts with `bbox`, `class`, `confidence`
   - `yolo_object_count`: integer count of detections
5. **Confidence threshold**: Default 0.5, configurable via constructor
6. **Input size**: Default 640x640, configurable. Must match model's training input size
7. **BGR frame convention**: Input is BGR, convert to RGB for inference, convert back to BGR for output
8. **Batch size 1**: Real-time inference processes one frame at a time
9. **Frame copy for drawing**: If drawing annotations on the frame, copy it first — never mutate the input array

## Processor Template

```python
from base_processor import BaseProcessor
import numpy as np
import os


class YOLODetector(BaseProcessor):
    def __init__(self, model="yolov8s", confidence=0.5, input_size=640, **kwargs):
        super().__init__(model=model, confidence=confidence, input_size=input_size, **kwargs)
        self.model_name = model
        self.confidence = confidence
        self.input_size = input_size
        self.session = self._load_model()

    def _load_model(self):
        model_path = self._get_or_download_model()
        import onnxruntime as ort
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        session = ort.InferenceSession(model_path, providers=providers)
        return session

    def _get_or_download_model(self):
        model_dir = os.path.join(os.path.dirname(__file__), "models")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{self.model_name}.onnx")
        if not os.path.exists(model_path):
            self._download_model(model_path)
        return model_path

    def _download_model(self, path):
        from ultralytics import YOLO
        model = YOLO(f"{self.model_name}.pt")
        model.export(format="onnx", imgsz=self.input_size)
        import shutil
        exported = f"{self.model_name}.onnx"
        shutil.move(exported, path)

    def _preprocess(self, frame):
        import cv2
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_size, self.input_size))
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)
        return img

    def _postprocess(self, outputs, original_shape):
        pass

    def process(self, frame, metadata=None):
        if metadata is None:
            metadata = {}
        original_shape = frame.shape
        input_tensor = self._preprocess(frame)
        outputs = self.session.run(None, {self.session.get_inputs()[0].name: input_tensor})
        detections = self._postprocess(outputs, original_shape)
        metadata["yolo_detections"] = detections
        metadata["yolo_object_count"] = len(detections)
        return frame, metadata
```

## Reference Documents

| Document | Use When |
|----------|----------|
| [references/yolo_models.md](references/yolo_models.md) | Supported models, download, ONNX export |
| [references/inference_config.md](references/inference_config.md) | GPU/CPU selection, ONNX Runtime config, post-processing |
| [references/drawing_annotations.md](references/drawing_annotations.md) | Bbox, label, count drawing, OpenCV annotations |
