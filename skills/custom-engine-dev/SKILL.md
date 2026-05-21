---
name: custom-engine-dev
description: Custom Engine processor pipeline development with BaseProcessor interface. Use when building video processing modules, YOLO inference, OCR, object tracking, or any frame processing for Custom Engine.
owner: OpenZeka
service: custom-engine
version: 1.0.0
---

# Custom Engine Development Skill

When this skill is active, **ALWAYS read the relevant reference documents** before generating code. Do NOT rely on memory — the reference documents contain critical details about exact API usage, metadata formats, and common pitfalls.

## Architecture Overview

Custom Engine uses a **processor pipeline** pattern. Each processing module implements the `BaseProcessor` interface and is added to `CustomEngine.processors` via `add_processor()`.

### Pipeline Flow

```
Frame → Processor 1 → Processor 2 → ... → Processor N → Annotated Frame
                                            ↓
                                      message_buffer (optional metadata)
```

### Key Components

| Component | Role | File |
|-----------|------|------|
| CustomEngine | Runs processor pipeline, collects metadata | `custom_engine/custom_engine.py` |
| BaseProcessor | Abstract interface for all processors | `custom_engine/base_processor.py` |
| [Processor] | User-created modules (YOLO, OCR, etc.) | `custom_engine/[name].py` |

## Critical Rules

1. **Always use BaseProcessor**: Never write logic directly inside `__call__`. Create a new processor class that extends `BaseProcessor`.
2. **Write to custom_engine/ only**: All processor modules go in `custom_engine/`. This is the only directory volume-mounted in Docker.
3. **Do not modify CustomEngine internals**: `CustomEngine`, `__init__`, `__call__`, and `set_data` are fixed. Only add processors via `add_processor()`.
4. **Frame format is BGR**: OpenCV convention — frames arrive as BGR NumPy arrays. Processors must return BGR frames.
5. **Metadata is a dict**: Each processor adds its own keys to the shared metadata dict. Never overwrite another processor's keys unless intentionally merging.
6. **GPU-first, CPU-fallback**: If a GPU is available, use ONNX Runtime with CUDAExecutionProvider or TensorRT. Fall back to CPUExecutionProvider gracefully.
7. **One processor, one responsibility**: A processor should do one thing well. Chain multiple processors for complex workflows.
8. **Processor naming**: File name = snake_case, class name = PascalCase. Example: `yolo_detector.py` → `YOLODetector`.
9. **is_active is True by default**: Do not set `is_active = False` unless explicitly asked.
10. **message_buffer is optional**: Only put metadata if `self.message_buffer` is not None and metadata is non-empty.

## Processor Template

When creating a new processor, follow this template:

```python
from base_processor import BaseProcessor


class MyProcessor(BaseProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def process(self, frame, metadata=None):
        if metadata is None:
            metadata = {}
        metadata["my_key"] = "my_value"
        return frame, metadata
```

## Adding Processors to CustomEngine

```python
from custom_engine import CustomEngine
from yolo_detector import YOLODetector
from ocr_reader import OCRReader

engine = CustomEngine(logger=logger, camera_id="cam_01", message_buffer=msg_queue)
engine.add_processor(YOLODetector(model="yolov8s", confidence=0.5))
engine.add_processor(OCRReader(language="eng"))
```

## Key Paths

- Custom Engine modules: `custom_engine/`
- Base processor: `custom_engine/base_processor.py`
- Main engine: `custom_engine/custom_engine.py`

## Reference Documents

**IMPORTANT**: Always read these documents for complete details. Do NOT generate code from memory.

| Document | Use When |
|----------|----------|
| [references/base_processor_api.md](references/base_processor_api.md) | BaseProcessor interface, process() contract, metadata format |
| [references/custom_engine_api.md](references/custom_engine_api.md) | CustomEngine class details, add_processor, __call__ flow |
| [references/pipeline_patterns.md](references/pipeline_patterns.md) | Processor chaining examples, ordering rules, common patterns |
| [references/troubleshooting.md](references/troubleshooting.md) | Import errors, GPU/CPU fallback, frame format issues |

## Quick Error Reference

| Error | Solution |
|-------|----------|
| `TypeError: must be a BaseProcessor instance` | Ensure processor extends `BaseProcessor` and is instantiated before passing to `add_processor()` |
| `ModuleNotFoundError` for processor | Check import path — processors are in `custom_engine/`, use relative or sys.path imports |
| Wrong colors on frame | Frame is BGR, not RGB. Convert if needed: `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)` |
| `onnxruntime` not found | Install `onnxruntime-gpu` or `onnxruntime` inside the Docker container |
| Metadata keys overwritten | Use unique keys per processor (e.g., `yolo_detections`, not `detections`) |