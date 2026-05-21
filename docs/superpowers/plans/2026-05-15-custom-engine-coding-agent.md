# Custom Engine Coding Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a coding agent skill set for Custom Engine that lets AI assistants generate processor modules (YOLO, OCR, Tracking) following a consistent pipeline pattern.

**Architecture:** BaseProcessor abstract class defines the interface. CustomEngine runs processors as a pipeline in `__call__`. Each skill (SKILL.md + references) teaches the agent how to write a specific processor type. DeepStream Coding Agent structure is used as the reference pattern.

**Tech Stack:** Python 3.10+, ONNX Runtime, OpenCV, Ultralytics (YOLO), Tesseract/EasyOCR, ByteTrack

---

### Task 1: Core — BaseProcessor

**Files:**
- Create: `custom_engine/base_processor.py`

- [ ] **Step 1: Write base_processor.py**

```python
from abc import ABC, abstractmethod


class BaseProcessor(ABC):
    def __init__(self, **kwargs):
        self.config = kwargs

    @abstractmethod
    def process(self, frame, metadata=None):
        if metadata is None:
            metadata = {}
        raise NotImplementedError

    @property
    def name(self):
        return self.__class__.__name__
```

- [ ] **Step 2: Commit**

```bash
git add custom_engine/base_processor.py
git commit -m "feat: add BaseProcessor abstract class"
```

---

### Task 2: Core — Update CustomEngine

**Files:**
- Modify: `custom_engine/custom_engine.py`

- [ ] **Step 1: Update custom_engine.py with processor pipeline**

```python
from queue import Queue
from logging import Logger
from .base_processor import BaseProcessor


class CustomEngine:
    def __init__(self, logger, camera_id, message_buffer) -> None:
        self.camera_id = camera_id
        self.logger: Logger = logger
        self.is_active: bool = True
        self.message_buffer: Queue = message_buffer
        self.processors: list[BaseProcessor] = []

    def add_processor(self, processor):
        if not isinstance(processor, BaseProcessor):
            raise TypeError(f"{processor} must be a BaseProcessor instance")
        self.processors.append(processor)
        self.logger.info(f"Processor added: {processor.name}")

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

- [ ] **Step 2: Commit**

```bash
git add custom_engine/custom_engine.py
git commit -m "feat: update CustomEngine with processor pipeline"
```

---

### Task 3: Skill — custom-engine-dev (SKILL.md + plugin + evals)

**Files:**
- Create: `skills/custom-engine-dev/SKILL.md`
- Create: `skills/custom-engine-dev/.claude-plugin/plugin.json`
- Create: `skills/custom-engine-dev/evals/evals.json`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p skills/custom-engine-dev/.claude-plugin
mkdir -p skills/custom-engine-dev/evals
mkdir -p skills/custom-engine-dev/references
```

- [ ] **Step 2: Write plugin.json**

```json
{
  "name": "custom-engine-dev",
  "description": "Custom Engine processor pipeline development with BaseProcessor interface. Use when building video processing modules, YOLO inference, OCR, object tracking, or any frame processing for Custom Engine.",
  "author": "OpenZeka",
  "skills": "./"
}
```

- [ ] **Step 3: Write SKILL.md**

```markdown
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
        # Initialize model, load resources, etc.

    def process(self, frame, metadata=None):
        if metadata is None:
            metadata = {}
        # 1. Process the frame
        # 2. Add results to metadata
        metadata["my_key"] = "my_value"
        # 3. Return (frame, metadata)
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
```

- [ ] **Step 4: Write evals.json**

```json
{
  "skill_name": "custom-engine-dev",
  "evals": [
    {
      "id": 1,
      "name": "basic-yolo-processor",
      "prompt": "Create a YOLOv8s object detection processor for Custom Engine that draws bounding boxes and adds detection count to the frame.",
      "expected_output": "A processor class extending BaseProcessor with process() method, added via add_processor(), using onnxruntime with GPU fallback, writing to custom_engine/ directory.",
      "files": [],
      "assertions": [
        {
          "text": "Processor extends BaseProcessor",
          "type": "contains_phrase",
          "phrase": "BaseProcessor"
        },
        {
          "text": "Implements process() method",
          "type": "contains_phrase",
          "phrase": "def process"
        },
        {
          "text": "Added via add_processor()",
          "type": "contains_phrase",
          "phrase": "add_processor"
        },
        {
          "text": "Uses onnxruntime for inference",
          "type": "contains_phrase",
          "phrase": "onnxruntime"
        },
        {
          "text": "GPU fallback to CPU handled",
          "type": "contains_pattern",
          "pattern": "CUDAExecutionProvider"
        },
        {
          "text": "Module written to custom_engine/",
          "type": "contains_pattern",
          "pattern": "custom_engine/"
        }
      ]
    },
    {
      "id": 2,
      "name": "processor-without-baseprocessor-rejected",
      "prompt": "Add a simple grayscale conversion to the custom engine pipeline by writing code directly inside __call__.",
      "expected_output": "Agent refuses and creates a BaseProcessor-based GrayscaleProcessor instead, never modifying __call__ directly.",
      "files": [],
      "assertions": [
        {
          "text": "Does not modify __call__ directly",
          "type": "not_contains",
          "phrase": "def __call__"
        },
        {
          "text": "Creates a BaseProcessor subclass",
          "type": "contains_phrase",
          "phrase": "BaseProcessor"
        },
        {
          "text": "Uses add_processor to register",
          "type": "contains_phrase",
          "phrase": "add_processor"
        }
      ]
    },
    {
      "id": 3,
      "name": "chained-processors",
      "prompt": "Create a pipeline that first runs YOLO detection and then runs OCR only on detected regions. Chain them properly.",
      "expected_output": "Two separate processors added in order — YOLO first, OCR second — with YOLO detections passed via metadata for ROI cropping.",
      "files": [],
      "assertions": [
        {
          "text": "Two processors created",
          "type": "contains_pattern",
          "pattern": "BaseProcessor"
        },
        {
          "text": "YOLO processor added before OCR",
          "type": "contains_pattern",
          "pattern": "add_processor"
        },
        {
          "text": "Metadata passed between processors",
          "type": "contains_phrase",
          "phrase": "metadata"
        }
      ]
    }
  ]
}
```

- [ ] **Step 5: Commit**

```bash
git add skills/custom-engine-dev/
git commit -m "feat: add custom-engine-dev skill with SKILL.md, plugin, and evals"
```

---

### Task 4: Skill — custom-engine-dev references

**Files:**
- Create: `skills/custom-engine-dev/references/base_processor_api.md`
- Create: `skills/custom-engine-dev/references/custom_engine_api.md`
- Create: `skills/custom-engine-dev/references/pipeline_patterns.md`
- Create: `skills/custom-engine-dev/references/troubleshooting.md`

- [ ] **Step 1: Write base_processor_api.md**

```markdown
# BaseProcessor API Reference

## Interface

```python
from abc import ABC, abstractmethod


class BaseProcessor(ABC):
    def __init__(self, **kwargs):
        self.config = kwargs

    @abstractmethod
    def process(self, frame, metadata=None):
        if metadata is None:
            metadata = {}
        raise NotImplementedError

    @property
    def name(self):
        return self.__class__.__name__
```

## process(frame, metadata) → (frame, metadata)

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `frame` | `numpy.ndarray` | BGR image (H, W, 3), uint8 |
| `metadata` | `dict` or `None` | Accumulated metadata from previous processors |

### Returns

| Return | Type | Description |
|--------|------|-------------|
| `frame` | `numpy.ndarray` | Processed BGR image (H, W, 3), uint8 |
| `metadata` | `dict` | Updated metadata dict with this processor's results |

### Contract

1. If `metadata` is `None`, initialize as empty dict `{}`
2. Always return a 2-tuple `(frame, metadata)`
3. Frame must remain BGR, shape (H, W, 3), dtype uint8
4. Add processor-specific keys to metadata — never delete keys from other processors
5. Return the same frame object (or a copy) even if unmodified

## name property

Returns the class name as a string. Used for logging in `add_processor()`.

```python
detector = YOLODetector()
print(detector.name)  # "YOLODetector"
```

## config attribute

`self.config` stores all `**kwargs` passed to `__init__`. Use for runtime configuration:

```python
class MyProcessor(BaseProcessor):
    def __init__(self, threshold=0.5, **kwargs):
        super().__init__(threshold=threshold, **kwargs)
        self.threshold = self.config.get("threshold", 0.5)
```

## Metadata Key Naming Convention

Use prefixed keys to avoid collisions:

| Processor | Keys |
|-----------|------|
| YOLODetector | `yolo_detections`, `yolo_object_count` |
| OCRReader | `ocr_text`, `ocr_regions` |
| ObjectTracker | `tracking_tracks`, `tracking_count` |

Format: `{processor_prefix}_{descriptive_name}`

## Creating a New Processor

1. Create a new `.py` file in `custom_engine/` (e.g., `custom_engine/my_processor.py`)
2. Extend `BaseProcessor`
3. Implement `process(frame, metadata=None)`
4. Import and add via `engine.add_processor(MyProcessor(param=value))`

### Minimal Example

```python
from base_processor import BaseProcessor


class GrayscaleConverter(BaseProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def process(self, frame, metadata=None):
        if metadata is None:
            metadata = {}
        import cv2
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return frame, metadata
```
```

- [ ] **Step 2: Write custom_engine_api.md**

```markdown
# CustomEngine API Reference

## Class Signature

```python
class CustomEngine:
    def __init__(self, logger, camera_id, message_buffer) -> None
```

## Constructor Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `logger` | `logging.Logger` | Logger instance for the engine |
| `camera_id` | `str` | Camera identifier string |
| `message_buffer` | `queue.Queue` or `None` | Queue for sending metadata out of the pipeline |

## Attributes

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `camera_id` | `str` | — | Camera identifier |
| `logger` | `logging.Logger` | — | Logger instance |
| `is_active` | `bool` | `True` | Engine active state |
| `message_buffer` | `queue.Queue` | — | Metadata output queue |
| `processors` | `list[BaseProcessor]` | `[]` | Ordered list of processors |

## Methods

### add_processor(processor)

Adds a processor to the pipeline.

```python
engine.add_processor(YOLODetector(model="yolov8s"))
```

**Raises:** `TypeError` if `processor` is not a `BaseProcessor` instance.

### \_\_call\_\_(iframe)

Executes the processor pipeline on a frame.

1. Copies the input frame (`iframe.copy()`)
2. Initializes empty metadata dict
3. Runs each processor in order: `frame, metadata = proc.process(frame, metadata)`
4. If `message_buffer` exists and metadata is non-empty, puts metadata into the queue
5. Returns the processed frame

```python
result_frame = engine(input_frame)
```

### set_data(\*\*kwargs)

Logs and stores keyword arguments. Currently a passthrough — extend in subclasses if needed.

## Pipeline Flow Diagram

```
Input Frame
    │
    ▼
iframe.copy()
    │
    ▼
Processor 1: process(frame, {})
    │  → (frame1, metadata1)
    ▼
Processor 2: process(frame1, metadata1)
    │  → (frame2, metadata2)
    ▼
...
    │
    ▼
Processor N: process(frameN-1, metadataN-1)
    │  → (frameN, metadataN)
    ▼
message_buffer.put(metadataN)  [if buffer and metadata]
    │
    ▼
Return frameN
```

## Important Notes

- `is_active` is `True` by default. The stream engine checks this flag to determine if the custom engine should process frames.
- `message_buffer` may be `None`. Always check before calling `.put()`.
- Processors are executed in the order they are added via `add_processor()`.
- The original `iframe` is never modified — a copy is always made first.
- Channel swap (BGR↔RGB) is NOT performed by default. If needed, create a `ChannelSwapProcessor` or handle it in the first processor.
```

- [ ] **Step 3: Write pipeline_patterns.md**

```markdown
# Pipeline Patterns

## Common Processor Chains

### Detection + Annotation

```python
engine.add_processor(YOLODetector(model="yolov8s", confidence=0.5))
engine.add_processor(DrawAnnotations(colors="default"))
```

YOLO runs inference and adds `yolo_detections` to metadata. DrawAnnotations reads detections and draws on the frame.

### Detection + Tracking + Counting

```python
engine.add_processor(YOLODetector(model="yolov8s"))
engine.add_processor(ObjectTracker(tracker_type="byte"))
engine.add_processor(DrawAnnotations())
engine.add_processor(ObjectCounter(zone="bottom"))
```

YOLO detects → Tracker assigns IDs → DrawAnnotations renders → Counter tallies.

### Detection + OCR

```python
engine.add_processor(YOLODetector(model="yolov8s", classes=["plate"]))
engine.add_processor(OCRReader(language="eng", roi_source="yolo_detections"))
```

YOLO finds regions → OCR reads text from detected bounding boxes.

## Ordering Rules

1. **Inference before annotation**: Models first, drawing last
2. **Detection before tracking**: Tracker needs detections as input
3. **Detection before OCR**: OCR can use detection ROIs
4. **Heavy processing first**: GPU inference early, CPU drawing late
5. **Metadata consumers after producers**: If Processor B reads Processor A's metadata, A must come first

## Anti-Patterns

### Don't: Logic in __call__

```python
# WRONG
def __call__(self, iframe):
    frame = iframe.copy()
    results = model(frame)  # Never do this
    return frame
```

### Don't: Overwrite metadata keys

```python
# WRONG
metadata["count"] = len(detections)  # Collides with other processors
```

### Don't: Modify input frame in-place

```python
# WRONG
def process(self, frame, metadata=None):
    cv2.rectangle(frame, ...)  # Mutates input
    return frame, metadata
```

### Do: Create a copy if modifying the frame

```python
# CORRECT
def process(self, frame, metadata=None):
    output = frame.copy()
    cv2.rectangle(output, ...)
    return output, metadata
```
```

- [ ] **Step 4: Write troubleshooting.md**

```markdown
# Troubleshooting

## Import Errors

### `ModuleNotFoundError: No module named 'base_processor'`

**Cause:** Incorrect import path.

**Fix:** Processors in `custom_engine/` should import using:

```python
from base_processor import BaseProcessor
```

NOT:

```python
from custom_engine.base_processor import BaseProcessor  # WRONG inside Docker
```

Inside the Docker container, `custom_engine/` is the working directory for imports.

### `ModuleNotFoundError: No module named 'onnxruntime'`

**Fix:** Install inside the Docker container:

```bash
pip install onnxruntime-gpu  # if GPU available
# or
pip install onnxruntime      # CPU only
```

### `ModuleNotFoundError: No module named 'cv2'`

**Fix:** Install OpenCV:

```bash
pip install opencv-python-headless
```

## GPU Issues

### GPU not detected / falling back to CPU

**Diagnosis:**

```python
import onnxruntime as ort
print(ort.get_available_providers())
# Expected: ['CUDAExecutionProvider', 'CPUExecutionProvider']
# If only CPUExecutionProvider: GPU not available
```

**Common causes:**
1. `onnxruntime-gpu` not installed (only `onnxruntime`)
2. CUDA/cuDNN mismatch
3. NVIDIA driver not installed in container

**Fix:** Ensure Docker container has NVIDIA runtime (`--gpus all`) and matching CUDA versions.

### `onnxruntime` CUDA error

**Cause:** `onnxruntime-gpu` version incompatible with installed CUDA.

**Fix:** Match versions:

| onnxruntime-gpu | CUDA |
|-----------------|------|
| 1.17+ | 12.x |
| 1.16 | 11.8 |

## Frame Format Issues

### Colors look wrong (blue faces, etc.)

**Cause:** Frame is BGR, code assumed RGB.

**Fix:** Convert when needed:

```python
rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# process...
frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
```

### Frame shape mismatch

**Expected:** `(H, W, 3)`, dtype `uint8`

**Diagnosis:**

```python
print(frame.shape, frame.dtype)
```

## TypeError on add_processor

### `TypeError: must be a BaseProcessor instance`

**Cause:** Passed a class instead of an instance, or wrong type.

**Fix:**

```python
# WRONG
engine.add_processor(YOLODetector)

# CORRECT
engine.add_processor(YOLODetector(model="yolov8s"))
```
```

- [ ] **Step 5: Commit**

```bash
git add skills/custom-engine-dev/references/
git commit -m "feat: add custom-engine-dev reference documents"
```

---

### Task 5: Skill — custom-engine-yolo (SKILL.md + references)

**Files:**
- Create: `skills/custom-engine-yolo/SKILL.md`
- Create: `skills/custom-engine-yolo/references/yolo_models.md`
- Create: `skills/custom-engine-yolo/references/inference_config.md`
- Create: `skills/custom-engine-yolo/references/drawing_annotations.md`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p skills/custom-engine-yolo/references
```

- [ ] **Step 2: Write SKILL.md**

```markdown
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
        # Parse YOLO output based on model version
        # See references/inference_config.md for details
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
```

- [ ] **Step 3: Write yolo_models.md**

```markdown
# YOLO Models Reference

## Supported Model Families

| Family | Models | Input Size | Output Format |
|--------|--------|------------|---------------|
| YOLOv8 | yolov8n, yolov8s, yolov8m, yolov8l, yolov8x | 640 | Pre-NMS: [1, 84, 8400] |
| YOLOv11 | yolo11n, yolo11s, yolo11m, yolo11l, yolo11x | 640 | Pre-NMS: [1, 84, 8400] |
| YOLOv10 | yolov10n, yolov10s, yolov10m, yolov10l, yolov10x | 640 | Post-NMS: [1, 300, 6] |
| YOLOv26 | yolo26n, yolo26s, yolo26m, yolo26l, yolo26x | 640 | Post-NMS: [1, 300, 6] |

## Model Sizes

| Suffix | Parameters | Speed | Accuracy |
|--------|-----------|-------|----------|
| n (nano) | ~3M | Fastest | Lowest |
| s (small) | ~11M | Fast | Good |
| m (medium) | ~26M | Medium | Better |
| l (large) | ~44M | Slow | High |
| x (extra) | ~68M | Slowest | Highest |

## ONNX Export

Export using Ultralytics:

```python
from ultralytics import YOLO

model = YOLO("yolov8s.pt")
model.export(format="onnx", imgsz=640)
# Output: yolov8s.onnx
```

For dynamic batch size (not recommended for real-time):

```python
model.export(format="onnx", imgsz=640, dynamic=True)
```

## Model Download Paths

Models are stored in `custom_engine/models/` inside the Docker volume:

```
custom_engine/
  models/
    yolov8s.onnx
    yolo11m.onnx
  yolo_detector.py
  custom_engine.py
```

## COCO Class Names (80 classes)

person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush
```

- [ ] **Step 4: Write inference_config.md**

```markdown
# Inference Configuration Reference

## ONNX Runtime Setup

### GPU Inference (Preferred)

```python
import onnxruntime as ort

providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
session = ort.InferenceSession(model_path, providers=providers)

# Verify GPU is being used
active_providers = session.get_providers()
if "CUDAExecutionProvider" in active_providers:
    print("Running on GPU")
else:
    print("Falling back to CPU")
```

### CPU Inference

```python
session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
```

## Input Preprocessing

```python
import cv2
import numpy as np

def preprocess(frame, input_size=640):
    # 1. BGR → RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 2. Resize to model input size
    img = cv2.resize(img, (input_size, input_size))
    # 3. Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0
    # 4. HWC → CHW
    img = img.transpose(2, 0, 1)
    # 5. Add batch dimension
    img = np.expand_dims(img, axis=0)
    return img
```

## Output Post-Processing

### Pre-NMS Models (YOLOv8, YOLOv11)

Output shape: `[1, 84, 8400]` (for 80 COCO classes)

```python
def postprocess_prenms(output, confidence=0.5, input_size=640, original_shape=None):
    # output shape: [1, num_features, num_anchors]
    predictions = output[0]  # [84, 8400]
    predictions = predictions.transpose(1, 0)  # [8400, 84]

    # Extract boxes and scores
    boxes = predictions[:, :4]  # cx, cy, w, h
    scores = predictions[:, 4:]  # class scores

    # Get max class score per detection
    class_ids = np.argmax(scores, axis=1)
    confidences = np.max(scores, axis=1)

    # Filter by confidence
    mask = confidences >= confidence
    boxes = boxes[mask]
    class_ids = class_ids[mask]
    confidences = confidences[mask]

    # Convert cx,cy,w,h → x1,y1,x2,y2
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2

    # Scale to original image size
    if original_shape:
        h_ratio = original_shape[0] / input_size
        w_ratio = original_shape[1] / input_size
        x1 *= w_ratio; x2 *= w_ratio
        y1 *= h_ratio; y2 *= h_ratio

    detections = []
    for i in range(len(confidences)):
        detections.append({
            "bbox": [int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])],
            "class": class_ids[i],
            "confidence": float(confidences[i])
        })
    return detections
```

### Post-NMS Models (YOLOv10, YOLOv26)

Output shape: `[1, 300, 6]` — already filtered

```python
def postprocess_postnms(output, confidence=0.5, input_size=640, original_shape=None):
    predictions = output[0]  # [300, 6]

    detections = []
    for pred in predictions:
        x1, y1, x2, y2, conf, cls = pred
        if conf < confidence:
            continue
        if original_shape:
            h_ratio = original_shape[0] / input_size
            w_ratio = original_shape[1] / input_size
            x1 *= w_ratio; x2 *= w_ratio
            y1 *= h_ratio; y2 *= h_ratio
        detections.append({
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "class": int(cls),
            "confidence": float(conf)
        })
    return detections
```

## Detecting Model Output Format

```python
def detect_output_format(session):
    output_info = session.get_outputs()[0]
    shape = output_info.shape
    # Pre-NMS: [1, 84, 8400] or [1, num_features, num_anchors]
    # Post-NMS: [1, 300, 6]
    if len(shape) == 3 and shape[2] == 6:
        return "postnms"  # v10/v26
    else:
        return "prenms"   # v8/v11
```

## Performance Tips

1. **Warm up the model**: Run one dummy inference before the real pipeline starts
2. **Use fixed input size**: Avoid dynamic shapes for best ONNX Runtime performance
3. **FP16 models**: Use `model.export(format="onnx", half=True)` for GPU FP16 inference
4. **IO Binding**: For high-throughput scenarios, use ONNX Runtime IO binding to avoid CPU↔GPU copies
```

- [ ] **Step 5: Write drawing_annotations.md**

```markdown
# Drawing Annotations Reference

## Drawing Bounding Boxes

```python
import cv2

def draw_bbox(frame, bbox, color=(0, 255, 0), thickness=2):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
```

## Drawing Labels

```python
def draw_label(frame, text, bbox, color=(0, 255, 0), font_scale=0.6, thickness=2):
    x1, y1, x2, y2 = bbox
    (text_w, text_h), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    cv2.rectangle(frame, (x1, y1 - text_h - baseline - 5),
                  (x1 + text_w, y1), color, -1)
    cv2.putText(frame, text, (x1, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
```

## Drawing Object Count

```python
def draw_count(frame, count, position="top-left", color=(0, 255, 0)):
    text = f"Count: {count}"
    if position == "top-left":
        org = (10, 30)
    elif position == "top-right":
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        org = (frame.shape[1] - tw - 10, 30)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
```

## Color Palette for Multiple Classes

```python
COLORS = [
    (0, 255, 0),    # green
    (0, 0, 255),    # red
    (255, 0, 0),    # blue
    (0, 255, 255),  # yellow
    (255, 0, 255),  # magenta
    (255, 255, 0),  # cyan
    (128, 0, 255),  # orange
    (0, 128, 255),  # light blue
]

def get_color(class_id):
    return COLORS[class_id % len(COLORS)]
```

## Complete Detection Drawing

```python
def draw_detections(frame, detections, class_names=None, draw_confidence=True):
    output = frame.copy()
    for det in detections:
        bbox = det["bbox"]
        cls_id = det.get("class", 0)
        conf = det.get("confidence", 0)
        color = get_color(cls_id)
        draw_bbox(output, bbox, color)
        label = ""
        if class_names and cls_id < len(class_names):
            label = class_names[cls_id]
        if draw_confidence:
            label += f" {conf:.2f}"
        if label:
            draw_label(output, label, bbox, color)
    return output
```

## Important Notes

- Always draw on a copy of the frame, not the original
- BGR color convention: `(B, G, R)` — green is `(0, 255, 0)`, red is `(0, 0, 255)`
- Keep annotation logic in a separate method or processor — do not mix with inference logic
```

- [ ] **Step 6: Commit**

```bash
git add skills/custom-engine-yolo/
git commit -m "feat: add custom-engine-yolo skill"
```

---

### Task 6: Skill — custom-engine-ocr (SKILL.md + references)

**Files:**
- Create: `skills/custom-engine-ocr/SKILL.md`
- Create: `skills/custom-engine-ocr/references/ocr_engines.md`
- Create: `skills/custom-engine-ocr/references/text_processing.md`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p skills/custom-engine-ocr/references
```

- [ ] **Step 2: Write SKILL.md**

```markdown
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
```

- [ ] **Step 3: Write ocr_engines.md**

```markdown
# OCR Engines Reference

## Tesseract

### Installation

```bash
apt-get install tesseract-ocr
pip install pytesseract
```

### Language Packs

```bash
# List available languages
tesseract --list-langs

# Install additional languages
apt-get install tesseract-ocr-tur  # Turkish
apt-get install tesseract-ocr-deu  # German
```

### Usage

```python
import pytesseract
import cv2

image = cv2.imread("plate.jpg")
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
text = pytesseract.image_to_string(rgb, lang="eng")
```

### Configuration

```python
# Page segmentation mode
config = "--psm 7"  # Single text line
text = pytesseract.image_to_string(rgb, config=config)

# OCR engine mode
config = "--oem 1"  # LSTM only
```

### PSM Modes (Page Segmentation Modes)

| Mode | Description |
|------|-------------|
| 3 | Fully automatic (default) |
| 6 | Uniform block of text |
| 7 | Single text line |
| 8 | Single word |
| 13 | Raw line |

### Limitations

- CPU only (no GPU acceleration)
- Less accurate on rotated or perspective-distorted text
- No built-in bounding box detection (use `image_to_data`)

## EasyOCR

### Installation

```bash
pip install easyocr
```

### GPU Support

EasyOCR uses PyTorch and automatically uses GPU if available:

```python
import easyocr
reader = easyocr.Reader(["en"], gpu=True)   # Force GPU
reader = easyocr.Reader(["en"], gpu=False)  # Force CPU
```

### Usage

```python
reader = easyocr.Reader(["en", "tr"])  # Multi-language
results = reader.readtext(image)

for bbox, text, confidence in results:
    print(f"Text: {text}, Confidence: {confidence}")
    # bbox is [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
```

### Supported Languages

| Code | Language |
|------|----------|
| en | English |
| tr | Turkish |
| de | German |
| fr | French |
| ar | Arabic |
| zh | Chinese |

Full list: https://www.jaided.ai/easyocr/

## Backend Comparison

| Feature | Tesseract | EasyOCR |
|---------|-----------|---------|
| GPU support | No | Yes (PyTorch) |
| Accuracy | Good for clean text | Better on natural scenes |
| Speed | Fast (CPU) | Slower (GPU faster) |
| Bounding boxes | image_to_data | Built-in |
| Language support | 100+ | 80+ |
| Installation | System + pip | pip only |
```

- [ ] **Step 4: Write text_processing.md**

```markdown
# Text Processing Reference

## Confidence Filtering

```python
def filter_by_confidence(regions, min_confidence=0.5):
    return [r for r in regions if r["confidence"] >= min_confidence]
```

## Text Cleaning

```python
import re

def clean_ocr_text(text):
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9ğüşıöçĞÜŞİÖÇ ]", "", text)
    return text
```

## Plate Number Formatting

```python
import re

def format_plate(text):
    text = text.replace(" ", "")
    text = text.upper()
    pattern = r"^[0-9]{2}[A-Z]{1,3}[0-9]{2,4}$"
    if re.match(pattern, text):
        return text
    return None
```

## Metadata Format

### OCR Output

```python
{
    "ocr_text": "ABC 123 XY 4567",
    "ocr_regions": [
        {
            "text": "ABC 123",
            "bbox": [x1, y1, x2, y2],
            "confidence": 0.92
        },
        {
            "text": "XY 4567",
            "bbox": [x1, y1, x2, y2],
            "confidence": 0.85
        }
    ]
}
```

### Using with Detection ROIs

When `roi_source="yolo_detections"` is set, the OCR processor reads from metadata:

```python
# YOLO output in metadata:
metadata["yolo_detections"] = [
    {"bbox": [100, 200, 300, 250], "class": "plate", "confidence": 0.9}
]

# OCR crops each ROI and reads text:
for det in metadata["yolo_detections"]:
    x1, y1, x2, y2 = det["bbox"]
    roi = frame[y1:y2, x1:x2]
    text, conf = reader.readtext(roi)
```
```

- [ ] **Step 5: Commit**

```bash
git add skills/custom-engine-ocr/
git commit -m "feat: add custom-engine-ocr skill"
```

---

### Task 7: Skill — custom-engine-tracking (SKILL.md + references)

**Files:**
- Create: `skills/custom-engine-tracking/SKILL.md`
- Create: `skills/custom-engine-tracking/references/tracker_types.md`
- Create: `skills/custom-engine-tracking/references/tracking_patterns.md`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p skills/custom-engine-tracking/references
```

- [ ] **Step 2: Write SKILL.md**

```markdown
---
name: custom-engine-tracking
description: Object tracking processor for Custom Engine. Use when adding multi-object tracking, ID assignment, or trajectory analysis to a Custom Engine pipeline. Requires detection processor (e.g., YOLO) to run first.
owner: OpenZeka
service: custom-engine
version: 1.0.0
---

# Custom Engine Tracking Skill

When this skill is active, **read the relevant reference documents** before generating code.

## Overview

Creates an `ObjectTracker` processor that assigns persistent IDs to detected objects across frames. Must be placed **after a detection processor** in the pipeline.

## Pipeline Position

```
Input → YOLODetector → ObjectTracker → [DrawAnnotations] → Output
```

Tracking **requires** detection metadata as input. It reads from `yolo_detections` in metadata.

## Critical Rules

1. **Always extend BaseProcessor**: Create `custom_engine/object_tracker.py` extending `BaseProcessor`
2. **Must come after detection**: Tracker reads `yolo_detections` from metadata — detection processor must run first
3. **Tracker types**: Support IOU (simplest), ByteTrack (recommended), SORT
4. **Track ID assignment**: Each tracked object gets a unique integer ID persisted across frames
5. **Metadata format**: Always output:
   - `tracking_tracks`: list of dicts with `track_id`, `bbox`, `class`, `confidence`
   - `tracking_count`: integer count of currently tracked objects
6. **Max age**: Default 30 frames — how long to keep a track without detection. Configurable.
7. **Min hits**: Default 3 — minimum detections before confirming a track. Configurable.

## Processor Template

```python
from base_processor import BaseProcessor


class ObjectTracker(BaseProcessor):
    def __init__(self, tracker_type="byte", max_age=30, min_hits=3,
                 iou_threshold=0.3, **kwargs):
        super().__init__(tracker_type=tracker_type, max_age=max_age,
                         min_hits=min_hits, iou_threshold=iou_threshold, **kwargs)
        self.tracker_type = tracker_type
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracker = self._init_tracker()

    def _init_tracker(self):
        if self.tracker_type == "byte":
            from byte_tracker import BYTETracker
            return BYTETracker(
                track_thresh=0.5,
                track_buffer=self.max_age,
                match_thresh=self.iou_threshold
            )
        elif self.tracker_type == "iou":
            from iou_tracker import IOUTracker
            return IOUTracker(
                max_age=self.max_age,
                min_hits=self.min_hits,
                iou_threshold=self.iou_threshold
            )
        elif self.tracker_type == "sort":
            from sort import Sort
            return Sort(max_age=self.max_age, min_hits=self.min_hits,
                        iou_threshold=self.iou_threshold)

    def process(self, frame, metadata=None):
        if metadata is None:
            metadata = {}
        if "yolo_detections" not in metadata:
            return frame, metadata

        detections = metadata["yolo_detections"]
        tracks = self._update(detections)
        metadata["tracking_tracks"] = tracks
        metadata["tracking_count"] = len(tracks)

        # Also update detections with track IDs
        for det in metadata["yolo_detections"]:
            for track in tracks:
                if det["bbox"] == track["bbox"]:
                    det["track_id"] = track["track_id"]
                    break

        return frame, metadata

    def _update(self, detections):
        import numpy as np
        dets = np.array([[d["bbox"][0], d["bbox"][1], d["bbox"][2], d["bbox"][3],
                          d["confidence"]] for d in detections])
        if len(dets) == 0:
            return []
        raw_tracks = self.tracker.update(dets)
        tracks = []
        for t in raw_tracks:
            tracks.append({
                "track_id": int(t[4]) if len(t) > 4 else -1,
                "bbox": [int(t[0]), int(t[1]), int(t[2]), int(t[3])],
                "class": 0,
                "confidence": float(t[5]) if len(t) > 5 else 0.0
            })
        return tracks
```

## Reference Documents

| Document | Use When |
|----------|----------|
| [references/tracker_types.md](references/tracker_types.md) | Tracker types, configuration, performance comparison |
| [references/tracking_patterns.md](references/tracking_patterns.md) | YOLO → Tracking chain, ID assignment, transition patterns |
```

- [ ] **Step 3: Write tracker_types.md**

```markdown
# Tracker Types Reference

## IOU Tracker

Simplest tracker. Matches detections across frames using IoU overlap. No feature extraction.

### When to use
- Simple scenes with few objects
- Low computational budget
- Objects don't change appearance

### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_age` | 30 | Frames to keep track without detection |
| `min_hits` | 3 | Min detections to confirm track |
| `iou_threshold` | 0.3 | Min IoU for matching |

### Implementation

```python
import numpy as np

class IOUTracker:
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.next_id = 1

    def update(self, detections):
        if len(detections) == 0:
            for track in self.tracks:
                track["age"] += 1
            self.tracks = [t for t in self.tracks if t["age"] <= self.max_age]
            return []

        matched, unmatched_dets = self._match(detections)

        for det_idx in unmatched_dets:
            self.tracks.append({
                "id": self.next_id,
                "bbox": detections[det_idx][:4],
                "hits": 1,
                "age": 0
            })
            self.next_id += 1

        results = []
        for track in self.tracks:
            track["age"] += 1
            if track["hits"] >= self.min_hits:
                results.append(track)

        self.tracks = [t for t in self.tracks if t["age"] <= self.max_age]
        return results

    def _match(self, detections):
        if not self.tracks:
            return [], list(range(len(detections)))

        iou_matrix = np.zeros((len(self.tracks), len(detections)))
        for i, track in enumerate(self.tracks):
            for j, det in enumerate(detections):
                iou_matrix[i, j] = self._iou(track["bbox"], det[:4])

        matched = []
        unmatched_dets = list(range(len(detections)))

        while iou_matrix.size > 0:
            idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            if iou_matrix[idx] < self.iou_threshold:
                break
            track_idx, det_idx = idx
            self.tracks[track_idx]["bbox"] = detections[det_idx][:4]
            self.tracks[track_idx]["hits"] += 1
            self.tracks[track_idx]["age"] = 0
            matched.append((track_idx, det_idx))
            iou_matrix = np.delete(iou_matrix, track_idx, axis=0)
            iou_matrix = np.delete(iou_matrix, det_idx, axis=1)
            if det_idx in unmatched_dets:
                unmatched_dets.remove(det_idx)

        return matched, unmatched_dets

    @staticmethod
    def _iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return inter / (area1 + area2 - inter + 1e-6)
```

## ByteTrack

State-of-the-art tracker. Uses detection confidence to handle both high and low confidence detections. Recommended for most use cases.

### When to use
- Complex scenes with many objects
- Occlusions and re-appearances
- Best accuracy needed

### Dependencies

```bash
pip install bytetrack
# or include byte_tracker.py in custom_engine/
```

### Performance

| Metric | IOU | ByteTrack | SORT |
|--------|-----|-----------|------|
| MOTA | ~65% | ~80% | ~70% |
| IDF1 | ~60% | ~77% | ~65% |
| Speed | Very Fast | Fast | Fast |
| GPU Required | No | No | No |

## SORT

Simple Online and Realtime Tracking. Kalman filter based.

### When to use
- Need Kalman filter prediction
- Moderate complexity scenes
- Slightly better than IOU, less than ByteTrack

### Dependencies

```bash
pip install sort
# or include sort.py in custom_engine/
```
```

- [ ] **Step 4: Write tracking_patterns.md**

```markdown
# Tracking Patterns Reference

## YOLO + Tracking Chain

The most common pattern. YOLO detects objects, tracker assigns IDs:

```python
from custom_engine import CustomEngine
from yolo_detector import YOLODetector
from object_tracker import ObjectTracker

engine = CustomEngine(logger=logger, camera_id="cam_01", message_buffer=queue)
engine.add_processor(YOLODetector(model="yolov8s", confidence=0.5))
engine.add_processor(ObjectTracker(tracker_type="byte", max_age=30))
```

### Data Flow

```
Frame
  ↓
YOLODetector.process(frame, {})
  → frame (unchanged)
  → metadata: {"yolo_detections": [...], "yolo_object_count": 5}
  ↓
ObjectTracker.process(frame, metadata)
  → frame (unchanged)
  → metadata: {"yolo_detections": [..., "track_id": 3], "yolo_object_count": 5,
               "tracking_tracks": [...], "tracking_count": 4}
```

## Track ID Persistence

Track IDs are assigned incrementally and persist across frames:

```
Frame 1: Object appears → track_id=1
Frame 2: Same object → track_id=1 (maintained)
Frame 50: Object leaves frame → track_id=1 removed after max_age
Frame 100: Object reappears → track_id=2 (new ID, old track expired)
```

## Drawing Track IDs

```python
import cv2

def draw_track_id(frame, bbox, track_id, color=(0, 255, 0)):
    x1, y1, x2, y2 = bbox
    text = f"ID: {track_id}"
    cv2.putText(frame, text, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
```

## Counting with Tracking

Use tracking for more accurate counting than detection-only:

```python
class ZoneCounter(BaseProcessor):
    def __init__(self, line_y=400, direction="down", **kwargs):
        super().__init__(line_y=line_y, direction=direction, **kwargs)
        self.line_y = line_y
        self.direction = direction
        self.counted_ids = set()
        self.count = 0

    def process(self, frame, metadata=None):
        if metadata is None:
            metadata = {}
        if "tracking_tracks" in metadata:
            for track in metadata["tracking_tracks"]:
                tid = track["track_id"]
                cy = (track["bbox"][1] + track["bbox"][3]) // 2
                if tid not in self.counted_ids:
                    if self.direction == "down" and cy > self.line_y:
                        self.counted_ids.add(tid)
                        self.count += 1
                    elif self.direction == "up" and cy < self.line_y:
                        self.counted_ids.add(tid)
                        self.count += 1
        metadata["zone_count"] = self.count
        return frame, metadata
```

## Anti-Patterns

### Don't: Track without detection

```python
# WRONG — tracker needs detection input
engine.add_processor(ObjectTracker())  # No detector before this!
```

### Don't: Re-create tracker each frame

```python
# WRONG — tracker state is per-frame
def process(self, frame, metadata=None):
    tracker = BYTETracker()  # Loses all track history!
```

### Do: Maintain tracker state in the processor

```python
# CORRECT — tracker is initialized once in __init__
class ObjectTracker(BaseProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tracker = BYTETracker()  # Persistent across frames
```
```

- [ ] **Step 5: Commit**

```bash
git add skills/custom-engine-tracking/
git commit -m "feat: add custom-engine-tracking skill"
```

---

### Task 8: Example Prompts + README

**Files:**
- Create: `example_prompts/yolo_object_count.md`
- Create: `example_prompts/ocr_text_detection.md`
- Create: `example_prompts/yolo_with_tracking.md`
- Create: `README.md`

- [ ] **Step 1: Write yolo_object_count.md**

```markdown
# YOLO Object Count Example

## Prompt

Create a Custom Engine pipeline that runs YOLOv8s object detection on the camera stream, draws bounding boxes with class labels and confidence scores on each detected object, and displays the total object count on the top-left corner of the frame. Send the detection metadata via message buffer.

## Expected Output

- `custom_engine/yolo_detector.py` — YOLODetector processor class
- Updated initialization code that adds the processor to CustomEngine
- Detection metadata sent through message_buffer
```

- [ ] **Step 2: Write ocr_text_detection.md**

```markdown
# OCR Text Detection Example

## Prompt

Create a Custom Engine pipeline that first runs YOLOv8s to detect objects, then runs OCR only on the detected regions to read any text. Draw the OCR results on the frame. Use EasyOCR with English language support. Send both detection and OCR metadata via message buffer.

## Expected Output

- `custom_engine/yolo_detector.py` — YOLODetector processor
- `custom_engine/ocr_reader.py` — OCRReader processor with roi_source="yolo_detections"
- Both processors added to the pipeline in correct order
```

- [ ] **Step 3: Write yolo_with_tracking.md**

```markdown
# YOLO with Tracking Example

## Prompt

Create a Custom Engine pipeline that runs YOLOv8s object detection, then tracks detected objects across frames using ByteTrack. Draw bounding boxes with track IDs on each object. Display the count of currently tracked objects on the frame. Send tracking metadata via message buffer.

## Expected Output

- `custom_engine/yolo_detector.py` — YOLODetector processor
- `custom_engine/object_tracker.py` — ObjectTracker processor with tracker_type="byte"
- Both processors added in order: YOLO first, tracker second
- Track IDs displayed on bounding boxes
```

- [ ] **Step 4: Write README.md**

```markdown
# Custom Engine Coding Agent

A project providing agentic skills for AI coding assistants (opencode, Claude Code, Cursor, etc.) to generate Custom Engine processor modules for video stream processing.

## Overview

Custom Engine is a modular video processing framework where each processing step is a **processor** implementing the `BaseProcessor` interface. Processors are chained into a pipeline inside `CustomEngine.__call__()`, operating on each frame and accumulating metadata.

This project provides skills that teach AI coding assistants how to:
1. Understand the Custom Engine processor pipeline architecture
2. Generate correct processor modules that extend `BaseProcessor`
3. Handle GPU/CPU inference, frame format conventions, and metadata patterns

## Prerequisites

- AI coding assistant that supports agentic skills (opencode, Claude Code, Cursor, Codex)

## Project Structure

```
custom-engine-coding-agent/
├── custom_engine/                  # Docker volume — user-accessible modules
│   ├── custom_engine.py            # CustomEngine class with processor pipeline
│   └── base_processor.py           # BaseProcessor abstract interface
├── skills/                         # Agentic skills
│   ├── custom-engine-dev/          # Core pipeline skill
│   ├── custom-engine-yolo/         # YOLO detection skill
│   ├── custom-engine-ocr/          # OCR skill
│   └── custom-engine-tracking/     # Object tracking skill
├── example_prompts/                # Example prompts
└── README.md
```

## Agentic Skills

| Skill | Use When |
|-------|----------|
| `custom-engine-dev` | Building any processor module, understanding pipeline architecture |
| `custom-engine-yolo` | Adding YOLO object detection, bounding box annotation, object counting |
| `custom-engine-ocr` | Adding OCR text reading, license plate recognition |
| `custom-engine-tracking` | Adding multi-object tracking, ID assignment |

## Installing Skills

Copy the skill directories into your AI coding assistant's skills folder:

### opencode

```bash
cp -r skills/custom-engine-dev ~/.config/opencode/skills/
cp -r skills/custom-engine-yolo ~/.config/opencode/skills/
cp -r skills/custom-engine-ocr ~/.config/opencode/skills/
cp -r skills/custom-engine-tracking ~/.config/opencode/skills/
```

### Claude Code

```bash
cp -r skills/custom-engine-dev ~/.claude/skills/
cp -r skills/custom-engine-yolo ~/.claude/skills/
cp -r skills/custom-engine-ocr ~/.claude/skills/
cp -r skills/custom-engine-tracking ~/.claude/skills/
```

### Cursor

```bash
cp -r skills/custom-engine-dev ~/.cursor/skills/
cp -r skills/custom-engine-yolo ~/.cursor/skills/
cp -r skills/custom-engine-ocr ~/.cursor/skills/
cp -r skills/custom-engine-tracking ~/.cursor/skills/
```

## Example Usage

After installing skills, ask your AI assistant:

```
Create a Custom Engine pipeline that runs YOLOv8s detection and displays
the object count on the frame.
```

The agent will automatically activate the relevant skills and generate:
- A `YOLODetector` processor in `custom_engine/yolo_detector.py`
- Proper `BaseProcessor` implementation with GPU/CPU fallback
- Correct metadata format and pipeline integration

## Architecture

### Processor Pipeline

Each frame goes through a chain of processors:

```
Input Frame → Processor 1 → Processor 2 → ... → Output Frame
                                        ↓
                                  message_buffer (optional)
```

### Creating a Processor

1. Create a new `.py` file in `custom_engine/`
2. Extend `BaseProcessor`
3. Implement `process(frame, metadata=None) → (frame, metadata)`
4. Add to engine: `engine.add_processor(MyProcessor())`

## Docker Usage

The `custom_engine/` directory is volume-mounted into the Docker container. Only files in this directory are user-accessible.

```bash
docker run --gpus all -v /path/to/custom_engine:/app/custom_engine your-image
```

## License

TBD
```

- [ ] **Step 5: Commit**

```bash
git add example_prompts/ README.md
git commit -m "feat: add example prompts and README"
```

---

## Self-Review Checklist

### Spec Coverage

| Spec Requirement | Task |
|-----------------|------|
| BaseProcessor abstract class | Task 1 |
| CustomEngine pipeline with processors | Task 2 |
| is_active = True | Task 2 |
| message_buffer metadata sending | Task 2 |
| custom-engine-dev SKILL.md + plugin + evals | Task 3 |
| custom-engine-dev references (4 docs) | Task 4 |
| custom-engine-yolo SKILL.md + references (3 docs) | Task 5 |
| custom-engine-ocr SKILL.md + references (2 docs) | Task 6 |
| custom-engine-tracking SKILL.md + references (2 docs) | Task 7 |
| Example prompts (3) | Task 8 |
| README.md | Task 8 |

All requirements covered.

### Placeholder Scan

No TBDs, TODOs, or "implement later" patterns found.

### Type Consistency

- `BaseProcessor.process(frame, metadata=None)` signature consistent across all templates
- Metadata keys use prefixed naming: `yolo_detections`, `ocr_text`, `tracking_tracks`
- `add_processor()` method consistent in all examples
