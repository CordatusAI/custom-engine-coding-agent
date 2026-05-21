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