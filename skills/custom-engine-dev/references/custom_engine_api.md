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