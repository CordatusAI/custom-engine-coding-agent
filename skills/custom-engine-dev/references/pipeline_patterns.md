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