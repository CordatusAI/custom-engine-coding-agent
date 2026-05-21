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
