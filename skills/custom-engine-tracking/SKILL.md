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
