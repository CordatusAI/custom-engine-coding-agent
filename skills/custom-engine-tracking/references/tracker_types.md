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
```
