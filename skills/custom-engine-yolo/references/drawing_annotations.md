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
    (0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255),
    (255, 0, 255), (255, 255, 0), (128, 0, 255), (0, 128, 255),
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
- BGR color convention: (B, G, R) — green is (0, 255, 0), red is (0, 0, 255)
- Keep annotation logic separate from inference logic
