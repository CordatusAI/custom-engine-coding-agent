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
metadata["yolo_detections"] = [
    {"bbox": [100, 200, 300, 250], "class": "plate", "confidence": 0.9}
]

for det in metadata["yolo_detections"]:
    x1, y1, x2, y2 = det["bbox"]
    roi = frame[y1:y2, x1:x2]
    text, conf = reader.readtext(roi)
```
