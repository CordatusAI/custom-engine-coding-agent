# OCR Engines Reference

## Tesseract

### Installation

```bash
apt-get install tesseract-ocr
pip install pytesseract
```

### Language Packs

```bash
tesseract --list-langs
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
config = "--psm 7"  # Single text line
text = pytesseract.image_to_string(rgb, config=config)

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

```python
import easyocr
reader = easyocr.Reader(["en"], gpu=True)   # Force GPU
reader = easyocr.Reader(["en"], gpu=False)  # Force CPU
```

### Usage

```python
reader = easyocr.Reader(["en", "tr"])
results = reader.readtext(image)

for bbox, text, confidence in results:
    print(f"Text: {text}, Confidence: {confidence}")
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
