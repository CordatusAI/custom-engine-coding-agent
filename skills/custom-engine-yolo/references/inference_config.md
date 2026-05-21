# Inference Configuration Reference

## ONNX Runtime Setup

### GPU Inference (Preferred)

```python
import onnxruntime as ort

providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
session = ort.InferenceSession(model_path, providers=providers)

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
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (input_size, input_size))
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)
    return img
```

## Output Post-Processing

### Pre-NMS Models (YOLOv8, YOLOv11)

Output shape: `[1, 84, 8400]` (for 80 COCO classes)

```python
def postprocess_prenms(output, confidence=0.5, input_size=640, original_shape=None):
    predictions = output[0]
    predictions = predictions.transpose(1, 0)
    boxes = predictions[:, :4]
    scores = predictions[:, 4:]
    class_ids = np.argmax(scores, axis=1)
    confidences = np.max(scores, axis=1)
    mask = confidences >= confidence
    boxes = boxes[mask]
    class_ids = class_ids[mask]
    confidences = confidences[mask]
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
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
    predictions = output[0]
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
    if len(shape) == 3 and shape[2] == 6:
        return "postnms"
    else:
        return "prenms"
```

## Performance Tips

1. Warm up the model: Run one dummy inference before the real pipeline starts
2. Use fixed input size: Avoid dynamic shapes for best ONNX Runtime performance
3. FP16 models: Use `model.export(format="onnx", half=True)` for GPU FP16 inference
4. IO Binding: For high-throughput scenarios, use ONNX Runtime IO binding to avoid CPU↔GPU copies
