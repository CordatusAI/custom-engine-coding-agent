# YOLO Models Reference

## Supported Model Families

| Family | Models | Input Size | Output Format |
|--------|--------|------------|---------------|
| YOLOv8 | yolov8n, yolov8s, yolov8m, yolov8l, yolov8x | 640 | Pre-NMS: [1, 84, 8400] |
| YOLOv11 | yolo11n, yolo11s, yolo11m, yolo11l, yolo11x | 640 | Pre-NMS: [1, 84, 8400] |
| YOLOv10 | yolov10n, yolov10s, yolov10m, yolov10l, yolov10x | 640 | Post-NMS: [1, 300, 6] |
| YOLOv26 | yolo26n, yolo26s, yolo26m, yolo26l, yolo26x | 640 | Post-NMS: [1, 300, 6] |

## Model Sizes

| Suffix | Parameters | Speed | Accuracy |
|--------|-----------|-------|----------|
| n (nano) | ~3M | Fastest | Lowest |
| s (small) | ~11M | Fast | Good |
| m (medium) | ~26M | Medium | Better |
| l (large) | ~44M | Slow | High |
| x (extra) | ~68M | Slowest | Highest |

## ONNX Export

Export using Ultralytics:

```python
from ultralytics import YOLO

model = YOLO("yolov8s.pt")
model.export(format="onnx", imgsz=640)
```

For dynamic batch size (not recommended for real-time):

```python
model.export(format="onnx", imgsz=640, dynamic=True)
```

## Model Download Paths

Models are stored in `custom_engine/models/` inside the Docker volume:

```
custom_engine/
  models/
    yolov8s.onnx
    yolo11m.onnx
  yolo_detector.py
  custom_engine.py
```

## COCO Class Names (80 classes)

person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush
