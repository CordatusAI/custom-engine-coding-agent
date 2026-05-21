# OCR Text Detection Example

## Prompt

Create a Custom Engine pipeline that first runs YOLOv8s to detect objects, then runs OCR only on the detected regions to read any text. Draw the OCR results on the frame. Use EasyOCR with English language support. Send both detection and OCR metadata via message buffer.

## Expected Output

- `custom_engine/yolo_detector.py` — YOLODetector processor
- `custom_engine/ocr_reader.py` — OCRReader processor with roi_source="yolo_detections"
- Both processors added to the pipeline in correct order
