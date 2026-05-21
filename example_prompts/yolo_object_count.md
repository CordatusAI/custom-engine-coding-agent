# YOLO Object Count Example

## Prompt

Create a Custom Engine pipeline that runs YOLOv8s object detection on the camera stream, draws bounding boxes with class labels and confidence scores on each detected object, and displays the total object count on the top-left corner of the frame. Send the detection metadata via message buffer.

## Expected Output

- `custom_engine/yolo_detector.py` — YOLODetector processor class
- Updated initialization code that adds the processor to CustomEngine
- Detection metadata sent through message_buffer
