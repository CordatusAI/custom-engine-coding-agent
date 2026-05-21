# YOLO with Tracking Example

## Prompt

Create a Custom Engine pipeline that runs YOLOv8s object detection, then tracks detected objects across frames using ByteTrack. Draw bounding boxes with track IDs on each object. Display the count of currently tracked objects on the frame. Send tracking metadata via message buffer.

## Expected Output

- `custom_engine/yolo_detector.py` — YOLODetector processor
- `custom_engine/object_tracker.py` — ObjectTracker processor with tracker_type="byte"
- Both processors added in order: YOLO first, tracker second
- Track IDs displayed on bounding boxes
