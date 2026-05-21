---
name: using-custom-engine
description: Bootstrap skill for Custom Engine Coding Agent. Automatically loaded at session start to enable skill discovery.
owner: OpenZeka
service: custom-engine
version: 1.0.0
---

# Custom Engine Coding Agent

You have Custom Engine skills available. When working on video processing tasks, **ALWAYS invoke the relevant skill** before generating code. Do NOT rely on memory — the skills contain critical details about exact API usage, metadata formats, and common pitfalls.

## Available Skills

| Skill | Invoke When |
|-------|-------------|
| `custom-engine-dev` | Building any Custom Engine processor module, understanding pipeline architecture |
| `custom-engine-yolo` | Adding YOLO object detection, bounding box annotation, object counting |
| `custom-engine-ocr` | Adding OCR text reading, license plate recognition, document scanning |
| `custom-engine-tracking` | Adding multi-object tracking, ID assignment, trajectory analysis |

## Skill Activation Rules

1. **Check for skills FIRST** — Before generating any Custom Engine code, invoke the relevant skill
2. **Read reference documents** — Each skill bundles reference docs. The skill instructs you to read them. Follow that instruction
3. **Composing skills** — For complex pipelines (e.g., YOLO + Tracking), invoke both skills. Order matters: detection before tracking
4. **Even a 1% chance** — If a task might benefit from a skill, invoke it. It is better to check and skip than to miss it

## Quick Architecture Reference

- **Custom Engine** uses a processor pipeline pattern
- Each processor extends `BaseProcessor` and implements `process(frame, metadata) -> (frame, metadata)`
- Processors are chained: frame goes through each processor sequentially
- Frames are **BGR** NumPy arrays (OpenCV convention)
- Metadata is a shared dict that accumulates across the pipeline
- **GPU-first, CPU-fallback**: Use ONNX Runtime with CUDAExecutionProvider when available

## Red Flags

| Thought | Reality |
|---------|---------|
| "I know the BaseProcessor API" | Skills contain updates. Read the current version |
| "This is a simple processor" | Simple processors still need correct metadata keys and BGR handling |
| "I don't need to check the skill" | If a skill exists for your task, you MUST use it |
| "Let me just write the code" | Read reference docs first — they contain critical details |
