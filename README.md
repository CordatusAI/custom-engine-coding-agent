# Custom Engine Coding Agent

Agentic skills for AI coding assistants (opencode, Claude Code, Cursor, Codex) to generate **Custom Engine** processor modules for video stream processing pipelines.

This project is designed for use with the **Custom Engine** feature of the [Cordatus](https://app.cordatus.ai) platform. Cordatus is an AI-powered video intelligence platform that enables deploying and managing computer vision pipelines on edge devices. Custom Engine is the Cordatus module that lets developers write their own processing logic in Python — this coding agent provides the skills needed to generate correct, pipeline-ready processor code for it.

> **Disclaimer:** Code generated with AI coding assistants is intended as a development starting point. All generated code must undergo your full software development lifecycle (SDLC) — including code review, testing, and security validation — before production use.

---

## Overview

Custom Engine is a modular video processing framework within the Cordatus platform where each processing step is a **processor** implementing the `BaseProcessor` interface. Processors are chained into a pipeline inside `CustomEngine.__call__()`, operating on each frame and accumulating metadata.

This project provides **agentic skills** that teach AI coding assistants how to:

1. Understand the Custom Engine processor pipeline architecture
2. Generate correct processor modules that extend `BaseProcessor`
3. Handle GPU/CPU inference, frame format conventions, and metadata patterns

---

## Prerequisites

### For code generation (using the skills and prompts)

- **AI coding assistant** that supports agentic skills (opencode, Claude Code, Cursor, Codex)

No GPU, SDK, or special hardware is required — the skills and example prompts work on any system.

### For running the generated code

The following are required on the target execution environment:

- **Python 3.10+**
- **NVIDIA GPU** (recommended) with CUDA support
- **ONNX Runtime** (`onnxruntime-gpu` for GPU, `onnxruntime` for CPU fallback)
- **OpenCV** (`opencv-python`)
- Additional dependencies vary by processor (e.g., `ultralytics` for YOLO, `easyocr` for OCR)

> The `custom_engine/` directory is volume-mounted into the Docker container. Only files in this directory are user-accessible at runtime.

---

## Project Structure

```
custom-engine-coding-agent/
├── custom_engine/                  # Docker volume — user-accessible modules
│   ├── custom_engine.py            # CustomEngine class with processor pipeline
│   └── base_processor.py           # BaseProcessor abstract interface
├── skills/                         # Agentic skills
│   ├── using-custom-engine/        # Bootstrap skill (auto-loaded at session start)
│   ├── custom-engine-dev/          # Core pipeline skill
│   ├── custom-engine-yolo/         # YOLO detection skill
│   ├── custom-engine-ocr/          # OCR skill
│   └── custom-engine-tracking/     # Object tracking skill
├── hooks/                          # SessionStart hooks for Claude Code / Cursor
│   ├── hooks.json                  # Claude Code hook config
│   ├── hooks-cursor.json           # Cursor hook config
│   ├── session-start               # Bootstrap injection script
│   └── run-hook.cmd                # Cross-platform hook wrapper
├── .opencode/                      # OpenCode plugin
│   ├── plugins/custom-engine.js    # Plugin: registers skills + injects bootstrap
│   └── INSTALL.md                  # OpenCode install instructions
├── .claude-plugin/plugin.json      # Claude Code plugin manifest
├── .cursor-plugin/plugin.json      # Cursor plugin manifest
├── .codex-plugin/plugin.json       # Codex plugin manifest
├── gemini-extension.json           # Gemini CLI extension manifest
├── GEMINI.md                       # Gemini CLI context
├── example_prompts/                # Example prompts
├── docs/                           # Design docs and specs
├── package.json                    # npm package manifest
└── README.md
```

---

## Agentic Skills

An **agentic skill** is a structured knowledge package that an AI coding assistant can automatically discover and activate during code generation. It contains domain-specific rules, reference documentation, and guardrails that guide the AI agent to produce accurate, idiomatic code — without the developer needing to manually reference files in every conversation.

This project ships **four complementary skills**:

| Skill | Use When |
|-------|----------|
| `custom-engine-dev` | Building any processor module, understanding pipeline architecture |
| `custom-engine-yolo` | Adding YOLO object detection, bounding box annotation, object counting |
| `custom-engine-ocr` | Adding OCR text reading, license plate recognition |
| `custom-engine-tracking` | Adding multi-object tracking, ID assignment (requires detection processor first) |

### Skill: custom-engine-dev

The core skill for Custom Engine development. When activated, it instructs the AI agent to consult bundled reference documents before generating any code, ensuring correct API usage and pipeline integration.

**Bundled reference topics:**

| Reference | Coverage |
|-----------|----------|
| `base_processor_api.md` | BaseProcessor interface, `process()` contract, metadata format |
| `custom_engine_api.md` | CustomEngine class details, `add_processor`, `__call__` flow |
| `pipeline_patterns.md` | Processor chaining examples, ordering rules, common patterns |
| `troubleshooting.md` | Import errors, GPU/CPU fallback, frame format issues |

### Skill: custom-engine-yolo

Creates a `YOLODetector` processor that runs YOLO object detection on each frame. Supports Ultralytics YOLO family models (YOLOv8/v11/v10/v26) with ONNX Runtime inference (GPU-first, CPU-fallback).

**Pipeline position:** Should be one of the first processors, as other processors (tracking, OCR) may depend on its metadata.

```
Input → YOLODetector → [Tracker] → [OCR] → [DrawAnnotations] → Output
```

**Bundled reference topics:**

| Reference | Coverage |
|-----------|----------|
| `yolo_models.md` | Supported models, download, ONNX export |
| `inference_config.md` | GPU/CPU selection, ONNX Runtime config, post-processing |
| `drawing_annotations.md` | Bbox, label, count drawing, OpenCV annotations |

### Skill: custom-engine-ocr

Creates an `OCRReader` processor that runs OCR on each frame or on specific ROIs. Supports Tesseract and EasyOCR backends.

**Pipeline position:** Should come after detection processors when reading from detected regions (e.g., license plates).

```
Input → [YOLODetector] → OCRReader → [DrawAnnotations] → Output
```

**Bundled reference topics:**

| Reference | Coverage |
|-----------|----------|
| `ocr_engines.md` | Tesseract/EasyOCR setup, language packs, GPU support |
| `text_processing.md` | OCR result processing, confidence filtering, metadata format |

### Skill: custom-engine-tracking

Creates an `ObjectTracker` processor that assigns persistent IDs to detected objects across frames. Supports IOU, ByteTrack, and SORT trackers.

**Pipeline position:** Must be placed after a detection processor — it reads from `yolo_detections` in metadata.

```
Input → YOLODetector → ObjectTracker → [DrawAnnotations] → Output
```

**Bundled reference topics:**

| Reference | Coverage |
|-----------|----------|
| `tracker_types.md` | Tracker types, configuration, performance comparison |
| `tracking_patterns.md` | YOLO → Tracking chain, ID assignment, transition patterns |

---

## Installing Skills

Find the skills directory for your AI coding assistant and copy the skill folders from this project into it.

**Skill directories by tool:**

| Tool | User-level path | Workspace-level path |
|------|----------------|---------------------|
| OpenCode | `~/.config/opencode/skills/` | `<workspace>/.opencode/skills/` |
| Claude Code | `~/.claude/skills/` | `<workspace>/.claude/skills/` |
| Cursor | `~/.cursor/skills/` | `<workspace>/.cursor/skills/` |
| Codex | `~/.codex/skills/` | `<workspace>/.codex/skills/` |
| GitHub Copilot | `<workspace>/.github/` | `<workspace>/.github/` |

Copy these four folders into the skills directory you identified:

```
skills/custom-engine-dev/
skills/custom-engine-yolo/
skills/custom-engine-ocr/
skills/custom-engine-tracking/
```

Restart your coding assistant after copying.

### Verifying the Installation

1. Open (or restart) your AI coding assistant.
2. Ask a Custom Engine question, for example:

   ```
   Create a Custom Engine pipeline that runs YOLOv8s detection and displays
   the object count on the frame.
   ```

3. The agent should automatically activate the relevant skills and consult their reference documents before generating code.

> **Tip:** The skills are most effective in **Agent mode**. In agent mode, the AI assistant automatically selects and activates relevant skills based on the task context — no manual file referencing needed.

---

## Architecture

### Processor Pipeline

Each frame goes through a chain of processors:

```
Input Frame → Processor 1 → Processor 2 → ... → Processor N → Annotated Frame
                                                        ↓
                                                  message_buffer (optional metadata)
```

### BaseProcessor Interface

Every processor extends `BaseProcessor` and implements the `process()` method:

```python
from base_processor import BaseProcessor


class MyProcessor(BaseProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def process(self, frame, metadata=None):
        if metadata is None:
            metadata = {}
        # ... processing logic ...
        metadata["my_key"] = "my_value"
        return frame, metadata
```

**Contract:**
- `process(frame, metadata)` returns `(frame, metadata)`
- Frames are BGR NumPy arrays (OpenCV convention) — processors must return BGR frames
- Metadata is a dict that accumulates across the pipeline — each processor adds its own keys
- GPU usage is optional; each processor handles GPU/CPU fallback internally

### Adding Processors to CustomEngine

```python
from custom_engine import CustomEngine
from yolo_detector import YOLODetector
from ocr_reader import OCRReader
from object_tracker import ObjectTracker

engine = CustomEngine(logger=logger, camera_id="cam_01", message_buffer=msg_queue)
engine.add_processor(YOLODetector(model="yolov8s", confidence=0.5))
engine.add_processor(ObjectTracker(tracker_type="byte"))
engine.add_processor(OCRReader(engine="easyocr", language="en", roi_source="yolo_detections"))
```

### Creating a New Processor

1. Create a new `.py` file in `custom_engine/`
2. Extend `BaseProcessor`
3. Implement `process(frame, metadata=None) → (frame, metadata)`
4. Add to engine: `engine.add_processor(MyProcessor())`

### Metadata Format

Each processor adds its own keys to the shared metadata dict:

| Key | Processor | Description |
|-----|-----------|-------------|
| `yolo_detections` | YOLODetector | List of dicts with `bbox`, `class`, `confidence` |
| `yolo_object_count` | YOLODetector | Integer count of detections |
| `ocr_text` | OCRReader | Concatenated string of all detected text |
| `ocr_regions` | OCRReader | List of dicts with `text`, `bbox`, `confidence` |
| `tracking_tracks` | ObjectTracker | List of dicts with `track_id`, `bbox`, `class`, `confidence` |
| `tracking_count` | ObjectTracker | Integer count of currently tracked objects |

---

## Example Usage

After installing skills, ask your AI assistant:

### YOLO Object Detection

```
Create a Custom Engine pipeline that runs YOLOv8s object detection on the camera stream,
draws bounding boxes with class labels and confidence scores on each detected object,
and displays the total object count on the top-left corner of the frame.
Send the detection metadata via message buffer.
```

### OCR with Detection

```
Create a Custom Engine pipeline that first runs YOLOv8s to detect objects, then runs OCR
only on the detected regions to read any text. Draw the OCR results on the frame.
Use EasyOCR with English language support. Send both detection and OCR metadata via message buffer.
```

### YOLO with Tracking

```
Create a Custom Engine pipeline that runs YOLOv8s object detection, then tracks detected
objects across frames using ByteTrack. Draw bounding boxes with track IDs on each object.
Display the count of currently tracked objects on the frame. Send tracking metadata via message buffer.
```

For more examples, see the [`example_prompts/`](example_prompts/) directory.

---

## Using Example Prompts

The `example_prompts/` directory contains pre-built prompts for generating Custom Engine applications.

### Available Prompts

| Prompt File | Purpose |
|-------------|---------|
| `yolo_object_count.md` | YOLO detection with bounding boxes and object count |
| `ocr_text_detection.md` | YOLO detection followed by OCR on detected regions |
| `yolo_with_tracking.md` | YOLO detection with ByteTrack tracking and track IDs |

### Step-by-Step Guide

1. **Open the AI chat / agent panel** in your coding assistant.
2. **Reference the prompt file** using your tool's file-referencing feature (e.g., `@` mentions):

   ```
   @example_prompts/yolo_object_count.md
   ```

3. **Execute the prompt** — instruct the agent to follow it:

   ```
   Follow the instructions in @example_prompts/yolo_object_count.md to generate the processor.
   ```

4. **Review and iterate** — inspect the generated code, accept or reject changes, and ask for refinements.

---

## Docker Usage

The `custom_engine/` directory is volume-mounted into the Docker container. Only files in this directory are user-accessible at runtime.

```bash
docker run --gpus all -v /path/to/custom_engine:/app/custom_engine your-image
```

---

## Best Practices for AI-Assisted Development

### Writing Effective Prompts

1. **Be specific** — Include exact requirements, constraints, and expected outputs
2. **Reference context** — Use `@` mentions to include relevant files and documents
3. **Break down complex tasks** — Divide large features into smaller, focused prompts
4. **Include examples** — Show expected input/output formats when applicable
5. **Specify hardware target** — Mention GPU availability so the agent generates GPU-optimized code

### Iterating on Generated Code

1. **Review before accepting** — Always inspect generated processors for correct metadata keys and pipeline integration
2. **Test incrementally** — Run the pipeline after each new processor rather than building everything at once
3. **Use the troubleshooting reference** — If a pipeline fails, ask the agent to consult `troubleshooting.md` for known error patterns
4. **Provide error output** — When debugging, paste the full error log into the chat for more accurate fixes

---

## Contributing

This project is currently not accepting contributions.

---

## License

This project is licensed under [Apache-2.0](LICENSE).

SPDX-License-Identifier: Apache-2.0
