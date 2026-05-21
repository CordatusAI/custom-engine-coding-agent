# Troubleshooting

## Import Errors

### `ModuleNotFoundError: No module named 'base_processor'`

**Cause:** Incorrect import path.

**Fix:** Processors in `custom_engine/` should import using:

```python
from base_processor import BaseProcessor
```

NOT:

```python
from custom_engine.base_processor import BaseProcessor  # WRONG inside Docker
```

Inside the Docker container, `custom_engine/` is the working directory for imports.

### `ModuleNotFoundError: No module named 'onnxruntime'`

**Fix:** Install inside the Docker container:

```bash
pip install onnxruntime-gpu  # if GPU available
# or
pip install onnxruntime      # CPU only
```

### `ModuleNotFoundError: No module named 'cv2'`

**Fix:** Install OpenCV:

```bash
pip install opencv-python-headless
```

## GPU Issues

### GPU not detected / falling back to CPU

**Diagnosis:**

```python
import onnxruntime as ort
print(ort.get_available_providers())
# Expected: ['CUDAExecutionProvider', 'CPUExecutionProvider']
# If only CPUExecutionProvider: GPU not available
```

**Common causes:**
1. `onnxruntime-gpu` not installed (only `onnxruntime`)
2. CUDA/cuDNN mismatch
3. NVIDIA driver not installed in container

**Fix:** Ensure Docker container has NVIDIA runtime (`--gpus all`) and matching CUDA versions.

### `onnxruntime` CUDA error

**Cause:** `onnxruntime-gpu` version incompatible with installed CUDA.

**Fix:** Match versions:

| onnxruntime-gpu | CUDA |
|-----------------|------|
| 1.17+ | 12.x |
| 1.16 | 11.8 |

## Frame Format Issues

### Colors look wrong (blue faces, etc.)

**Cause:** Frame is BGR, code assumed RGB.

**Fix:** Convert when needed:

```python
rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# process...
frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
```

### Frame shape mismatch

**Expected:** `(H, W, 3)`, dtype `uint8`

**Diagnosis:**

```python
print(frame.shape, frame.dtype)
```

## TypeError on add_processor

### `TypeError: must be a BaseProcessor instance`

**Cause:** Passed a class instead of an instance, or wrong type.

**Fix:**

```python
# WRONG
engine.add_processor(YOLODetector)

# CORRECT
engine.add_processor(YOLODetector(model="yolov8s"))
```