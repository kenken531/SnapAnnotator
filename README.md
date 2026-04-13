# SnapAnnotator 📸

SnapAnnotator is a **live vision annotator**: press a key to capture a webcam frame, send it to a local vision-language model (moondream via ollama), and get back a scene description and a list of detected objects. Click any object label to ask a follow-up question about it — all offline, zero cloud. It's built for the **BUILDCORED ORCAS — Day 12** challenge.

## How it works

- Uses **OpenCV** to capture live webcam frames and display a real-time overlay.
- Press `SPACE` to freeze a frame and encode it as a base64 JPEG, resized to 512px for fast inference.
- Sends the image to **moondream via ollama** in a background thread — one query for the scene description, one for the object list — so the UI never blocks.
- Detected object names are rendered as **clickable labels** directly on the frame.
- Clicking any label sends a focused follow-up query about that specific object and displays the answer inline.
- Press `SPACE` again to return to live view and capture a new frame.

## Requirements

- Python 3.10.x
- [ollama](https://ollama.com/download) installed and running
- The moondream model pulled locally (see Setup)
- A working webcam

## Python packages:

```bash
pip install ollama opencv-python pillow numpy
```

## Setup

1. Download and install ollama from [ollama.com/download](https://ollama.com/download).
2. In a **separate terminal**, start the ollama server:
```
ollama serve
```
3. Pull the vision model (~800 MB):
```
ollama pull moondream
```
4. Install the Python packages (see above or run:
```
pip install -r requirements.txt
```
after downloading `requirements.txt`)

## Usage

From the project folder:

```bash
python snapannotator.py
```

| Key / Action | Effect |
|---|---|
| `SPACE` (live view) | Capture frame and send to model |
| `SPACE` (annotated view) | Return to live camera |
| Click an object label | Ask follow-up question about that object |
| `Q` or `ESC` | Quit |

- A coloured status badge shows `LIVE`, `CAPTURED`, `PROCESSING`, or `ANNOTATED`.
- Detected objects appear as clickable yellow-highlighted buttons when hovered.
- Follow-up answers appear as a panel at the bottom of the frame.

## Common fixes

**moondream not found** — run `ollama pull moondream` and wait for the ~800 MB download to complete.

**Very slow inference** — the frame is already resized to 512px before sending. If it's still slow, close other heavy applications. moondream needs ~2 GB free RAM.

**Model returns vague object lists** — occasionally moondream returns a paragraph instead of a numbered list. The parser handles bullets, commas, and plain lines as fallbacks — but rephrasing as a new capture usually helps.

**Audio output / no webcam** — check Windows Device Manager to confirm your webcam is detected, then try changing `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` at the bottom of the script.

**ollama not running** — open a separate terminal and run `ollama serve` before launching SnapAnnotator.

## Hardware concept

This project mirrors how a camera module feeds frames to an embedded vision processor — Jetson Nano, Raspberry Pi with Coral, or any edge AI accelerator. The pipeline is: **sensor → ISP → memory → inference → action**. The frame resize to 512px is the ISP step — it limits memory bandwidth and inference latency, the same tradeoff an embedded engineer makes when choosing resolution vs. frame rate on a constrained device.

## Credits

- Vision-language model: [moondream](https://ollama.com/library/moondream) via [ollama](https://ollama.com)
- Video capture & display: [OpenCV](https://opencv.org/)
- Image encoding: [Pillow](https://python-pillow.org/)

Built as part of the **BUILDCORED ORCAS — Day 12: SnapAnnotator** challenge.
