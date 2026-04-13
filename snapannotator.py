"""
SnapAnnotator  —  Windows Edition
====================================
Press SPACE to capture a webcam frame.
moondream (via ollama) returns a scene description and detected objects.
Click any object tag to ask a follow-up question.

Prerequisites:
    1. Install ollama:        https://ollama.com/download
    2. Start ollama server:   ollama serve   (separate terminal)
    3. Pull the model:        ollama pull moondream   (~800 MB)
    4. Install Python deps:   pip install ollama opencv-python pillow numpy

Usage:
    python snapannotator.py
    python snapannotator.py --model moondream
"""

import argparse
import base64
import re
import sys
import time
import threading
import io
import math
import numpy as np

try:
    import cv2
except ImportError:
    print("ERROR: opencv-python not installed. Run: pip install opencv-python")
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    print("ERROR: Pillow not installed. Run: pip install pillow")
    sys.exit(1)

try:
    import ollama
except ImportError:
    print("ERROR: ollama not installed. Run: pip install ollama")
    sys.exit(1)

# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_MODEL  = "moondream"
INFERENCE_SIZE = 512
WEBCAM_W       = 1280
WEBCAM_H       = 720

# Layout: camera feed on left, sidebar on right
SIDEBAR_W      = 360
WIN_W          = WEBCAM_W + SIDEBAR_W
WIN_H          = WEBCAM_H

# ── Colour palette (BGR) ─────────────────────────────────────────────────────

C = {
    # Sidebar background layers
    "sidebar_bg":    ( 14,  14,  22),
    "sidebar_card":  ( 22,  24,  36),
    "sidebar_border":( 45,  48,  72),

    # Accent colours
    "cyan":          (220, 200,   0),   # primary accent
    "green":         ( 60, 210,  80),   # success / live
    "yellow":        (  0, 210, 220),   # highlight / captured
    "red":           ( 70,  70, 220),   # error / stop
    "purple":        (200,  80, 180),   # follow-up

    # Text
    "text_primary":  (235, 235, 245),
    "text_secondary":(160, 165, 185),
    "text_dim":      ( 90,  95, 115),

    # Tag colours (for object chips)
    "tag_bg":        ( 35,  40,  60),
    "tag_border":    ( 70,  80, 120),
    "tag_hover_bg":  (  0, 180, 240),
    "tag_hover_txt": ( 10,  10,  15),

    # Camera overlay
    "cam_overlay":   (  0,   0,   0),
    "flash":         (255, 255, 255),
}

# ── Fonts ────────────────────────────────────────────────────────────────────

F  = cv2.FONT_HERSHEY_SIMPLEX
FB = cv2.FONT_HERSHEY_DUPLEX   # slightly bolder

# Terminal colours
TC = {
    "cyan":  "\033[96m", "green": "\033[92m", "yellow": "\033[93m",
    "red":   "\033[91m", "bold":  "\033[1m",  "dim":    "\033[2m",
    "reset": "\033[0m",
}
def tc(text, k): return f"{TC[k]}{text}{TC['reset']}"

# ── Drawing primitives ────────────────────────────────────────────────────────

def blend_rect(img, x, y, w, h, color, alpha=0.85):
    """Semi-transparent filled rectangle."""
    sub = img[y:y+h, x:x+w]
    if sub.shape[0] == 0 or sub.shape[1] == 0:
        return
    rect = np.full_like(sub, color)
    cv2.addWeighted(rect, alpha, sub, 1 - alpha, 0, sub)
    img[y:y+h, x:x+w] = sub


def rounded_rect(img, x, y, w, h, r, color, alpha=1.0, border=None, border_color=None):
    """
    Filled rounded rectangle drawn with 4 corner circles + 3 filled rects.
    Works on the image in-place.
    """
    if alpha < 1.0:
        overlay = img.copy()
        _draw_rounded_rect_solid(overlay, x, y, w, h, r, color)
        if border and border_color:
            _draw_rounded_rect_border(overlay, x, y, w, h, r, border_color, border)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    else:
        _draw_rounded_rect_solid(img, x, y, w, h, r, color)
        if border and border_color:
            _draw_rounded_rect_border(img, x, y, w, h, r, border_color, border)


def _draw_rounded_rect_solid(img, x, y, w, h, r, color):
    r = min(r, w // 2, h // 2)
    cv2.rectangle(img, (x + r, y),     (x + w - r, y + h),     color, -1)
    cv2.rectangle(img, (x,     y + r), (x + w,     y + h - r), color, -1)
    cv2.circle(img, (x + r,         y + r),         r, color, -1)
    cv2.circle(img, (x + w - r,     y + r),         r, color, -1)
    cv2.circle(img, (x + r,         y + h - r),     r, color, -1)
    cv2.circle(img, (x + w - r,     y + h - r),     r, color, -1)


def _draw_rounded_rect_border(img, x, y, w, h, r, color, thick):
    r = min(r, w // 2, h // 2)
    cv2.line(img,  (x + r, y),         (x + w - r, y),         color, thick)
    cv2.line(img,  (x + r, y + h),     (x + w - r, y + h),     color, thick)
    cv2.line(img,  (x,     y + r),     (x,         y + h - r), color, thick)
    cv2.line(img,  (x + w, y + r),     (x + w,     y + h - r), color, thick)
    cv2.ellipse(img, (x + r,     y + r),     (r, r), 180,  0, 90, color, thick)
    cv2.ellipse(img, (x + w - r, y + r),     (r, r), 270,  0, 90, color, thick)
    cv2.ellipse(img, (x + r,     y + h - r), (r, r),  90,  0, 90, color, thick)
    cv2.ellipse(img, (x + w - r, y + h - r), (r, r),   0,  0, 90, color, thick)


def put(img, text, x, y, color, scale=0.5, thick=1, font=F):
    cv2.putText(img, str(text), (x, y), font, scale, color, thick, cv2.LINE_AA)


def wrap_text(text: str, max_chars: int, max_lines: int) -> list:
    lines, line = [], ""
    for word in text.split():
        candidate = (line + " " + word).strip()
        if len(candidate) > max_chars:
            if line:
                lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)
    return lines[:max_lines]

# ── Spinner ───────────────────────────────────────────────────────────────────

def draw_spinner(img, cx, cy, radius, t, color, thick=2):
    """Animated arc spinner. t = time.time()."""
    angle = (t * 280) % 360
    start = int(angle)
    end   = int(angle + 260) % 360
    cv2.ellipse(img, (cx, cy), (radius, radius), 0, start, start + 240,
                color, thick, cv2.LINE_AA)

# ── Object tag layout ─────────────────────────────────────────────────────────

TAG_H   = 30
TAG_PAD = 8
TAG_GAP = 6
TAG_ROW_GAP = 6

def compute_tag_layout(objects: list, sidebar_x: int, start_y: int,
                        available_w: int) -> list:
    """
    Flow-layout: pack tags left-to-right, wrapping to next line.
    Returns list of (idx, x, y, w, h) rects.
    """
    rects  = []
    cx     = sidebar_x + 12
    cy     = start_y
    row_h  = TAG_H

    for idx, obj in enumerate(objects):
        label = obj[:20]
        (tw, _), _ = cv2.getTextSize(label, F, 0.42, 1)
        tag_w = tw + TAG_PAD * 2 + 2

        if cx + tag_w > sidebar_x + available_w - 12:
            cx  = sidebar_x + 12
            cy += row_h + TAG_ROW_GAP

        rects.append((idx, cx, cy, tag_w, TAG_H))
        cx += tag_w + TAG_GAP

    return rects


def hit_test_tags(mx: int, my: int, tag_rects: list) -> int:
    for idx, tx, ty, tw, th in tag_rects:
        if tx <= mx <= tx + tw and ty <= my <= ty + th:
            return idx
    return -1

# ── Parse objects ─────────────────────────────────────────────────────────────

def parse_objects(text: str) -> list:
    objects = []
    numbered = re.findall(r'\d+\.\s*([^\n,\.]+)', text)
    if numbered:
        objects = [o.strip() for o in numbered if o.strip()]
    if not objects:
        bulleted = re.findall(r'[-•*]\s*([^\n]+)', text)
        if bulleted:
            objects = [o.strip() for o in bulleted if o.strip()]
    if not objects:
        for line in text.splitlines():
            line_c = re.sub(r'^(I can see|objects?|things?|items?)\s*[:;]\s*',
                            '', line, flags=re.IGNORECASE)
            if ',' in line_c and len(line_c) < 200:
                parts = [p.strip().rstrip('.') for p in line_c.split(',')]
                if all(len(p) < 40 for p in parts):
                    objects = [p for p in parts if p]
                    break
    if not objects:
        for line in text.splitlines():
            line = line.strip().rstrip('.')
            if 2 < len(line) < 50 and not line.startswith(
                    ('The ', 'This ', 'A ', 'An ', 'I ')):
                objects.append(line)
    seen, clean = set(), []
    for obj in objects:
        obj = obj.strip().rstrip('.').lower()
        if obj and obj not in seen and len(obj) > 1:
            seen.add(obj); clean.append(obj)
    return clean[:12]

# ── LLM queries ───────────────────────────────────────────────────────────────

def frame_to_base64(frame: np.ndarray, max_size: int = INFERENCE_SIZE) -> str:
    h, w   = frame.shape[:2]
    scale  = max_size / max(h, w)
    if scale < 1.0:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def query_describe(model, b64):
    print(f"  {tc('Describing scene...', 'dim')}", end="", flush=True)
    t0 = time.time()
    try:
        r = ollama.chat(model=model, messages=[{
            "role": "user",
            "content": "Describe what you see in this image in 2-3 sentences.",
            "images": [b64],
        }])
        desc = r["message"]["content"].strip()
    except Exception as e:
        desc = f"[Error: {e}]"
    print(f" {tc(f'{time.time()-t0:.1f}s', 'dim')}")

    print(f"  {tc('Detecting objects...', 'dim')}", end="", flush=True)
    t1 = time.time()
    try:
        r2 = ollama.chat(model=model, messages=[{
            "role": "user",
            "content": (
                "List every distinct object you can see in this image. "
		"Return only the names of the objects."
            ),
            "images": [b64],
        }])
        objs = parse_objects(r2["message"]["content"].strip())
    except Exception:
        objs = []
    print(f" {tc(f'{time.time()-t1:.1f}s', 'dim')}")
    print(f"  {tc(f'Found {len(objs)} object(s)', 'green')}")
    return desc, objs


def query_followup(model, b64, obj):
    print(f"\n  {tc(f'Follow-up: {obj}...', 'dim')}", end="", flush=True)
    t0 = time.time()
    try:
        r = ollama.chat(model=model, messages=[{
            "role": "user",
            "content": (
                f"Focus only on the '{obj}' in this image. "
                f"Describe its appearance, color, condition, and any notable features "
                f"in 2-3 sentences."
            ),
            "images": [b64],
        }])
        ans = r["message"]["content"].strip()
    except Exception as e:
        ans = f"[Error: {e}]"
    print(f" {tc(f'{time.time()-t0:.1f}s', 'dim')}")
    return ans

# ── Full UI renderer ──────────────────────────────────────────────────────────

def draw_ui(cam_frame: np.ndarray, st: dict,
            tag_rects: list, hover_idx: int, t: float) -> np.ndarray:
    """
    Build the full composite frame:
      Left  (WEBCAM_W × WIN_H): camera feed with minimal overlay
      Right (SIDEBAR_W × WIN_H): dark sidebar with all text panels
    """
    h, w = WIN_H, WIN_W
    sx   = WEBCAM_W   # sidebar starts here

    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    # ── Camera area ──────────────────────────────────────────────────────────
    cam_h, cam_w = cam_frame.shape[:2]
    # Fit camera to left region
    scale  = min(WEBCAM_W / cam_w, WIN_H / cam_h)
    dw, dh = int(cam_w * scale), int(cam_h * scale)
    ox     = (WEBCAM_W - dw) // 2
    oy     = (WIN_H   - dh) // 2
    resized = cv2.resize(cam_frame, (dw, dh), interpolation=cv2.INTER_LINEAR)
    canvas[oy:oy+dh, ox:ox+dw] = resized

    # Flash effect on capture
    if st.get("flash", 0) > 0:
        fade = st["flash"]
        blend_rect(canvas, 0, 0, WEBCAM_W, WIN_H, C["flash"], alpha=fade * 0.9)
        st["flash"] = max(0.0, fade - 0.12)

    # Live indicator (pulsing dot top-left of camera)
    if st["state"] == "live":
        pulse = 0.5 + 0.5 * math.sin(t * 4)
        dot_color = tuple(int(c * pulse) for c in C["green"])
        cv2.circle(canvas, (18, 18), 7, dot_color, -1, cv2.LINE_AA)
        put(canvas, "LIVE", 30, 23, C["green"], 0.45, 1)

    # "Tap SPACE" hint on camera when live
    if st["state"] == "live":
        hint = "Press  SPACE  to analyse this scene"
        (tw, _), _ = cv2.getTextSize(hint, F, 0.5, 1)
        hx = (WEBCAM_W - tw) // 2
        blend_rect(canvas, hx - 12, WIN_H - 46, tw + 24, 30,
                   C["cam_overlay"], alpha=0.55)
        put(canvas, hint, hx, WIN_H - 26, C["text_secondary"], 0.5)

    # Hairline separator
    cv2.line(canvas, (sx - 1, 0), (sx - 1, WIN_H), C["sidebar_border"], 1)

    # ── Sidebar background ────────────────────────────────────────────────────
    canvas[:, sx:] = C["sidebar_bg"]

    # ── Header ────────────────────────────────────────────────────────────────
    rounded_rect(canvas, sx, 0, SIDEBAR_W, 56, 0,
                 C["sidebar_card"], alpha=1.0,
                 border=1, border_color=C["sidebar_border"])

    put(canvas, "SnapAnnotator", sx + 14, 28, C["cyan"], 0.65, 2, FB)
    put(canvas, "vision analysis", sx + 16, 46, C["text_dim"], 0.38)

    # Model badge
    model_lbl = st.get("model_name", DEFAULT_MODEL)
    (mw, _), _ = cv2.getTextSize(model_lbl, F, 0.38, 1)
    rounded_rect(canvas, sx + SIDEBAR_W - mw - 24, 16, mw + 16, 22, 4,
                 C["sidebar_border"], alpha=1.0)
    put(canvas, model_lbl, sx + SIDEBAR_W - mw - 16, 32,
        C["text_secondary"], 0.38)

    # ── State badge ───────────────────────────────────────────────────────────
    cy_cur = 66
    state_map = {
        "live":       ("● LIVE",        C["green"]),
        "captured":   ("■ CAPTURED",    C["yellow"]),
        "annotated":  ("✓ ANNOTATED",   C["cyan"]),
    }
    if st["processing"]:
        badge_txt, badge_col = "⟳ ANALYSING", C["yellow"]
    else:
        badge_txt, badge_col = state_map.get(st["state"], ("?", C["text_dim"]))

    (bw, _), _ = cv2.getTextSize(badge_txt, FB, 0.48, 1)
    rounded_rect(canvas, sx + 12, cy_cur, bw + 20, 26, 6,
                 badge_col, alpha=0.18)
    rounded_rect(canvas, sx + 12, cy_cur, bw + 20, 26, 6,
                 (0,0,0), alpha=0.0, border=1, border_color=badge_col)
    put(canvas, badge_txt, sx + 22, cy_cur + 17, badge_col, 0.48, 1, FB)
    cy_cur += 36

    # ── Divider ───────────────────────────────────────────────────────────────
    def divider(y):
        cv2.line(canvas, (sx + 12, y), (sx + SIDEBAR_W - 12, y),
                 C["sidebar_border"], 1)

    divider(cy_cur)
    cy_cur += 12

    # ── Description card ──────────────────────────────────────────────────────
    if st["description"]:
        put(canvas, "SCENE", sx + 14, cy_cur + 12,
            C["text_dim"], 0.38, 1)
        cy_cur += 18

        desc_lines = wrap_text(st["description"], 40, 5)
        card_h     = 10 + len(desc_lines) * 19 + 8
        rounded_rect(canvas, sx + 10, cy_cur, SIDEBAR_W - 20, card_h, 8,
                     C["sidebar_card"], alpha=1.0,
                     border=1, border_color=C["sidebar_border"])
        for i, line in enumerate(desc_lines):
            put(canvas, line, sx + 18, cy_cur + 16 + i * 19,
                C["text_primary"], 0.44)
        cy_cur += card_h + 12

    elif st["processing"] and st["state"] == "captured":
        # Spinner card while waiting
        card_h = 60
        rounded_rect(canvas, sx + 10, cy_cur, SIDEBAR_W - 20, card_h, 8,
                     C["sidebar_card"], alpha=1.0,
                     border=1, border_color=C["sidebar_border"])
        scx = sx + SIDEBAR_W // 2
        scy = cy_cur + card_h // 2
        draw_spinner(canvas, scx, scy, 16, t, C["cyan"], 2)
        put(canvas, "Analysing scene...", scx - 66, scy + 5,
            C["text_secondary"], 0.44)
        cy_cur += card_h + 12

    else:
        # Idle placeholder
        card_h = 50
        rounded_rect(canvas, sx + 10, cy_cur, SIDEBAR_W - 20, card_h, 8,
                     C["sidebar_card"], alpha=1.0,
                     border=1, border_color=C["sidebar_border"])
        put(canvas, "No scene captured yet.", sx + 18, cy_cur + 20,
            C["text_dim"], 0.44)
        put(canvas, "Press SPACE to begin.", sx + 18, cy_cur + 38,
            C["text_dim"], 0.4)
        cy_cur += card_h + 12

    divider(cy_cur)
    cy_cur += 12

    # ── Objects section ───────────────────────────────────────────────────────
    put(canvas, "OBJECTS", sx + 14, cy_cur + 12,
        C["text_dim"], 0.38)
    cy_cur += 18

    if st["objects"]:
        # Render flow-layout tags
        for idx, tx, ty, tw_tag, th_tag in tag_rects:
            is_hover = (idx == hover_idx)
            bg       = C["tag_hover_bg"]  if is_hover else C["tag_bg"]
            border   = C["cyan"]          if is_hover else C["tag_border"]
            txt_col  = C["tag_hover_txt"] if is_hover else C["text_primary"]

            rounded_rect(canvas, tx, ty, tw_tag, th_tag, 6,
                         bg, alpha=1.0, border=1, border_color=border)
            lbl = st["objects"][idx][:20]
            put(canvas, lbl, tx + TAG_PAD, ty + 20, txt_col, 0.42)

        # Advance cy_cur past tags
        if tag_rects:
            last_ty = max(r[2] for r in tag_rects)
            cy_cur  = last_ty + TAG_H + TAG_ROW_GAP + 10

        # Hint
        put(canvas, "Click a tag to ask about it",
            sx + 14, cy_cur, C["text_dim"], 0.38)
        cy_cur += 18

    elif st["processing"]:
        put(canvas, "Detecting...", sx + 14, cy_cur + 14,
            C["text_dim"], 0.42)
        cy_cur += 28

    else:
        put(canvas, "No objects detected yet.", sx + 14, cy_cur + 14,
            C["text_dim"], 0.42)
        cy_cur += 28

    divider(cy_cur)
    cy_cur += 12

    # ── Follow-up panel ───────────────────────────────────────────────────────
    put(canvas, "FOLLOW-UP", sx + 14, cy_cur + 12,
        C["text_dim"], 0.38)
    cy_cur += 18

    if st["followup_answer"]:
        obj_badge = st["followup_obj"]
        (obw, _), _ = cv2.getTextSize(obj_badge, F, 0.42, 1)
        rounded_rect(canvas, sx + 12, cy_cur, obw + 20, 24, 5,
                     C["purple"], alpha=0.25,
                     border=1, border_color=C["purple"])
        put(canvas, obj_badge, sx + 22, cy_cur + 16,
            C["purple"], 0.42)
        cy_cur += 30

        ans_lines = wrap_text(st["followup_answer"], 40, 6)
        card_h    = 10 + len(ans_lines) * 19 + 8
        rounded_rect(canvas, sx + 10, cy_cur, SIDEBAR_W - 20, card_h, 8,
                     C["sidebar_card"], alpha=1.0,
                     border=1, border_color=C["purple"])
        for i, aline in enumerate(ans_lines):
            put(canvas, aline, sx + 18, cy_cur + 16 + i * 19,
                C["text_primary"], 0.44)
        cy_cur += card_h + 10

    elif st["processing"] and st["followup_obj"]:
        card_h = 50
        rounded_rect(canvas, sx + 10, cy_cur, SIDEBAR_W - 20, card_h, 8,
                     C["sidebar_card"], alpha=1.0,
                     border=1, border_color=C["purple"])
        scx = sx + SIDEBAR_W // 2
        scy = cy_cur + card_h // 2
        draw_spinner(canvas, scx, scy, 14, t, C["purple"], 2)
        lbl = f"Asking about '{st['followup_obj']}'..."
        put(canvas, lbl, sx + 18, scy + 5, C["text_secondary"], 0.42)
        cy_cur += card_h + 10
    else:
        put(canvas, "Click an object tag above.",
            sx + 14, cy_cur + 14, C["text_dim"], 0.42)
        cy_cur += 28

    # ── Controls footer ───────────────────────────────────────────────────────
    footer_y = WIN_H - 50
    cv2.line(canvas, (sx, footer_y), (sx + SIDEBAR_W, footer_y),
             C["sidebar_border"], 1)
    canvas[footer_y:, sx:] = C["sidebar_card"]

    controls = [
        ("SPACE", "capture / reset"),
        ("Q / ESC", "quit"),
    ]
    fx = sx + 14
    for key_lbl, key_desc in controls:
        (kw, _), _ = cv2.getTextSize(key_lbl, FB, 0.4, 1)
        rounded_rect(canvas, fx, footer_y + 10, kw + 12, 22, 4,
                     C["sidebar_border"], alpha=1.0)
        put(canvas, key_lbl, fx + 6, footer_y + 25, C["text_primary"], 0.4, 1, FB)
        fx += kw + 18
        put(canvas, key_desc, fx, footer_y + 25, C["text_dim"], 0.38)
        (dw2, _), _ = cv2.getTextSize(key_desc, F, 0.38, 1)
        fx += dw2 + 20

    return canvas

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SnapAnnotator — webcam + local vision model"
    )
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL,
                        help=f"Ollama vision model (default: {DEFAULT_MODEL})")
    args = parser.parse_args()

    # Verify ollama
    print(f"\n  {tc('Connecting to ollama...', 'dim')}", end="", flush=True)
    try:
        models    = ollama.list()
        available = [m["model"] for m in models.get("models", [])]
        base      = args.model.split(":")[0]
        if not any(base in m for m in available):
            print(f"\n\n  {tc('WARNING:', 'yellow')} Model '{args.model}' not found.")
            print(f"  Run: {tc(f'ollama pull {args.model}', 'green')}\n")
        else:
            print(f" {tc('OK', 'green')}")
    except Exception as e:
        print(f"\n\n  {tc('ERROR:', 'red')} Cannot reach ollama.")
        print(f"  Run {tc('ollama serve', 'green')} in a separate terminal.\n  {e}\n")
        sys.exit(1)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(f"\n  {tc('ERROR:', 'red')} Cannot open webcam.\n")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WEBCAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_H)
    print(f"  {tc('Webcam open.', 'green')} Press SPACE to capture. Q to quit.\n")

    # State dict
    st = {
        "state":          "live",
        "captured_frame": None,
        "b64_image":      None,
        "description":    "",
        "objects":        [],
        "followup_obj":   "",
        "followup_answer":"",
        "processing":     False,
        "flash":          0.0,
        "model_name":     args.model,
    }

    llm_lock   = threading.Lock()
    llm_result = {}
    mouse_pos  = [0, 0]
    tag_rects  = []   # recomputed each frame when objects exist

    def run_description(b64):
        desc, objs = query_describe(args.model, b64)
        with llm_lock:
            llm_result["description"] = desc
            llm_result["objects"]     = objs
            llm_result["desc_done"]   = True

    def run_followup(b64, obj):
        ans = query_followup(args.model, b64, obj)
        with llm_lock:
            llm_result["followup_obj"]    = obj
            llm_result["followup_answer"] = ans
            llm_result["followup_done"]   = True

    def on_mouse(event, x, y, flags, param):
        mouse_pos[0] = x
        mouse_pos[1] = y
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if st["state"] != "annotated" or not st["objects"] or st["processing"]:
            return
        idx = hit_test_tags(x, y, tag_rects)
        if idx < 0:
            return
        st["followup_obj"]    = st["objects"][idx]
        st["followup_answer"] = ""
        st["processing"]      = True
        with llm_lock:
            llm_result["followup_done"] = False
        threading.Thread(
            target=run_followup,
            args=(st["b64_image"], st["objects"][idx]),
            daemon=True,
        ).start()

    cv2.namedWindow("SnapAnnotator", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("SnapAnnotator", WIN_W, WIN_H)
    cv2.setMouseCallback("SnapAnnotator", on_mouse)

    while True:
        t_now = time.time()

        # Get camera frame
        if st["state"] == "live":
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            cam_frame = frame
        else:
            cam_frame = st["captured_frame"]

        # Poll LLM results
        with llm_lock:
            if llm_result.get("desc_done"):
                st["description"] = llm_result.pop("description", "")
                st["objects"]     = llm_result.pop("objects", [])
                llm_result.pop("desc_done")
                st["processing"]  = False
                st["state"]       = "annotated"

            if llm_result.get("followup_done"):
                st["followup_obj"]    = llm_result.pop("followup_obj", "")
                st["followup_answer"] = llm_result.pop("followup_answer", "")
                llm_result.pop("followup_done")
                st["processing"]      = False

        # Recompute tag rects (needed for hit test and rendering)
        if st["objects"]:
            tag_rects = compute_tag_layout(
                st["objects"], WEBCAM_W, 0, SIDEBAR_W
            )
            # Recompute with correct start_y based on current sidebar layout
            # Approximate: header(56) + badge(36) + div(24) + scene_label(18) +
            # desc_card + div + objects_label = ~200 + desc height
            desc_lines = len(wrap_text(st["description"], 40, 5)) if st["description"] else 0
            desc_card_h = 10 + desc_lines * 19 + 8 + 12 if desc_lines else 62 + 12
            obj_y = 56 + 36 + 24 + 18 + desc_card_h + 12 + 18
            tag_rects = compute_tag_layout(
                st["objects"], WEBCAM_W, obj_y, SIDEBAR_W
            )
        else:
            tag_rects = []

        # Hover
        hover_idx = hit_test_tags(mouse_pos[0], mouse_pos[1], tag_rects)

        # Render
        frame_out = draw_ui(cam_frame, st, tag_rects, hover_idx, t_now)
        cv2.imshow("SnapAnnotator", frame_out)

        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            break

        elif key == ord(" "):
            if st["state"] == "live":
                ret, frame = cap.read()
                if ret:
                    frame = cv2.flip(frame, 1)
                    st["captured_frame"]  = frame.copy()
                    st["b64_image"]       = frame_to_base64(frame)
                    st["state"]           = "captured"
                    st["description"]     = ""
                    st["objects"]         = []
                    st["followup_obj"]    = ""
                    st["followup_answer"] = ""
                    st["processing"]      = True
                    st["flash"]           = 1.0
                    with llm_lock:
                        llm_result.clear()
                    print(f"\n  {tc('Frame captured!', 'green')} Sending to model...")
                    threading.Thread(
                        target=run_description,
                        args=(st["b64_image"],),
                        daemon=True,
                    ).start()
            else:
                st["state"]           = "live"
                st["description"]     = ""
                st["objects"]         = []
                st["followup_obj"]    = ""
                st["followup_answer"] = ""
                st["processing"]      = False
                tag_rects             = []
                with llm_lock:
                    llm_result.clear()

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n  {tc('SnapAnnotator stopped.', 'red')}\n")


if __name__ == "__main__":
    main()