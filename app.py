import os
import io
import base64
import tempfile
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

# ── App setup ────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024   # 200 MB

ALLOWED_IMAGE = {"jpg", "jpeg", "png", "webp", "bmp"}
ALLOWED_VIDEO = {"mp4", "avi", "mov", "mkv"}

# ── Model ─────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
model_path = os.path.join(os.path.dirname(__file__), "best_model.pth")
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
model = model.to(device)
model.eval()

print(f"[OK] Model loaded on {device}")

# ── Transforms ────────────────────────────────────────────────────────────────
inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
])

# ── Helpers ───────────────────────────────────────────────────────────────────
def predict_image(pil_image: Image.Image, threshold: float = 0.5) -> dict:
    """Run inference on a single PIL image (with TTA)."""
    pil_image = pil_image.convert("RGB")
    tensor = inference_transform(pil_image).unsqueeze(0).to(device)
    flipped = torch.flip(tensor, dims=[3])

    with torch.no_grad():
        prob1 = torch.sigmoid(model(tensor)).item()
        prob2 = torch.sigmoid(model(flipped)).item()

    prob = (prob1 + prob2) / 2.0
    label = "FAKE" if prob > threshold else "REAL"
    confidence = prob if label == "FAKE" else (1 - prob)

    return {"label": label, "fake_prob": round(prob, 4), "confidence": round(confidence, 4)}


def predict_video(video_path: str, threshold: float = 0.5, frame_skip: int = 15, max_frames: int = 30) -> dict:
    """Sample frames from a video and aggregate predictions."""
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    probs = []
    sampled_frames_b64 = []   # store a few preview frames

    while cap.isOpened() and len(probs) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            result = predict_image(pil, threshold=0.0)   # raw prob only
            probs.append(result["fake_prob"])

            # Save small preview (first 5)
            if len(sampled_frames_b64) < 5:
                thumb = pil.resize((160, 90))
                buf = io.BytesIO()
                thumb.save(buf, format="JPEG", quality=70)
                sampled_frames_b64.append(
                    "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
                )

        frame_count += 1

    cap.release()

    if not probs:
        return {"error": "Could not extract frames from video"}

    final_prob = float(np.mean(probs))
    label = "FAKE" if final_prob > threshold else "REAL"
    confidence = final_prob if label == "FAKE" else (1 - final_prob)

    return {
        "label": label,
        "fake_prob": round(final_prob, 4),
        "confidence": round(confidence, 4),
        "frames_analyzed": len(probs),
        "frame_previews": sampled_frames_b64,
    }

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("templates", "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    threshold = float(request.form.get("threshold", 0.5))
    ext = file.filename.rsplit(".", 1)[-1].lower()

    try:
        if ext in ALLOWED_IMAGE:
            pil = Image.open(file.stream).convert("RGB")
            # Return preview as base64
            thumb = pil.copy()
            thumb.thumbnail((400, 400))
            buf = io.BytesIO()
            thumb.save(buf, format="JPEG", quality=85)
            preview = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

            result = predict_image(pil, threshold=threshold)
            result["preview"] = preview
            result["type"] = "image"
            return jsonify(result)

        elif ext in ALLOWED_VIDEO:
            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
                file.save(tmp.name)
                tmp_path = tmp.name

            frame_skip = int(request.form.get("frame_skip", 15))
            max_frames = int(request.form.get("max_frames", 30))
            result = predict_video(tmp_path, threshold=threshold,
                                   frame_skip=frame_skip, max_frames=max_frames)
            os.unlink(tmp_path)
            result["type"] = "video"
            return jsonify(result)

        else:
            return jsonify({"error": f"Unsupported file type: .{ext}"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok", "device": str(device)})


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
