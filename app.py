from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from sklearn.cluster import KMeans
import colorsys
import os
import uuid
import base64
import io

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# ----------------------------
# Skin Tone Classification
# ----------------------------
def classify_skin_tone(avg_bgr):
    brightness = np.mean(avg_bgr)
    if brightness > 200:
        return "Fair"
    elif brightness > 170:
        return "Medium"
    elif brightness > 140:
        return "Wheatish"
    else:
        return "Dark"

# ----------------------------
# Extract Dominant Color (Deterministic)
# ----------------------------
def get_dominant_color(image):
    # Resize for consistency
    img_resized = cv2.resize(image, (100, 100))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    pixels = img_rgb.reshape(-1, 3)

    # Make KMeans deterministic
    kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
    kmeans.fit(pixels)
    dominant = kmeans.cluster_centers_[0]
    return dominant

# ----------------------------
# Convert RGB to Hue
# ----------------------------
def rgb_to_hue(color):
    r, g, b = color / 255.0
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    return h * 360

# ----------------------------
# Compatibility Logic
# ----------------------------
def evaluate_match(skin_hue, outfit_hue):
    diff = abs(skin_hue - outfit_hue)
    if diff > 180:
        diff = 360 - diff

    # Determine match level
    if diff <= 30 or 150 <= diff <= 210:
        level = "Match"
    elif diff <= 60:
        level = "Partial Match"
    else:
        level = "Not a Match"

    score = int(max(40, 100 - (diff / 180 * 100)))
    return level, score

# ----------------------------
# Routes
# ----------------------------
@app.route("/")
def home():
    return render_template("index.html")

# ----------------------------
# Helper: save an uploaded file
# (handles regular uploads AND camera-captured blobs)
# ----------------------------
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def save_upload(file_storage, label):
    """
    Save a werkzeug FileStorage to disk.
    Camera captures may arrive with a generic name like 'blob' or
    'face-capture.jpg', so we always generate a unique filename.
    Returns the saved file path or None on failure.
    """
    # Determine extension from original filename or default to .jpg
    orig = secure_filename(file_storage.filename or "")
    ext = orig.rsplit(".", 1)[1].lower() if "." in orig else "jpg"
    if ext not in ALLOWED_EXTENSIONS:
        ext = "jpg"

    unique_name = f"{label}_{uuid.uuid4().hex[:8]}.{ext}"
    save_path = os.path.join(UPLOAD_FOLDER, unique_name)
    file_storage.save(save_path)
    return save_path


# ----------------------------
# Helper: decode a base64 image string
# ----------------------------
def save_base64_image(data_uri, label):
    """
    Accept a base64 data URI (e.g. data:image/jpeg;base64,...) or raw base64
    string, decode it, and save to disk. Returns the file path.
    """
    try:
        if "," in data_uri:
            header, encoded = data_uri.split(",", 1)
        else:
            encoded = data_uri

        img_bytes = base64.b64decode(encoded)
        unique_name = f"{label}_{uuid.uuid4().hex[:8]}.jpg"
        save_path = os.path.join(UPLOAD_FOLDER, unique_name)
        with open(save_path, "wb") as f:
            f.write(img_bytes)
        return save_path
    except Exception:
        return None


# ----------------------------
# Helper: get image from request (file upload OR base64)
# ----------------------------
def get_image_from_request(field_name):
    """
    Try to get an image from either:
      1. A multipart file upload  (request.files[field_name])
      2. A base64 string          (request.form[field_name] or JSON body)
    Returns (cv2_image, file_path) or (None, None) on failure.
    """
    # --- Option 1: multipart file upload ---
    if field_name in request.files:
        f = request.files[field_name]
        if f and f.filename:
            path = save_upload(f, field_name)
            if path:
                img = cv2.imread(path)
                if img is not None:
                    return img, path

    # --- Option 2: base64 in form data ---
    b64 = request.form.get(f"{field_name}_base64") or request.form.get(field_name)
    if b64 and ("base64" in b64 or len(b64) > 500):
        path = save_base64_image(b64, field_name)
        if path:
            img = cv2.imread(path)
            if img is not None:
                return img, path

    # --- Option 3: JSON body with base64 ---
    if request.is_json:
        data = request.get_json(silent=True)
        if data and field_name in data:
            path = save_base64_image(data[field_name], field_name)
            if path:
                img = cv2.imread(path)
                if img is not None:
                    return img, path

    return None, None


@app.route("/analyze", methods=["POST"])
def analyze():
    face_img, face_path = get_image_from_request("face")
    outfit_img, outfit_path = get_image_from_request("outfit")

    if face_img is None or outfit_img is None:
        missing = []
        if face_img is None:
            missing.append("face")
        if outfit_img is None:
            missing.append("outfit")
        return jsonify({"error": f"Could not read image(s): {', '.join(missing)}. Please upload or capture valid photos."})

    # Face detection
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return jsonify({"error": "No face detected. Please use a clear, front-facing photo."})

    x, y, w, h = faces[0]
    face_region = face_img[y:y+h, x:x+w]

    # Extract dominant colors (deterministic)
    avg_skin_color = get_dominant_color(face_region)
    skin_tone = classify_skin_tone(avg_skin_color)

    outfit_color = get_dominant_color(outfit_img)

    skin_hue = rgb_to_hue(avg_skin_color)
    outfit_hue = rgb_to_hue(outfit_color)

    result, score = evaluate_match(skin_hue, outfit_hue)

    # Clean up uploaded files after processing
    for p in [face_path, outfit_path]:
        try:
            if p and os.path.exists(p):
                os.remove(p)
        except OSError:
            pass

    return jsonify({
        "skin_tone": skin_tone,
        "result": result,
        "score": score,
        "skin_color": avg_skin_color.tolist(),
        "outfit_color": outfit_color.tolist()
    })

if __name__ == "__main__":
    app.run(debug=True)