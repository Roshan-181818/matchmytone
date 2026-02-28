from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from sklearn.cluster import KMeans
import colorsys
import os
import uuid
import base64

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


# ----------------------------
# Skin Tone Classification using Perceptual Luminance Y (ITU-R BT.601)
# ----------------------------
def classify_skin_tone(avg_rgb):
    """
    Y = 0.299*R + 0.587*G + 0.114*B  (perceptual brightness, 0-255)
    Much more stable than HSV value under varying camera exposure.

    Thresholds (calibrated for South Asian / warm skin tones):
      Fair:     Y > 180
      Medium:   Y > 145
      Wheatish: Y > 110   <-- most warm/South Asian skin lands here
      Dark:     Y <= 110
    """
    r, g, b = float(avg_rgb[0]), float(avg_rgb[1]), float(avg_rgb[2])
    Y = 0.299 * r + 0.587 * g + 0.114 * b

    if Y > 155:
        return "Fair"
    elif Y > 128:
        return "Medium"
    elif Y > 100:
        return "Wheatish"
    else:
        return "Dark"


# ----------------------------
# Extract Skin Pixels using YCrCb (widened range for all skin tones)
# ----------------------------
def get_skin_color(face_img):
    """
    Isolate skin pixels via YCrCb with a wider Cr range (128-185)
    that covers warm, South Asian, olive, and darker skin.
    Previous range Cr 133-173 missed warm/South Asian tones (Cr ~175-185).
    """
    ycrcb = cv2.cvtColor(face_img, cv2.COLOR_BGR2YCrCb)

    lower = np.array([0,   128, 70],  dtype=np.uint8)  # Y, Cr, Cb
    upper = np.array([255, 185, 135], dtype=np.uint8)
    skin_mask = cv2.inRange(ycrcb, lower, upper)


    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_DILATE, kernel)

    skin_pixels = face_img[skin_mask > 0]

    if len(skin_pixels) < 50:
        # Fallback: center crop (avoids hair/background edges)
        h, w = face_img.shape[:2]
        cy, cx = h // 2, w // 2
        patch = face_img[cy - h // 4: cy + h // 4, cx - w // 4: cx + w // 4]
        skin_pixels = patch.reshape(-1, 3)

    # Instead of averaging ALL skin pixels (which includes beard shadow, neck, ears),
    # use the cheek/mid-face region which is the most representative skin area.
    # Cheek zone: vertically 35%-65% of face, horizontally 15%-85%
    fh, fw = face_img.shape[:2]
    cheek = face_img[int(fh*0.35):int(fh*0.65), int(fw*0.15):int(fw*0.85)]
    cheek_pixels = cheek.reshape(-1, 3)

    # Further filter cheek pixels through skin mask if enough remain
    cheek_ycrcb = cv2.cvtColor(cheek, cv2.COLOR_BGR2YCrCb)
    cheek_mask = cv2.inRange(cheek_ycrcb, lower, upper)
    cheek_skin = cheek[cheek_mask > 0]

    if len(cheek_skin) >= 30:
        avg_bgr = np.mean(cheek_skin, axis=0)
    else:
        avg_bgr = np.mean(cheek_pixels, axis=0)

    avg_rgb = avg_bgr[::-1].copy()  # BGR -> RGB
    return avg_rgb


# ----------------------------
# Dominant Foreground Color (ignores white background)
# ----------------------------
def get_dominant_color(image):
    img_resized = cv2.resize(image, (150, 150))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    pixels = img_rgb.reshape(-1, 3).astype(np.float32)

    brightness = pixels.mean(axis=1)
    mask = (brightness < 230) & (brightness > 20)
    filtered = pixels[mask]

    if len(filtered) < 100:
        filtered = pixels

    n_clusters = min(3, len(filtered))
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    kmeans.fit(filtered)

    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    dominant_label = labels[np.argmax(counts)]
    return kmeans.cluster_centers_[dominant_label]


# ----------------------------
# RGB to Hue
# ----------------------------
def rgb_to_hue(color):
    r, g, b = color[0] / 255.0, color[1] / 255.0, color[2] / 255.0
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    return float(h * 360)


# ----------------------------
# Outfit Compatibility
# ----------------------------
def evaluate_match(skin_hue, outfit_hue):
    diff = abs(skin_hue - outfit_hue)
    if diff > 180:
        diff = 360 - diff

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


ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def save_upload(file_storage, label):
    orig = secure_filename(file_storage.filename or "")
    ext = orig.rsplit(".", 1)[1].lower() if "." in orig else "jpg"
    if ext not in ALLOWED_EXTENSIONS:
        ext = "jpg"
    unique_name = f"{label}_{uuid.uuid4().hex[:8]}.{ext}"
    save_path = os.path.join(UPLOAD_FOLDER, unique_name)
    file_storage.save(save_path)
    return save_path


def save_base64_image(data_uri, label):
    try:
        if "," in data_uri:
            _, encoded = data_uri.split(",", 1)
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


def get_image_from_request(field_name):
    if field_name in request.files:
        f = request.files[field_name]
        if f and f.filename:
            path = save_upload(f, field_name)
            if path:
                img = cv2.imread(path)
                if img is not None:
                    return img, path

    b64 = request.form.get(f"{field_name}_base64") or request.form.get(field_name)
    if b64 and ("base64" in b64 or len(b64) > 500):
        path = save_base64_image(b64, field_name)
        if path:
            img = cv2.imread(path)
            if img is not None:
                return img, path

    if request.is_json:
        data = request.get_json(silent=True)
        if data and field_name in data:
            path = save_base64_image(data[field_name], field_name)
            if path:
                img = cv2.imread(path)
                if img is not None:
                    return img, path

    return None, None



@app.route("/debug-skin", methods=["POST"])
def debug_skin():
    """
    POST a face image to this endpoint to get raw color values.
    Helps calibrate skin tone thresholds.
    """
    face_img, face_path = get_image_from_request("face")
    if face_img is None:
        return jsonify({"error": "No face image received"})

    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return jsonify({"error": "No face detected"})

    x, y, w, h = faces[0]
    face_region = face_img[y: y + h, x: x + w]

    # YCrCb skin mask
    ycrcb = cv2.cvtColor(face_region, cv2.COLOR_BGR2YCrCb)
    lower = np.array([0, 128, 70], dtype=np.uint8)
    upper = np.array([255, 185, 135], dtype=np.uint8)
    skin_mask = cv2.inRange(ycrcb, lower, upper)
    skin_pixel_count = int(np.sum(skin_mask > 0))
    total_pixels = face_region.shape[0] * face_region.shape[1]

    avg_skin_rgb = get_skin_color(face_region)
    r, g, b = float(avg_skin_rgb[0]), float(avg_skin_rgb[1]), float(avg_skin_rgb[2])
    Y = 0.299 * r + 0.587 * g + 0.114 * b

    # Also check center crop directly
    cy, cx = face_region.shape[0]//2, face_region.shape[1]//2
    h2, w2 = face_region.shape[0]//4, face_region.shape[1]//4
    center = face_region[cy-h2:cy+h2, cx-w2:cx+w2]
    center_mean_bgr = np.mean(center.reshape(-1,3), axis=0)
    center_rgb = center_mean_bgr[::-1]
    cr2, cg2, cb2 = float(center_rgb[0]), float(center_rgb[1]), float(center_rgb[2])
    Y_center = 0.299*cr2 + 0.587*cg2 + 0.114*cb2

    try:
        if face_path and os.path.exists(face_path):
            os.remove(face_path)
    except OSError:
        pass

    return jsonify({
        "face_detected": True,
        "skin_pixels_found": skin_pixel_count,
        "total_face_pixels": total_pixels,
        "skin_mask_coverage_pct": round(skin_pixel_count / total_pixels * 100, 1),
        "avg_skin_rgb": [round(r, 1), round(g, 1), round(b, 1)],
        "Y_luma": round(Y, 1),
        "Y_luma_center_crop": round(Y_center, 1),
        "center_crop_rgb": [round(cr2,1), round(cg2,1), round(cb2,1)],
        "current_classification": classify_skin_tone(avg_skin_rgb),
        "thresholds": {
            "Fair": "Y > 165",
            "Medium": "Y > 130",
            "Wheatish": "Y > 90",
            "Dark": "Y <= 90"
        }
    })

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
        return jsonify({
            "error": f"Could not read image(s): {', '.join(missing)}. Please upload or capture valid photos."
        })

    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return jsonify({"error": "No face detected. Please use a clear, front-facing photo."})

    x, y, w, h = faces[0]
    face_region = face_img[y: y + h, x: x + w]

    avg_skin_rgb = get_skin_color(face_region)
    skin_tone = classify_skin_tone(avg_skin_rgb)

    outfit_color = get_dominant_color(outfit_img)

    skin_hue = rgb_to_hue(avg_skin_rgb)
    outfit_hue = rgb_to_hue(outfit_color)

    result, score = evaluate_match(skin_hue, outfit_hue)

    r, g, b = float(avg_skin_rgb[0]), float(avg_skin_rgb[1]), float(avg_skin_rgb[2])
    Y_luma = round(0.299 * r + 0.587 * g + 0.114 * b, 1)

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
        "skin_color": [float(x) for x in avg_skin_rgb],
        "outfit_color": [float(x) for x in outfit_color],
        "debug": {
            "skin_Y_luma": Y_luma,
            "skin_hue": round(float(skin_hue), 1),
            "outfit_hue": round(float(outfit_hue), 1),
            "skin_rgb": [round(r), round(g), round(b)]
        }
    })


if __name__ == "__main__":
    app.run(debug=True)