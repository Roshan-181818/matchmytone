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

MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp", "heic", "heif"}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def validate_image_size(data_bytes):
    return len(data_bytes) <= MAX_FILE_SIZE_BYTES


def classify_skin_tone(avg_rgb):
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


def get_skin_color(face_img):
    ycrcb = cv2.cvtColor(face_img, cv2.COLOR_BGR2YCrCb)
    lower = np.array([0, 128, 70], dtype=np.uint8)
    upper = np.array([255, 185, 135], dtype=np.uint8)
    skin_mask = cv2.inRange(ycrcb, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_DILATE, kernel)
    skin_pixels = face_img[skin_mask > 0]
    if len(skin_pixels) < 50:
        h, w = face_img.shape[:2]
        cy, cx = h // 2, w // 2
        patch = face_img[cy - h // 4: cy + h // 4, cx - w // 4: cx + w // 4]
        skin_pixels = patch.reshape(-1, 3)
    fh, fw = face_img.shape[:2]
    cheek = face_img[int(fh*0.35):int(fh*0.65), int(fw*0.15):int(fw*0.85)]
    cheek_pixels = cheek.reshape(-1, 3)
    cheek_ycrcb = cv2.cvtColor(cheek, cv2.COLOR_BGR2YCrCb)
    cheek_mask = cv2.inRange(cheek_ycrcb, lower, upper)
    cheek_skin = cheek[cheek_mask > 0]
    if len(cheek_skin) >= 30:
        avg_bgr = np.median(cheek_skin, axis=0)
    else:
        avg_bgr = np.median(cheek_pixels, axis=0)
    avg_rgb = avg_bgr[::-1].copy()
    return avg_rgb


def get_largest_face(faces):
    return max(faces, key=lambda f: f[2] * f[3])


def get_background_color(img_rgb):
    h, w = img_rgb.shape[:2]
    corner_size = max(10, min(h, w) // 8)
    corners = np.concatenate([
        img_rgb[:corner_size, :corner_size].reshape(-1, 3),
        img_rgb[:corner_size, -corner_size:].reshape(-1, 3),
        img_rgb[-corner_size:, :corner_size].reshape(-1, 3),
        img_rgb[-corner_size:, -corner_size:].reshape(-1, 3),
    ], axis=0).astype(np.float32)
    return np.median(corners, axis=0)


def color_distance(c1, c2):
    return np.sqrt(np.sum((np.array(c1, dtype=np.float32) - np.array(c2, dtype=np.float32)) ** 2))


def get_dominant_color(image):
    img_resized = cv2.resize(image, (200, 200))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    corner_size = max(10, min(h, w) // 8)
    corners = np.concatenate([
        img_rgb[:corner_size, :corner_size].reshape(-1, 3),
        img_rgb[:corner_size, -corner_size:].reshape(-1, 3),
        img_rgb[-corner_size:, :corner_size].reshape(-1, 3),
        img_rgb[-corner_size:, -corner_size:].reshape(-1, 3),
    ], axis=0).astype(np.float32)
    bg_color = np.median(corners, axis=0)

    # Use center 50% of image where outfit is
    cy, cx = h // 2, w // 2
    margin_y, margin_x = h // 4, w // 4
    center_crop = img_rgb[cy - margin_y:cy + margin_y, cx - margin_x:cx + margin_x]
    pixels = center_crop.reshape(-1, 3).astype(np.float32)

    brightness = pixels.mean(axis=1)
    dist_from_bg = np.sqrt(((pixels - bg_color) ** 2).sum(axis=1))
    mask = (brightness < 235) & (dist_from_bg > 60)
    filtered = pixels[mask]

    if len(filtered) < 100:
        filtered = pixels[pixels.mean(axis=1) < 235]
    if len(filtered) < 100:
        filtered = pixels

    n_clusters = min(8, max(4, len(filtered) // 100))
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    kmeans.fit(filtered)
    labels, counts = np.unique(kmeans.labels_, return_counts=True)

    outfit_clusters = [
        (kmeans.cluster_centers_[label], count)
        for label, count in zip(labels, counts)
        if color_distance(kmeans.cluster_centers_[label], bg_color) > 60
    ]
    if not outfit_clusters:
        return kmeans.cluster_centers_[labels[np.argmax(counts)]]

    outfit_clusters.sort(key=lambda x: x[1], reverse=True)
    outfit_total = sum(c for _, c in outfit_clusters)

    if outfit_clusters[0][1] / outfit_total > 0.60:
        return outfit_clusters[0][0]

    top_clusters, covered = [], 0
    for center, count in outfit_clusters:
        top_clusters.append((center, count))
        covered += count
        if covered / outfit_total >= 0.70:
            break

    weights = np.array([c for _, c in top_clusters], dtype=np.float32)
    colors = np.array([col for col, _ in top_clusters], dtype=np.float32)
    weights /= weights.sum()
    return (colors * weights[:, np.newaxis]).sum(axis=0)


def rgb_to_hue(color):
    r, g, b = color[0] / 255.0, color[1] / 255.0, color[2] / 255.0
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    return float(h * 360)


def is_neutral_color(color):
    r, g, b = color[0] / 255.0, color[1] / 255.0, color[2] / 255.0
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    return s < 0.15


def hue_diff(h1, h2):
    diff = abs(h1 - h2)
    if diff > 180:
        diff = 360 - diff
    return diff


def evaluate_match(skin_hue, outfit_hue, outfit_color=None, skin_tone=None):
    # Step 1: Neutrals (black, white, grey) — universally flattering but not perfect
    if outfit_color is not None and is_neutral_color(outfit_color):
        r, g, b = outfit_color[0]/255.0, outfit_color[1]/255.0, outfit_color[2]/255.0
        _, _, v = colorsys.rgb_to_hsv(r, g, b)
        # Pure white/black score slightly lower than mid-neutrals
        if v > 0.92 or v < 0.15:
            return "Match", 80
        return "Match", 84

    diff = hue_diff(skin_hue, outfit_hue)

    # Step 2: Check if outfit hue is close to a palette-recommended hue
    if skin_tone and outfit_color is not None:
        palette = SKIN_TONE_PALETTES.get(skin_tone, {})
        best_colors = palette.get("best", [])
        min_hue_dist = 360
        for rec_color in best_colors:
            rec_hue = rec_color.get("hue", 0)
            d = hue_diff(outfit_hue, rec_hue)
            if d < min_hue_dist:
                min_hue_dist = d
        # Outfit hue is very close to a recommended hue — boost score
        if min_hue_dist <= 20:
            return "Match", 93
        elif min_hue_dist <= 40:
            return "Match", 87

    # Step 3: Pure hue-based scoring — continuous, not bucketed
    # Analogous (similar hues): great match
    # Complementary (opposite ~180°): strong contrast match
    # Split-complementary (~150° or ~210°): good match
    # Everything else: scaled score

    if diff <= 20:
        # Very analogous — harmonious
        score = int(90 + (20 - diff) / 20 * 8)   # 90–98
        level = "Match"
    elif diff <= 40:
        score = int(82 + (40 - diff) / 20 * 8)   # 82–90
        level = "Match"
    elif diff <= 60:
        score = int(70 + (60 - diff) / 20 * 12)  # 70–82
        level = "Partial Match"
    elif diff <= 90:
        score = int(55 + (90 - diff) / 30 * 15)  # 55–70
        level = "Partial Match"
    elif diff <= 120:
        score = int(38 + (120 - diff) / 30 * 17) # 38–55
        level = "Not a Match"
    elif diff <= 150:
        # Approaching complementary
        score = int(55 + (diff - 120) / 30 * 15) # 55–70
        level = "Partial Match"
    elif diff <= 180:
        # Complementary — high contrast, stylistically strong
        score = int(80 + (diff - 150) / 30 * 13) # 80–93
        level = "Match"
    else:
        score = 50
        level = "Partial Match"

    score = max(20, min(98, score))
    return level, score


def check_lighting(face_region):
    gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
    avg_brightness = float(np.mean(gray))
    std_brightness = float(np.std(gray))
    if avg_brightness < 50:
        return "⚠ Very dark photo detected. Skin tone accuracy may be reduced — please use a well-lit photo."
    elif avg_brightness < 80:
        return "⚠ Low lighting detected. For best results, take the photo in natural or bright indoor light."
    elif avg_brightness > 230:
        return "⚠ Overexposed photo detected. Avoid direct flash or harsh sunlight on your face."
    elif std_brightness < 15:
        return "⚠ Flat lighting detected. Results may be less accurate — try a photo with natural lighting."
    return None


SKIN_TONE_PALETTES = {
    "Fair": {
        "best": [
            {"name": "Navy Blue", "hex": "#1B3A6B", "hue": 220},
            {"name": "Emerald Green", "hex": "#2E8B57", "hue": 146},
            {"name": "Dusty Rose", "hex": "#C4829D", "hue": 337},
            {"name": "Lavender", "hex": "#8B7BB5", "hue": 260},
            {"name": "Burgundy", "hex": "#7D1935", "hue": 345},
        ],
        "avoid": ["Neon Yellow", "Very Pale Pastels", "White-on-White"],
        "tip": "Fair skin tones shine in jewel tones and rich, deep shades that create beautiful contrast."
    },
    "Medium": {
        "best": [
            {"name": "Terracotta", "hex": "#C36A2D", "hue": 26},
            {"name": "Olive Green", "hex": "#6B7C3A", "hue": 77},
            {"name": "Royal Purple", "hex": "#6A1B9A", "hue": 280},
            {"name": "Teal", "hex": "#008080", "hue": 180},
            {"name": "Coral", "hex": "#E2725B", "hue": 10},
        ],
        "avoid": ["Beige", "Khaki", "Muddy Browns"],
        "tip": "Medium skin tones look stunning in warm earth tones and vibrant jewel-toned shades."
    },
    "Wheatish": {
        "best": [
            {"name": "Deep Orange", "hex": "#D2691E", "hue": 25},
            {"name": "Forest Green", "hex": "#228B22", "hue": 120},
            {"name": "Mustard Yellow", "hex": "#C9A227", "hue": 44},
            {"name": "Cobalt Blue", "hex": "#0047AB", "hue": 215},
            {"name": "Magenta", "hex": "#C2185B", "hue": 337},
        ],
        "avoid": ["Pale Pastels", "Light Beige", "Washed-out Yellows"],
        "tip": "Wheatish skin tones glow in bold, saturated colors — especially warm oranges, golds, and blues."
    },
    "Dark": {
        "best": [
            {"name": "Bright White", "hex": "#F5F5F5", "hue": 0},
            {"name": "Electric Blue", "hex": "#0066FF", "hue": 220},
            {"name": "Bright Red", "hex": "#CC0000", "hue": 0},
            {"name": "Gold", "hex": "#FFD700", "hue": 51},
            {"name": "Hot Pink", "hex": "#FF69B4", "hue": 330},
        ],
        "avoid": ["Dark Browns", "Very Dark Navy", "Black-on-Black"],
        "tip": "Dark skin tones look incredible in bright, bold colors and high-contrast combinations."
    }
}


def get_recommendations(skin_tone, result, score=0):
    palette = SKIN_TONE_PALETTES.get(skin_tone, SKIN_TONE_PALETTES["Medium"])

    # Perfect score — no suggestions needed
    if result == "Match" and score >= 95:
        return {
            "show": False,
            "boost": False,
            "tip": palette["tip"],
            "message": f"Perfect match! Your outfit looks amazing with your {skin_tone.lower()} skin tone. 🎉"
        }

    # Great match but room to improve (score 80–94)
    if result == "Match" and score < 95:
        # Pick top 3 colors that would score 95+ (closest hue matches to palette)
        boost_colors = palette["best"][:3]
        return {
            "show": True,
            "boost": True,
            "best_colors": boost_colors,
            "tip": palette["tip"],
            "message": f"Great match! Want a perfect score? Switching to one of these colors will boost you to 95+:"
        }

    # Partial match or not a match — show full recommendations
    rec = {
        "show": True,
        "boost": False,
        "best_colors": palette["best"],
        "avoid": palette["avoid"],
        "tip": palette["tip"],
    }
    if result == "Not a Match":
        rec["message"] = f"This outfit doesn't complement your {skin_tone.lower()} skin tone well. Try one of these colors instead:"
    else:
        rec["message"] = f"Decent match! For your {skin_tone.lower()} skin tone, these colors would look even better:"
    return rec


@app.route("/")
def home():
    return render_template("index.html")


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
        if not validate_image_size(img_bytes):
            return None, "Image exceeds 5MB limit. Please upload a smaller file."
        unique_name = f"{label}_{uuid.uuid4().hex[:8]}.jpg"
        save_path = os.path.join(UPLOAD_FOLDER, unique_name)
        with open(save_path, "wb") as f:
            f.write(img_bytes)
        return save_path, None
    except Exception:
        return None, "Failed to decode image."


def get_image_from_request(field_name):
    if field_name in request.files:
        f = request.files[field_name]
        if f and f.filename:
            if not allowed_file(f.filename):
                return None, None, f"Invalid file type for {field_name}. Only JPEG and PNG are allowed."
            f.seek(0, 2)
            size = f.tell()
            f.seek(0)
            if size > MAX_FILE_SIZE_BYTES:
                return None, None, f"{field_name} image exceeds 5MB limit."
            path = save_upload(f, field_name)
            if path:
                img = cv2.imread(path)
                if img is not None:
                    return img, path, None
    b64 = request.form.get(f"{field_name}_base64") or request.form.get(field_name)
    if b64 and ("base64" in b64 or len(b64) > 500):
        path, err = save_base64_image(b64, field_name)
        if err:
            return None, None, err
        if path:
            img = cv2.imread(path)
            if img is not None:
                return img, path, None
    if request.is_json:
        data = request.get_json(silent=True)
        if data and field_name in data:
            path, err = save_base64_image(data[field_name], field_name)
            if err:
                return None, None, err
            if path:
                img = cv2.imread(path)
                if img is not None:
                    return img, path, None
    return None, None, None


def cleanup_files(*paths):
    for p in paths:
        try:
            if p and os.path.exists(p):
                os.remove(p)
        except OSError:
            pass


@app.route("/debug-skin", methods=["POST"])
def debug_skin():
    face_img, face_path, err = get_image_from_request("face")
    if err:
        return jsonify({"error": err})
    if face_img is None:
        return jsonify({"error": "No face image received"})
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        cleanup_files(face_path)
        return jsonify({"error": "No face detected"})
    x, y, w, h = get_largest_face(faces)
    face_region = face_img[y: y + h, x: x + w]
    ycrcb = cv2.cvtColor(face_region, cv2.COLOR_BGR2YCrCb)
    lower = np.array([0, 128, 70], dtype=np.uint8)
    upper = np.array([255, 185, 135], dtype=np.uint8)
    skin_mask = cv2.inRange(ycrcb, lower, upper)
    skin_pixel_count = int(np.sum(skin_mask > 0))
    total_pixels = face_region.shape[0] * face_region.shape[1]
    avg_skin_rgb = get_skin_color(face_region)
    r, g, b = float(avg_skin_rgb[0]), float(avg_skin_rgb[1]), float(avg_skin_rgb[2])
    Y = 0.299 * r + 0.587 * g + 0.114 * b
    cy, cx = face_region.shape[0]//2, face_region.shape[1]//2
    h2, w2 = face_region.shape[0]//4, face_region.shape[1]//4
    center = face_region[cy-h2:cy+h2, cx-w2:cx+w2]
    center_mean_bgr = np.mean(center.reshape(-1, 3), axis=0)
    center_rgb = center_mean_bgr[::-1]
    cr2, cg2, cb2 = float(center_rgb[0]), float(center_rgb[1]), float(center_rgb[2])
    Y_center = 0.299*cr2 + 0.587*cg2 + 0.114*cb2
    lighting_warning = check_lighting(face_region)
    cleanup_files(face_path)
    return jsonify({
        "face_detected": True,
        "faces_found": len(faces),
        "skin_pixels_found": skin_pixel_count,
        "total_face_pixels": total_pixels,
        "skin_mask_coverage_pct": round(skin_pixel_count / total_pixels * 100, 1),
        "avg_skin_rgb": [round(r, 1), round(g, 1), round(b, 1)],
        "Y_luma": round(Y, 1),
        "Y_luma_center_crop": round(Y_center, 1),
        "center_crop_rgb": [round(cr2, 1), round(cg2, 1), round(cb2, 1)],
        "current_classification": classify_skin_tone(avg_skin_rgb),
        "lighting_warning": lighting_warning,
        "thresholds": {"Fair": "Y > 155", "Medium": "Y > 128", "Wheatish": "Y > 100", "Dark": "Y <= 100"}
    })


@app.route("/analyze", methods=["POST"])
def analyze():
    face_img, face_path, face_err = get_image_from_request("face")
    outfit_img, outfit_path, outfit_err = get_image_from_request("outfit")

    if face_err or outfit_err:
        cleanup_files(face_path, outfit_path)
        return jsonify({"error": face_err or outfit_err})

    if face_img is None or outfit_img is None:
        missing = []
        if face_img is None: missing.append("face")
        if outfit_img is None: missing.append("outfit")
        cleanup_files(face_path, outfit_path)
        return jsonify({"error": f"Could not read image(s): {', '.join(missing)}. Please upload valid JPEG or PNG photos."})

    # Check lighting on full image FIRST before face detection
    full_lighting_warning = check_lighting(face_img)

    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        cleanup_files(face_path, outfit_path)
        if full_lighting_warning:
            return jsonify({"error": f"Face not detected. {full_lighting_warning.replace(chr(9888) + ' ', '')}"})
        return jsonify({"error": "No face detected. Please use a clear, front-facing photo with good lighting."})

    x, y, w, h = get_largest_face(faces)
    face_region = face_img[y: y + h, x: x + w]
    lighting_warning = check_lighting(face_region) or full_lighting_warning
    avg_skin_rgb = get_skin_color(face_region)
    skin_tone = classify_skin_tone(avg_skin_rgb)
    outfit_color = get_dominant_color(outfit_img)
    skin_hue = rgb_to_hue(avg_skin_rgb)
    outfit_hue = rgb_to_hue(outfit_color)

    # Pass skin_tone so recommended colors get high scores
    result, score = evaluate_match(skin_hue, outfit_hue, outfit_color, skin_tone)

    r, g, b = float(avg_skin_rgb[0]), float(avg_skin_rgb[1]), float(avg_skin_rgb[2])
    Y_luma = round(0.299 * r + 0.587 * g + 0.114 * b, 1)
    cleanup_files(face_path, outfit_path)
    recommendations = get_recommendations(skin_tone, result, score)

    response = {
        "skin_tone": skin_tone,
        "result": result,
        "score": score,
        "skin_color": [float(x) for x in avg_skin_rgb],
        "outfit_color": [float(x) for x in outfit_color],
        "recommendations": recommendations,
        "debug": {
            "skin_Y_luma": Y_luma,
            "skin_hue": round(float(skin_hue), 1),
            "outfit_hue": round(float(outfit_hue), 1),
            "skin_rgb": [round(r), round(g), round(b)],
            "faces_detected": len(faces),
            "outfit_is_neutral": bool(is_neutral_color(outfit_color))
        }
    }
    if lighting_warning:
        response["warning"] = lighting_warning
    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True)