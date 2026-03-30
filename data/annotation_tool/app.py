import os
import json
from flask import Flask, render_template, request, jsonify, send_from_directory

app = Flask(__name__)

# 🔥 BASE DIRECTORY (critical fix)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ✅ Correct paths (relative to app.py)
IMAGE_FOLDER = os.path.join(BASE_DIR, "images")
ANNOTATION_FILE = os.path.join(BASE_DIR, "annotations", "labels.json")

# -------------------------------
# LOAD IMAGES
# -------------------------------
def load_images():
    image_paths = []

    for root, _, files in os.walk(IMAGE_FOLDER):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                full_path = os.path.join(root, file)

                # make path relative to IMAGE_FOLDER
                rel_path = os.path.relpath(full_path, IMAGE_FOLDER)

                # normalize slashes (important for browser)
                rel_path = rel_path.replace("\\", "/")

                image_paths.append(rel_path)

    image_paths = sorted(image_paths)

    print(f"Loaded {len(image_paths)} images")
    print("Example:", image_paths[:3])

    return image_paths

# -------------------------------
# LOAD ANNOTATIONS
# -------------------------------
def load_annotations():
    if not os.path.exists(ANNOTATION_FILE):
        return {}

    try:
        with open(ANNOTATION_FILE, "r") as f:
            content = f.read().strip()

            if not content:
                return {}

            return json.loads(content)

    except json.JSONDecodeError:
        print("Invalid JSON detected. Resetting annotations file.")
        return {}

# -------------------------------
# SAVE ANNOTATIONS
# -------------------------------
def save_annotations(data):
    os.makedirs(os.path.dirname(ANNOTATION_FILE), exist_ok=True)
    with open(ANNOTATION_FILE, "w") as f:
        json.dump(data, f, indent=2)

# -------------------------------
# INIT DATA
# -------------------------------
images = load_images()
annotations = load_annotations()

# -------------------------------
# ROUTES
# -------------------------------
@app.route("/")
def index():
    return render_template("index.html", total=len(images))

# 🔥 Serve images properly
@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory(IMAGE_FOLDER, filename)

@app.route("/get_image/<int:index>")
def get_image(index):
    if index < 0 or index >= len(images):
        return jsonify({"error": "out of range"})

    image_path = images[index]
    annotation = annotations.get(image_path, {})

    return jsonify({
        "image_path": image_path,
        "annotation": annotation
    })

@app.route("/save", methods=["POST"])
def save():
    data = request.json
    image_path = data["image_path"]
    labels = data["labels"]

    annotations[image_path] = labels
    save_annotations(annotations)

    return jsonify({"status": "saved"})

# -------------------------------
# RUN
# -------------------------------
if __name__ == "__main__":
    print("BASE_DIR:", BASE_DIR)
    print("IMAGE_FOLDER:", IMAGE_FOLDER)
    print("Exists:", os.path.exists(IMAGE_FOLDER))

    app.run(debug=True)