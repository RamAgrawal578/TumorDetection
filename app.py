import os
import sys
import logging

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)

logger.info("Starting app imports...")

try:
    import uuid
    import numpy as np
    from flask import Flask, render_template, request
    from PIL import Image
    from tensorflow import keras
    from werkzeug.utils import secure_filename
    logger.info("Core imports OK")
except Exception as e:
    logger.error(f"Core import failed: {e}")
    raise

try:
    from src.config import IMG_SIZE
    from src.utils import load_class_names, preprocess_pil_image
    logger.info(f"src imports OK — IMG_SIZE={IMG_SIZE}")
except Exception as e:
    logger.error(f"src import failed: {e}")
    raise

app = Flask(__name__, template_folder="templates", static_folder="static")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "brain_tumor_classifier.keras")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "models", "class_names.json")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

logger.info(f"MODEL_PATH exists: {os.path.exists(MODEL_PATH)}")
logger.info(f"CLASS_NAMES_PATH exists: {os.path.exists(CLASS_NAMES_PATH)}")

model = None
class_names = None


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def prettify_label(label: str) -> str:
    cleaned = str(label).lower().replace("-", "").replace("_", "").replace(" ", "")
    if cleaned == "notumor":
        return "No Tumor"
    elif cleaned == "glioma":
        return "Glioma"
    elif cleaned == "meningioma":
        return "Meningioma"
    elif cleaned == "pituitary":
        return "Pituitary"
    return str(label).title()


def load_assets():
    loaded_model = keras.models.load_model(MODEL_PATH)
    loaded_class_names = load_class_names(CLASS_NAMES_PATH)
    return loaded_model, loaded_class_names


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    return "OK", 200


@app.route("/predict", methods=["POST"])
def predict():
    global model, class_names
    try:
        if model is None or class_names is None:
            model, class_names = load_assets()

        if "file" not in request.files:
            return "No file uploaded", 400

        file = request.files["file"]
        if file.filename == "":
            return "No file selected", 400
        if not allowed_file(file.filename):
            return "Invalid file type", 400

        original_name = secure_filename(file.filename)
        ext = original_name.rsplit(".", 1)[1].lower()
        unique_name = f"{uuid.uuid4().hex}.{ext}"
        absolute_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
        file.save(absolute_path)

        relative_path = f"uploads/{unique_name}"
        image = Image.open(absolute_path).convert("RGB")
        batch = preprocess_pil_image(image, IMG_SIZE)

        preds = model.predict(batch, verbose=0)[0]
        idx = int(np.argmax(preds))
        prediction = prettify_label(class_names[idx])
        confidence = round(float(preds[idx]) * 100, 2)

        probability_items = sorted(
            [
                {
                    "label": prettify_label(label),
                    "score": float(score),
                    "percent": round(float(score) * 100, 2),
                }
                for label, score in zip(class_names, preds.tolist())
            ],
            key=lambda x: x["score"],
            reverse=True,
        )

        return render_template(
            "result.html",
            prediction=prediction,
            confidence=confidence,
            image_file=relative_path,
            probabilities=probability_items,
            uploaded_name=original_name,
        )

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return f"Error: {str(e)}", 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)