import os
from pathlib import Path

import joblib
import torch
from flask import Flask, Response, request, jsonify
from google.cloud import storage
from loguru import logger
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class TextClassificationPredictor:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.label_encoder = None

    def load(self) -> None:
        model_path = os.environ["AIP_STORAGE_URI"]
        logger.info(f"Loading model from {model_path}")

        storage_client = storage.Client()
        bucket_name = model_path.split("/")[2]
        bucket = storage_client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix="/".join(model_path.split("/")[3:]))
        model_dir = Path("./model")
        counter = 0
        for blob in blobs:
            if blob.name.endswith("/"):
                continue
            file_split = blob.name.split("/")
            sub_dir = Path("/".join(file_split[:-1]))
            download_path = str(model_dir / Path(blob.name).name)
            logger.info(f"Downloading {blob.name} to {download_path}")
            (model_dir / sub_dir).mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(download_path)
            counter += 1
        logger.info(f"Downloaded {counter} files")
        logger.debug(list(model_dir.iterdir()))

        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.label_encoder = joblib.load(model_dir / "label_encoder.joblib")
        logger.info("Model loaded")

    def predict(self, instances: list[dict]) -> list[dict]:
        results = []
        for instance in instances:
            text = instance.get("text", "")
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)

            with torch.no_grad():
                outputs = self.model(**inputs)

            predicted_class_idx = outputs.logits.argmax().item()
            predicted_class = self.label_encoder.inverse_transform([predicted_class_idx])[0]
            probabilities = outputs.logits.softmax(dim=-1).tolist()[0]

            results.append({
                "predicted_class": predicted_class,
                "probabilities": {
                    self.label_encoder.inverse_transform([i])[0]: prob
                    for i, prob in enumerate(probabilities)
                },
            })
        return results


app = Flask(__name__)
predictor = TextClassificationPredictor()


@app.route("/health", methods=["GET"])
def health() -> Response:
    return jsonify({"status": "OK"})


@app.route("/predict", methods=["POST"])
def predict() -> Response:
    logger.debug(request.content_type)

    data = request.json
    instances = data.get("instances", [])
    _ = data.get("parameters", {})

    predictions = predictor.predict(instances)
    return jsonify({"predictions": predictions})


if __name__ == "__main__":
    predictor.load()
    app.run(host="0.0.0.0", port=8080)
