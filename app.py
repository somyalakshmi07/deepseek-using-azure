from flask import Flask, request, jsonify
import requests
import os
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

HF_API_URL = os.getenv("HF_API_URL", "https://api-inference.huggingface.co/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
HF_TOKEN = os.getenv("HF_TOKEN")

@app.route("/")
def home():
    return "DeepSeek API is running!", 200

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid or missing JSON body"}), 400

    prompt = data.get("prompt", "")
    if not isinstance(prompt, str) or not prompt.strip():
        return jsonify({"error": "Missing or empty 'prompt' field"}), 400

    if not HF_TOKEN:
        logging.error("HF_TOKEN is not set")
        return jsonify({"error": "HF_TOKEN environment variable is not set"}), 500

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {"inputs": prompt}

    try:
        resp = requests.post(HF_API_URL, headers=headers, json=payload, timeout=30)
    except requests.RequestException as e:
        logging.exception("Request to Hugging Face API failed")
        return jsonify({"error": "Failed to contact Hugging Face API", "details": str(e)}), 502

    # Forward upstream status and JSON or text
    content_type = resp.headers.get("Content-Type", "")
    try:
        body = resp.json() if "application/json" in content_type else {"text": resp.text}
    except ValueError:
        body = {"text": resp.text}

    return jsonify({
        "upstream_status": resp.status_code,
        "upstream_response": body
    }), (200 if resp.ok else 502)