from flask import Flask, request, jsonify
import requests
import os

app = Flask(__name__)

HF_API_URL = "https://api-inference.huggingface.co/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
HF_TOKEN = os.getenv("HF_TOKEN")

@app.route("/")
def home():
    return "DeepSeek API is running!"

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    prompt = data.get("prompt", "")
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    response = requests.post(HF_API_URL, headers=headers, json={"inputs": prompt})
    return jsonify(response.json())