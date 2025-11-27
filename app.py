from flask import Flask, request, jsonify
import requests
import base64
import os

app = Flask(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") # <-- replace here
GEMINI_MODEL = "gemini-2.0-flash"        # vision model

def get_base64_from_url(image_url):
    """
    Download image from URL and convert to Base64
    """
    img_bytes = requests.get(image_url).content
    return base64.b64encode(img_bytes).decode("utf-8")


def call_gemini_ocr(base64_image):
    """
    Call Gemini Vision API with Base64 image for OCR
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": "Extract all text clearly from this prescription image. Output clean plain text only."},
                    {
                        "inlineData": {
                            "mimeType": "image/jpeg",
                            "data": base64_image
                        }
                    }
                ]
            }
        ]
    }

    headers = {"Content-Type": "application/json"}
    response = requests.post(url, json=payload, headers=headers)

    if response.status_code != 200:
        return f"Gemini Error: {response.text}"

    data = response.json()
    
    # Extract text
    try:
        extracted_text = data["candidates"][0]["content"]["parts"][0]["text"]
        return extracted_text.strip()
    except:
        return "No text extracted."


@app.route("/ocr", methods=["POST"])
def ocr():
    """
    Main OCR endpoint
    Zobot calls this with {"image_url": "..."}
    """

    req = request.get_json()

    if "image_url" not in req:
        return jsonify({"error": "image_url missing"}), 400

    image_url = req["image_url"]

    # 1. Convert image URL → Base64
    base64_img = get_base64_from_url(image_url)

    # 2. Send to Gemini Vision
    result = call_gemini_ocr(base64_img)

    return jsonify({
        "extracted_text": result
    })


if __name__ == "__main__":
    app.run(debug=True)