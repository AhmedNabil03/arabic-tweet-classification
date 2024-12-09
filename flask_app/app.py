from flask import Flask, request, jsonify
from flask_cors import CORS

from flask_workflow import classify_text, extract_entities

app = Flask(__name__)
CORS(app)

# Routes
@app.route("/", methods=["GET"])
def home():
    return "Flask app is running!"

@app.route("/classify", methods=["POST"])
def classify():
    data = request.get_json()
    text = data.get("text")
    
    if not text:
        return jsonify({"error": "Text is required"}), 400
    
    # Classify the text
    classification_result = classify_text(text)
    
    # If the classification is "com", extract entities
    if classification_result == "com":
        entities = extract_entities(text)
        return jsonify({"classification": classification_result, "entities": entities})
    
    # If it's not "com", just return the classification
    return jsonify({"classification": classification_result})

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
