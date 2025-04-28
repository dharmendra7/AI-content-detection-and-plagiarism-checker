from flask import Flask, jsonify, request
from flask_cors import CORS

from detect_text import detecting_text

app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({"message": "Welcome to the Plagiarism Checker API!"})


@app.route('/ai-detection', methods=['POST'])
def check_plagiarism():
    # Placeholder for the actual plagiarism detection logic
    # In a real application, you would process the input text and return results
    data = request.get_json()
    text = data.get('text', '')
    
    probability, predicted_label = detecting_text(text)
    predicted_label = f"{'AI Generated' if predicted_label == 1 else 'Not AI Generated'}"
    return jsonify({"probability": probability, "predicted_label": predicted_label})

if __name__ == '__main__':
    app.run(debug=True)