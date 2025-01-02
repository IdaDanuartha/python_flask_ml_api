from flask import Flask, request, jsonify
from controllers.ml import load_model, predict
import numpy as np

app = Flask(__name__)

# Load the ML model at startup
model = load_model('SignLanguage.h5')

# Define class labels for predictions
class_labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

@app.route('/questions', methods=['GET'])
def get_questions():
    questions = [
        {"question": label, "hint": f"/static/sign_language_dataset/{label}.jpg"} for label in class_labels
    ]
    return jsonify({"questions": questions})

@app.route('/answers', methods=['POST'])
def post_answers():
    # Get the question and image data from the request
    if 'question' not in request.form or 'image' not in request.files:
        return jsonify({"error": "Question or image file is required"}), 400
    
    question = request.form['question']
    image_file = request.files['image']
    image_buffer = image_file.read()
    
    # Make predictions
    predictions = predict(model, image_buffer)
    
    # Debug: Print predictions to verify output
    print("Predictions:", predictions)
    
    # Find the predicted label
    predicted_index = np.argmax(predictions)
    predicted_label = class_labels[predicted_index]
    confidence = float(predictions[0][predicted_index] * 100)  # Convert to percentage and ensure it's a float
    
    # Debug: Print confidence to verify calculation
    print("Predicted Label:", predicted_label)
    print("Answer:", question)
    print("Confidence:", f"{confidence:.2f}%")
    
    # Determine if the prediction matches the question
    status = predicted_label == question
    response = {
        "status": status,
        "percentage": f"{confidence:.2f}%",
        "predicted_label": predicted_label,
        "answer": question
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)