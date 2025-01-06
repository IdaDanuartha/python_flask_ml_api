from flask import Flask, request, jsonify
from controllers.ml import load_model, predict
import numpy as np
import os

app = Flask(__name__)

# Load the ML model at startup
model = load_model('SignLanguage.h5')

# Folder for save temporary files
UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define class labels for predictions
class_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

@app.route('/questions', methods=['GET'])
def get_questions():
    questions = [
        {"question": label, "hint": f"/static/sign_language_dataset/{label}.jpeg"} for label in class_labels
    ]
    return jsonify({"questions": questions})

@app.route('/answers', methods=['POST'])
def post_answers():
    # Get the question and image data from the request
    if 'question' not in request.form or 'image' not in request.files:
        return jsonify({"error": "Question or image file is required"}), 400
    
    question = request.form['question']
    image_file = request.files['image']

    if image_file:
        # Save file to upload folder
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
        image_file.save(file_path)

        # Make predictions
        predictions = predict(model, file_path)
        
        # Debug: Print predictions to verify output
        print("Predictions:", predictions)
        
        # Check if predictions are empty
        if predictions.size == 0:
            return jsonify({"error": "Model did not return any predictions"}), 500
        
        # Find the predicted label
        predicted_index = np.argmax(predictions)
        predicted_label = class_labels[predicted_index]
        confidence = float(predictions[0][predicted_index] * 100)  # Convert to percentage and ensure it's a float
        
        # Debug: Print confidence to verify calculation
        print("Predicted Label:", predicted_label)
        print("Answer:", question)
        print("Percentage:", f"{confidence:.2f}%")
        
        # Determine if the prediction matches the question
        # status = predicted_label == question

        # Validate confidence percentage
        if confidence < 80:
            predicted_label = "Your answer is unpredictable"
            status = False
        else:
            status = True


        response = {
            "status": status,
            "percentage": f"{confidence:.2f}%",
            "predicted_label": predicted_label,
            "answer": question
        }
        
        # Remove file after prediction
        os.remove(file_path)
        
        return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, port=5001)