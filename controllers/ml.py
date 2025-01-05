import tensorflow as tf
import cv2
import numpy as np

def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

def remove_background(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply a binary threshold to the image
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a mask with the same dimensions as the image
    mask = np.zeros_like(image)
    
    # Draw the contours on the mask
    cv2.drawContours(mask, contours, -1, (255, 255, 255), thickness=cv2.FILLED)
    
    # Bitwise-and the mask with the original image to remove the background
    result = cv2.bitwise_and(image, mask)
    
    return result

def predict(model, image_path):
    # Decode the image buffer
    # image = cv2.imdecode(np.frombuffer(image_buffer, np.uint8), cv2.IMREAD_COLOR)
    image = cv2.imread(image_path)

    # Remove the background from the image
    image = remove_background(image)

    # Resize the image to match the dataset input shape
    image = cv2.resize(image, (128, 128))
    
    # Normalize pixel values to match dataset preprocessing
    image = image.astype(np.float32) / 255.0  # Same as Rescaling(1.0/255)
    
    # Expand dimensions to add a batch dimension
    image = np.expand_dims(image, axis=0)
    
    # Make predictions
    predictions = model.predict(image)
    
    print("Predictions shape:", predictions.shape)

    return predictions