import tensorflow as tf
import cv2
import numpy as np

def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

def predict(model, image_path):
    # Decode the image buffer
    # image = cv2.imdecode(np.frombuffer(image_buffer, np.uint8), cv2.IMREAD_COLOR)
    image = cv2.imread(image_path)

    # Resize the image to match the dataset input shape
    image = cv2.resize(image, (128, 128))
    
    # Normalize pixel values to match dataset preprocessing
    image = image.astype(np.float32) / 255.0  # Same as Rescaling(1.0/255)
    
    # Expand dimensions to add a batch dimension
    image = np.expand_dims(image, axis=0)
    
    # Make predictions
    predictions = model.predict(image)
    
    return predictions