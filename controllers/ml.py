import tensorflow as tf
import cv2
import numpy as np

def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

def predict(model, image_buffer):
    # Decode the image buffer
    image = cv2.imdecode(np.frombuffer(image_buffer, np.uint8), cv2.IMREAD_COLOR)

    # Resize the image to match the dataset input shape
    image = cv2.resize(image, (128, 128))
    
    # Normalize pixel values to match dataset preprocessing
    image = image.astype(np.float32) / 255.0  # Same as Rescaling(1.0/255)
    
    # Expand dimensions to add a batch dimension
    image = np.expand_dims(image, axis=0)
    
    # Make predictions
    predictions = model.predict(image)
    
    return predictions

# def predict(model, image_buffer):
#     # Decode the image buffer
#     image = cv2.imdecode(np.frombuffer(image_buffer, np.uint8), cv2.IMREAD_COLOR)

#     # Convert to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Apply a threshold to create a binary mask
#     _, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

#     # Invert the mask
#     mask_inv = cv2.bitwise_not(mask)

#     # Create a black background
#     background = np.zeros_like(image)

#     # Use the mask to change the background to black
#     image = cv2.bitwise_and(image, image, mask=mask_inv)
#     image = cv2.add(image, background, mask=mask)

#     # Resize the image to match the dataset input shape
#     image = cv2.resize(image, (128, 128))
    
#     # Normalize pixel values to match dataset preprocessing
#     image = image.astype(np.float32) / 255.0  # Same as Rescaling(1.0/255)
    
#     # Expand dimensions to add a batch dimension
#     image = np.expand_dims(image, axis=0)
    
#     # Make predictions
#     predictions = model.predict(image)
#     return predictions
