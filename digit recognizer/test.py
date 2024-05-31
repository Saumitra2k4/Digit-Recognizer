import tensorflow as tf
import cv2
import time
import pyscreenshot as ImageGrab
from pathlib import Path
import numpy as np 

# Load model
model = tf.keras.models.load_model('digit_recognizer.h5')
model.summary()

# Set images folder
images_folder = Path("img/")

# Create images folder if it doesn't exist
if not images_folder.exists():
    images_folder.mkdir()

# Set delay between iterations
delay = 0.1

while True:
    try:
        # Capture screenshot
        img = ImageGrab.grab(bbox=(60, 170, 400, 500))
        # Save screenshot to file
        img_path = images_folder / "img.png"
        img.save(img_path)
        # Read image using OpenCV
        im = cv2.imread(str(img_path))
        # Resize image
        im_resized = cv2.resize(im, (28, 28))
        # Convert image to grayscale
        im_gray = cv2.cvtColor(im_resized, cv2.COLOR_BGR2GRAY)
        # Normalize pixel values
        im_gray_normalized = im_gray / 255.0
        # Reshape image for model input
        im_gray_final = im_gray_normalized.reshape((1, 28, 28, 1))
        # Display resized image
        cv2.imshow("Resized Image", im_resized)
        cv2.waitKey(1)
        # Make prediction
        predictions = model.predict(im_gray_final)
        print("Predictions:", predictions)
        # Display prediction on image if confidence is above 10%
        if np.max(predictions) > 0.1:
            cv2.putText(im_resized, f"Prediction is: {np.argmax(predictions)}", (20, 20), 0, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        # Display image
        cv2.imshow("Result", im_resized)
        # Exit on Enter key press
        if cv2.waitKey(1) == 13:
            break
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Add delay between iterations
        time.sleep(delay)

cv2.destroyAllWindows()
