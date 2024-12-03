import shutil
from PIL import Image
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
from tensorflow.keras.preprocessing import image

model = tf.keras.models.load_model('railcnn2.h5')
model.summary()

# Function to classify images and move "flat" images to a separate folder
def classify_and_move_images(source_folder, flat_folder):
    # Create the output directory if it doesn't exist
    os.makedirs(flat_folder, exist_ok=True)

    # Loop through each image in the source folder
    for filename in os.listdir(source_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(source_folder, filename)

            # Predict if the image is "flat" or "non-flat"
            result = predict_image(image_path)  # Modify predict_image to return result

            # If the image is classified as "flat", move it to the flat_folder
            if result == 'flat':
                shutil.move(image_path, os.path.join(flat_folder, filename))
                print(f"Moved {filename} to flat images folder.")
            else:
                print(f"Image {filename} is classified as non-flat. Skipping.")


# Modify the predict_image function to return 'flat' or 'no flat' for easier usage
def predict_image(image_path):
    test_image = image.load_img(image_path, target_size=(224, 224))
    test_image = image.img_to_array(test_image)
    test_image = test_image / 255.0  # Normalize the image
    test_image = np.expand_dims(test_image, axis=0)

    result = model.predict(test_image)

    # Return 'flat' or 'no flat' based on the prediction threshold
    if result[0] > 0.3:
        return 'flat'  # Classifies as flat
    else:
        return 'no flat'  # Classifies as non-flat


# Call the function to classify and move images
source_folder = "C:/Users/mailv/OneDrive/Pictures/SourcePics"  # Folder with images to classify
flat_folder = "C:/Users/mailv/OneDrive/Pictures/FlatResults"  # Folder to store "flat" images

classify_and_move_images(source_folder, flat_folder)
