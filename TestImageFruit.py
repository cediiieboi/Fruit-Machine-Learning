import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import os

# Dynamically load class names from the dataset directory
dataset_dir = r'C:\Users\Cedie\Documents\4th Year\1st Semester\ML\Activity 2\dataset'

# Load the trained model
model = load_model('fruit_classifier_model.h5')

# Class names (must match the classes in the training dataset)
class_names = sorted(os.listdir(dataset_dir))

# Function to preprocess the image and make a prediction
def predict_fruit(image_path):
    try:
        # Load the image
        img = image.load_img(image_path, target_size=(224, 224))  # Resize to match the input size of the model
        img_array = image.img_to_array(img) / 255.0  # Normalize pixel values to [0, 1]
        img_array = np.expand_dims(img_array, axis=0)  # Add a batch dimension

        # Make a prediction
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions)  # Get the index of the highest probability
        predicted_class = class_names[predicted_index]
        confidence = predictions[0][predicted_index]

        return predicted_class, confidence
    except Exception as e:
        return f"Error: {str(e)}", None

# Function to upload an image and display the prediction
def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if not file_path:
        return
    
    # Display the selected image
    img = Image.open(file_path)
    img = img.resize((200, 200))  # Resize for display
    img = ImageTk.PhotoImage(img)
    image_label.config(image=img)
    image_label.image = img  # Keep a reference to avoid garbage collection

    # Predict the fruit type
    predicted_class, confidence = predict_fruit(file_path)
    if confidence:
        result_label.config(text=f"Predicted: {predicted_class}\nConfidence: {confidence:.2f}")
    else:
        result_label.config(text=predicted_class)  # Display the error message

# Create the GUI application
root = tk.Tk()
root.title("Fruit Classifier")

# GUI Elements
instruction_label = Label(root, text="Upload an image of a fruit to classify it!", font=("Arial", 14))
instruction_label.pack(pady=10)

upload_button = Button(root, text="Upload Image", command=upload_image, font=("Arial", 12))
upload_button.pack(pady=10)

image_label = Label(root)  # To display the uploaded image
image_label.pack(pady=10)

result_label = Label(root, text="", font=("Arial", 14), fg="blue")
result_label.pack(pady=10)

# Run the application
root.mainloop()

