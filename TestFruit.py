import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import numpy as np

# Load CSV file
data = pd.read_csv('sample_fruit_dataset.csv')

# Encode the "Color" column
label_encoder_color = LabelEncoder()
data['Color'] = label_encoder_color.fit_transform(data['Color'])

# Encode the "Fruit Type" (label)
label_encoder_fruit = LabelEncoder()
data['Fruit Type'] = label_encoder_fruit.fit_transform(data['Fruit Type'])

# Load the trained model
model = joblib.load('fruit_model.pkl')

# Function to predict new fruit
def predict_fruit(weight, color, diameter):
    try:
        # Encode the color input
        color_encoded = label_encoder_color.transform([color])[0]
    except ValueError:
        print(f"Invalid color '{color}'. Valid colors are: {list(label_encoder_color.classes_)}")
        return None
    
    # Prepare the feature vector
    input_data = pd.DataFrame(np.array([[weight, color_encoded, diameter]]), columns=['Weight', 'Color', 'Diameter'])

    # Predict the fruit type
    predicted_label = model.predict(input_data)[0]
    fruit_name = label_encoder_fruit.inverse_transform([predicted_label])[0]

    return fruit_name

# User input
print("Enter the fruit details:")
try:
    weight = float(input("Weight (grams): "))
    color = input("Color (Red, Yellow, Green, Orange): ").capitalize()
    diameter = float(input("Diameter (cm): "))

    predicted_fruit = predict_fruit(weight, color, diameter)
    if predicted_fruit:
        print(f"The predicted fruit is: {predicted_fruit}")
except ValueError:
    print("Invalid input! Please enter numerical values for Weight and Diameter.")
