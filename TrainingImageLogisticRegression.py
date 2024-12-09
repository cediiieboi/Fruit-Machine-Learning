import tensorflow as tf
from sklearn.metrics import classification_report
import numpy as np

# Path to the dataset
dataset_dir = r'C:\Users\Cedie\Documents\4th Year\1st Semester\ML\Activity 2, 3 & 4\dataset'

# Load the dataset and split into training and validation sets
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(224, 224),
    batch_size=32
)

val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(224, 224),
    batch_size=32
)

# Class names (Watermelon, Strawberry, Tomato)
class_names = train_dataset.class_names
print(f"Class Names: {class_names}")

# Normalize the pixel values to [0, 1]
train_dataset = train_dataset.map(lambda x, y: (x / 255.0, y))
val_dataset = val_dataset.map(lambda x, y: (x / 255.0, y))

# Create a Logistic Regression model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(224, 224, 3)),  # Flatten image to a vector
    tf.keras.layers.Dense(len(class_names), activation='softmax')  # Logistic Regression
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
print("Starting training...")
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10
)

# Save the trained model
model.save('fruit_classifier_logistic_model.h5')
print("Model saved as fruit_classifier_logistic_model.h5")

# Evaluate on validation dataset
y_true = []
y_pred = []

for images, labels in val_dataset:
    predictions = model.predict(images)
    y_true.extend(labels.numpy())  
    y_pred.extend(np.argmax(predictions, axis=1))

# Print classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))
