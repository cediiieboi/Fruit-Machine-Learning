import tensorflow as tf
from sklearn.svm import SVC
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

# Extract images and labels from the datasets
def extract_images_and_labels(dataset):
    images = []
    labels = []
    for img_batch, label_batch in dataset:
        images.append(img_batch.numpy())
        labels.append(label_batch.numpy())
    images = np.concatenate(images)
    labels = np.concatenate(labels)
    return images, labels

train_images, train_labels = extract_images_and_labels(train_dataset)
val_images, val_labels = extract_images_and_labels(val_dataset)

# Flatten the images for use with SVM
train_images = train_images.reshape(len(train_images), -1)  # Flatten
val_images = val_images.reshape(len(val_images), -1)        # Flatten

# Normalize pixel values to [0, 1]
train_images = train_images / 255.0
val_images = val_images / 255.0

# Train an SVM classifier
clf = SVC(kernel='linear', random_state=42)  # Using a linear kernel
clf.fit(train_images, train_labels)

# Predict on validation data
y_pred = clf.predict(val_images)

# Print classification report
print("\nClassification Report:")
print(classification_report(val_labels, y_pred, target_names=class_names))
