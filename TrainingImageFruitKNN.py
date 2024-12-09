import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf

# Load the dataset
dataset_dir = r'C:\Users\Cedie\Documents\4th Year\1st Semester\ML\Activity 2\dataset'

image_size = (224, 224)
batch_size = 32

# Load dataset and extract images and labels
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=image_size,
    batch_size=batch_size
)

val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=image_size,
    batch_size=batch_size
)

class_names = train_dataset.class_names
print(f"Class Names: {class_names}")

# Normalize the pixel values
train_images = []
train_labels = []
for images, labels in train_dataset:
    train_images.append(images.numpy())
    train_labels.append(labels.numpy())

train_images = np.concatenate(train_images)
train_labels = np.concatenate(train_labels)
train_images = train_images / 255.0

val_images = []
val_labels = []
for images, labels in val_dataset:
    val_images.append(images.numpy())
    val_labels.append(labels.numpy())

val_images = np.concatenate(val_images)
val_labels = np.concatenate(val_labels)
val_images = val_images / 255.0

# Flatten the images for KNN
train_images_flat = train_images.reshape(len(train_images), -1)
val_images_flat = val_images.reshape(len(val_images), -1)

# Build and train the KNN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(train_images_flat, train_labels)

# Predict and evaluate
val_predictions = knn.predict(val_images_flat)
print("Classification Report:\n", classification_report(val_labels, val_predictions, target_names=class_names))
print("Accuracy:", accuracy_score(val_labels, val_predictions))

# Generate confusion matrix
conf_matrix = confusion_matrix(val_labels, val_predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_names)

# Visualize confusion matrix
plt.figure(figsize=(8, 8))
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title('Confusion Matrix')
plt.show()