import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

# Load CSV file
data = pd.read_csv('sample_fruit_dataset.csv')

# Encode the "Color" column
label_encoder = LabelEncoder()
data['Color'] = label_encoder.fit_transform(data['Color'])

# Encode the "Fruit Type" (label)
data['Fruit Type'] = label_encoder.fit_transform(data['Fruit Type'])

# Split the features and labels
X = data[['Weight', 'Color', 'Diameter']]
y = data['Fruit Type']


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

print("LOADING LOADING LOADING...")

# Model evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

joblib.dump(model, 'fruit_model.pkl')
print("Model saved as fruit_model.pkl")
