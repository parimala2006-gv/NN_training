import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Create model folder
os.makedirs("model", exist_ok=True)

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Convert labels to categorical
y = to_categorical(y)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build Neural Network
model = Sequential()

model.add(Dense(16,
                activation='relu',
                kernel_initializer='he_normal',
                input_shape=(4,)))

model.add(Dense(8,
                activation='relu',
                kernel_initializer='he_normal'))

model.add(Dense(3,
                activation='softmax',
                kernel_initializer='glorot_uniform'))

# Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
history = model.fit(
    X_train,
    y_train,
    epochs=100,
    validation_split=0.4,
    verbose=1
)

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)

# Save model as PKL
with open("model/iris_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as model/iris_model.pkl")

# Plot accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'])
plt.show()

# Plot loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'])
plt.show()