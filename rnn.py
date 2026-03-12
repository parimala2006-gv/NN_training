# Import required libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb

# Step 1: Define vocabulary size
vocab_size = 10000

# Step 2: Load IMDB dataset
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

# Step 3: Define maximum review length
max_length = 100

# Step 4: Pad sequences to make equal length
x_train = pad_sequences(x_train, maxlen=max_length)
x_test = pad_sequences(x_test, maxlen=max_length)

# Step 5: Build the RNN model
model = Sequential()

# Embedding layer
model.add(Embedding(input_dim=vocab_size, output_dim=32, input_length=max_length))

# RNN layer
model.add(SimpleRNN(32))

# Output layer
model.add(Dense(1, activation='sigmoid'))

# Step 6: Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Step 7: Display model summary
model.summary()

# Step 8: Train the model
model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.2
)

# Step 9: Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)

print("Test Accuracy:", accuracy)

# Step 10: Make predictions
predictions = model.predict(x_test[:5])

print("Predictions:", predictions)
print("Actual Values:", y_test[:5])