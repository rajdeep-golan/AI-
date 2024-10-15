# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Load the dataset from a CSV file
# Assumes 'cancer.csv' is in the same directory
dataset = pd.read_csv('cancer.csv')

# Split features (x) and target label (y)
# We drop the 'diagnosis(1=m, 0=b)' column from x as it's the target
x = dataset.drop(columns=['diagnosis(1=m, 0=b)'])

# Target variable: 'diagnosis(1=m, 0=b)'
y = dataset['diagnosis(1=m, 0=b)']

# Split the dataset into training and test sets
# 80% training data, 20% test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Define a sequential model using TensorFlow's Keras API
model = tf.keras.models.Sequential()

# Add layers to the model
# First hidden layer with 256 neurons and sigmoid activation
model.add(tf.keras.layers.Dense(256, input_shape=(x_train.shape[1],), activation='sigmoid'))

# Second hidden layer with 256 neurons and sigmoid activation
model.add(tf.keras.layers.Dense(256, activation='sigmoid'))

# Output layer with a single neuron (binary classification) and sigmoid activation
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# Compile the model using Adam optimizer and binary crossentropy loss function
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model with the training data for 1000 epochs
model.fit(x_train, y_train, epochs=1000)

# Evaluate the model's performance on the test data
evaluation = model.evaluate(x_test, y_test)

# Print evaluation metrics: loss and accuracy on test data
print(f"Test Loss: {evaluation[0]}")
print(f"Test Accuracy: {evaluation[1]}")
