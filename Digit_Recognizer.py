import numpy as np
from matplotlib import pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import confusion_matrix

# %matplotlib inline

# Importing the Dataset

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("X-Train shape :", x_train.shape)
print("Y-train shape :", y_train.shape)
print("X-Test Shape :", x_test.shape)
print("Y-Test Shape:", y_test.shape)

# Exploring the images
# Example_1
plt.imshow(x_train[0], cmap="binary")
plt.show()
# Example_2
plt.imshow(x_train[54], cmap="binary")
plt.show()

# Exploring the Output data
print(set(y_train))

# One Hot Encoding
# 0 : [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
y_train_encoded = to_categorical(y_train)
y_test_encoded = to_categorical(y_test)

# Converting to NP Arrays, From matrix to Vector(1D array) (28 * 28) = 784
x_train_reshaped = np.reshape(x_train, (60000, 784))
x_test_reshaped = np.reshape(x_test, (10000, 784))

# Normalizing data for faster calculations
x_mean = np.mean(x_train_reshaped)
x_std = np.std(x_train_reshaped)

epsilon = 1e-10

x_train_norm = (x_train_reshaped - x_mean) / (x_std + epsilon)
x_test_norm = (x_test_reshaped - x_mean) / (x_std + epsilon)

# BUILDING THE MODEL
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Creating the model
model.compile(
    optimizer='sgd',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# Training the Model
model.fit(x_train_norm, y_train_encoded, epochs=10)

# Model Evaluation
loss, accuracy = model.evaluate(x_test_norm, y_test_encoded)
print("Test Set Accuracy :", accuracy * 100)
print("Loss :", loss)

# Prediction
predictions = model.predict(x_test_norm)
test_data_len = len(x_test_norm)

predicted_values = []
for i in range(test_data_len):
    predicted_values.append(np.argmax(predictions[i]))

conf_mat = confusion_matrix(y_test, predicted_values)
correct_predictions, wrong_predictions = 0, 0

for i in range(10):
    for j in range(10):
        if i == j:
            correct_predictions = correct_predictions + conf_mat[i][j]
        else:
            wrong_predictions = wrong_predictions + conf_mat[i][j]

print("Correct Predictions :", correct_predictions)
print("Incorrect Predictions :", wrong_predictions)
