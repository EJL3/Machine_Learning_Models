                    #Activation function : sigmoid vs relu


from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import numpy as np

(X_train, y_train), (X_val, y_val) = mnist.load_data()

print(X_val.shape)
print(y_val.shape)

# Normalize

X_train = X_train.astype('float32') / 255  # dividing by 255 to normalize the pixel values between 0 and 1
X_val = X_val.astype('float32') / 255

# One-hot encoding

n_classes = 10
y_train = to_categorical(y_train, n_classes)
y_val = to_categorical(y_val, n_classes)

print(y_train.shape)
print(X_train.shape)

# Flatten the data

X_train = np.reshape(X_train, (60000, 784))
X_val = np.reshape(X_val, (10000, 784))

# Testing with Sigmoid

model_sigmoid = Sequential()
model_sigmoid.add(Dense(700, input_dim=784, activation='sigmoid'))
model_sigmoid.add(Dense(350, activation='sigmoid'))
model_sigmoid.add(Dense(100, activation='sigmoid'))
model_sigmoid.add(Dense(10, activation='softmax'))

print("\n")

model_sigmoid.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model_sigmoid.summary()

print("\n")

n_epochs = 10
batch_size = 228
validation_split = 0.2

print("Testing with Sigmoid ⚙")
print("================================================================")

model_sigmoid.fit(X_train, y_train, epochs=n_epochs, batch_size=batch_size, validation_split=validation_split, verbose=2)

print("\n")

# Testing with Relu

model_relu = Sequential()
model_relu.add(Dense(700, input_dim=784, activation='relu'))
model_relu.add(Dense(350, activation='relu'))
model_relu.add(Dense(100, activation='relu'))
model_relu.add(Dense(10, activation='softmax'))

model_relu.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

n_epochs = 10
batch_size = 228
validation_split = 0.2

print("Testing with Relu ⚙")
print("================================================================")

model_sigmoid.fit(X_train, y_train, epochs=n_epochs, batch_size=batch_size, validation_split=validation_split, verbose=2)
