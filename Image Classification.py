                                                                                                # Importing Cifar data (10k+ imgs) --> Img Classification
                                                                                        
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dropout, Flatten, Dense, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from skimage.io import imread
from skimage.transform import resize

# CIFAR10 IS A SET OF 60K IMAGES 32 X 32 ON 3 CHANNELS 

IMG_CHANNELS = 3
IMG_ROWS = 32
IMG_COLS = 32

# Constants

BATCH_SIZE = 128
NB_EPOCH = 20
NB_CLASSES = 10
VERBOSE = 1
VALIDATION_SPLIT = 0.2
OPTIM = RMSprop()

# Load dataset

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print("\n")
print("X_train shape:", X_train.shape)
print(X_train.shape[0], "train samples")
print(X_test.shape[0], "test samples")
print("\n")

# Convert to categorical

Y_train = np_utils.to_categorical(y_train, NB_CLASSES)
Y_test = np_utils.to_categorical(y_test, NB_CLASSES)

# Float and normalize

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# Network

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=(IMG_ROWS, IMG_COLS, IMG_CHANNELS), activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(NB_CLASSES))
model.add(Activation('softmax'))
model.summary()

'''
# Train

model.compile(loss='categorical_crossentropy', optimizer=OPTIM, metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs= NB_EPOCH, validation_split=VALIDATION_SPLIT, verbose=VERBOSE)
score = model.evaluate(X_test, Y_test, batch_size=BATCH_SIZE, verbose=VERBOSE)

print("Test score:", score[0])
print("Test accuracy:", score[1])
'''
# Real world image testing

imag_names = ['use a animal img']
imgs = imread('use a animal img')
imgs = resize(imgs, (32, 32))
model.predict(imgs.reshape(1, 32, 32, 3))
model.predict(imgs.reshape(1, 32, 32, 3))
print(imgs)
