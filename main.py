import sys

import numpy as np

import keras
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.utils import to_categorical

mode = 'test'
try:
    if sys.argv[1] == 'test':
        mode = 'test'
    elif sys.argv[1] == 'train':
        mode = 'train'
    else:
        raise Exception('Enter valid argument')
except IndexError as e:
    pass
if mode == 'train':
    ROWS, COLS = 28, 28
    CLASSES = 10
    TRAIN_DATA = 60000
    TEST_DATA = 10000

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(TRAIN_DATA, ROWS, COLS, 1)*1/255
    x_test = x_test.reshape(TEST_DATA, ROWS, COLS, 1)*1/255
    y_test = (y_test == 1)

    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', input_shape=(28, 28, 1)))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=5, kernel_size=(1, 1), activation=None))
    # model.add(Flatten())
    # model.add(Dense(10, activation='relu'))
    # model.add(Dense(1, activation='sigmoid'))
    print(model.summary())
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # model.fit(x_test, y_test, epochs=5, batch_size=128, verbose=1)
    # model.save('./weights/model.h5')
else:
    try:
        model = load_model('./weights/model.h5')
        print('\n*****Model Loaded*****')
    except:
        raise Exception('Cannot Load Model file, train first')
    data = np.load('test.npz')
    score = model.evaluate(data['arr_0'], data['arr_1'])
    print(score)
