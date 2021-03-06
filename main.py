import sys

import numpy as np

import cv2

import keras
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

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

x = np.load('./Images/inputs.npy')
y = np.load('./Labels/labels.npy').reshape(-1, 1, 1, 5)
print(x.shape, y.shape)

if mode == 'train':
    model = Sequential()
    model.add(
        Conv2D(
            filters=16,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='sigmoid',
            input_shape=(28, 28, 1)))
    model.add(
        Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            activation='sigmoid'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(
        Conv2D(
            filters=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='sigmoid'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(
        Conv2D(
            filters=128,
            kernel_size=(5, 5),
            strides=(1, 1),
            activation='sigmoid'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=5, kernel_size=(1, 1), activation='linear'))
    print(model.summary())
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    model.fit(x, y, epochs=10, batch_size=10, verbose=1)
    model.save('./weights/model.h5')
    print('Model saved.')
    print(model.evaluate(x, y))
else:
    try:
        model = load_model('./weights/model.h5')
        print('\n*****Model Loaded*****')
        (x, _), (_, _) = keras.datasets.mnist.load_data()
        x = x[:100, :, :].reshape(100, 28, 28, 1)
    except:
        raise Exception('Cannot Load Model file, train first')

    predictions = model.predict(x).reshape(-1, 5)
    for i in range(x.shape[0]):
        xmin = int(predictions[i][1] * 28)
        ymin = int(predictions[i][2] * 28)
        xmax = int(predictions[i][3] * 28)
        ymax = int(predictions[i][4] * 28)
        print((xmin, ymin), (xmax, ymax))
        #        x[i] = x[i] + np.random.randn(28, 28, 1)
        #        x[i] = cv2.cvtColor(x[i], cv2.COLOR_GRAY2BGR)
        cv2.rectangle(x[i], (xmin, ymin), (xmax, ymax), (100, 100, 1), 1)
        cv2.imshow('Image', x[i])
        cv2.waitKey()
    # (_, _), (x_test, _) = keras.datasets.mnist.load_data()
    # predictions = model.predict(x_test.reshape(-1, 28, 28, 1)).reshape(-1, 5)
    # print(predictions)
    # for i in range(len(y)):
    #     xmin = int(predictions[i][1]*28)
    #     ymin = int(predictions[i][2]*28)
    #     xmax = int(predictions[i][3]*28)
    #     ymax = int(predictions[i][4]*28)
    #     print((xmin, ymin), (xmax, ymax))
    #     cv2.rectangle(x_test[i], (xmin, ymin), (xmax, ymax), (255, 0, 0), 1)
    #     cv2.imshow('Image', x_test[i])
    #     cv2.waitKey()
