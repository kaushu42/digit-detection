import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D

ROWS, COLS = 28, 28
CLASSES = 10
TRAIN_DATA = 60000
TEST_DATA = 10000

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.reshape(TRAIN_DATA, ROWS, COLS, 1)*1/255
x_test = x_test.reshape(TEST_DATA, ROWS, COLS, 1)*1/255

model = Sequential()
model.add(Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=16, kernel_size=(1, 1), activation=None))
print(model.summary())
    
