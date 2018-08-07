import os

import keras.datasets.mnist as m
import cv2


# for i in range(10):
#     os.mkdir('./Images/%d'%i)

(x_train, y_train), (x_test, y_test) = m.load_data()

# counts = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}
count = 0
for i in range(10000):
    if y_train[i] == 1:
        cv2.imwrite('./Images/Data0/%d.jpg'%(count), x_train[i].reshape(28, 28, 1))
        count = count + 1
    # counts[y_train[i]] += 1
    # print('./Images/%d/%d.jpg'%(y_train[i], counts[y_train[i]]))
