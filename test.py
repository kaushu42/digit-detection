import numpy as np
import cv2

x = np.load('./Images/inputs.npy')
y = np.load('./Labels/labels.npy')
print(x.shape, y.shape)


for i in range(len(x)):
    xmin = int(y[i][1]*28.0)
    ymin = int(y[i][2]*28.0)
    xmax = int(y[i][3]*28.0)
    ymax = int(y[i][4]*28.0)

    cv2.rectangle(x[i], (xmin, ymin), (xmax, ymax), (255, 0, 0), 1)
    print((ymin, xmin), (ymax, xmax))
    cv2.imshow('x', x[i])
    cv2.waitKey()
