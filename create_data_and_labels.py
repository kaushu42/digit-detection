
import os
import xml.etree.ElementTree as e

import numpy as np
import cv2

PATH = './Labels/dataxml/'
PATH_IMAGE = './Images/Data0/'
# Get all files in the directory of folder Labels/dataxml
files = [x for x in os.listdir(PATH) if x.endswith('.xml')]
print(len(files), 'files found')

images = []
labels = []
for file in files:
    xml = e.parse(PATH + file).getroot()
    # print(PATH + file)
    name = xml[-1][0]

    label = [int(name.text)]
    for x in xml[-1][-1]: #<object> is the last element of <annotation> and <bndbox> is the last element of <object>
        label.append(int(x.text))
    label = np.array(label, dtype=float)
    label[1:] = label[1:]/28
    labels.append(label)

    x = cv2.imread(PATH_IMAGE + file.replace('.xml', '.jpg'))[:, :, 0]
    images.append(x.reshape(28, 28, 1))

print(np.asarray(images).shape)
np.save('./Images/inputs.npy', np.asarray(images))
print(np.asarray(labels).shape)
np.save('./Labels/labels.npy', np.asarray(labels))
print(np.asarray(labels)*28)
