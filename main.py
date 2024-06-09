import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.utils import to_categorical

(trainin_images, trainin_lables), (testing_images, testing_lables) = datasets.cifar10.load_data()
trainin_images, testing_images = trainin_images/255, testing_images/255

class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

for i in range(16):
    plt.subplot(4,4,i+1)
    plt.xticks
    plt.yticks
    plt.imshow(trainin_images[i],cmap=plt.cm.binary)
    plt.xlabel(class_names[trainin_lables[i][0]])


trainin_images = trainin_images[:20000]
trainin_lables = trainin_lables[:20000]
testing_images = testing_images[:4000]
testing_lables = testing_lables[:4000]

trainin_lables = to_categorical(trainin_lables)
testing_lables = to_categorical(testing_lables)

model = models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(32,32,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(trainin_images, trainin_lables, epochs=10, validation_data=(testing_images,testing_lables))

model.save('image_classifier.keras')