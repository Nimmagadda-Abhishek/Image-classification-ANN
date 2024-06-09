import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from tensorflow.keras import datasets, layers, models

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255, test_images / 255

class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

model = models.load_model('image_classifier.keras')

input_shape = model.input_shape[1:]

img_path = input("Enter the path to the image: ")

img = cv.imread(img_path)

if img is None:
    print("Error: Unable to read image file. Please check the file path and integrity.")
    exit()

plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.title("Input Image")
plt.show()

img = cv.resize(img, (input_shape[0], input_shape[1]))

img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

img = img / 255.0

img = np.array([img])

prediction = model.predict(img)

index = np.argmax(prediction)

print(f"Prediction is {class_names[index]}")

plt.title(f"Predicted class: {class_names[index]}")