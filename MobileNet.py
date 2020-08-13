import os
from PIL import Image
import glob
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2


poketmon = ["Squirtle", "Digda", "Dragonite", "Modafi", "", "", "Snorlax", "Charmander", "Purin", "Pikachu"]
img_path = "D:\\Gongmo\\It_worked\\img\\"

label_list = os.listdir(img_path)
label = [i for i in range(10)]

x_train = []
x_label = []

y_verfi = []
y_label = []

z_test = []
z_label = []
for b in label:

    directory = img_path + str(b) + "\\" + poketmon[int(b)] + "_pp\\"
    files = glob.glob(directory + "*.jpg")
    for i, f in enumerate(files):
        img = Image.open(f)
        data = np.asarray(img) / 255
        if i <= 599:
            x_train.append(data)
            x_label.append(b)
        elif 599 < i and i <= 799:
            y_verfi.append(data)
            y_label.append(b)
        elif i < 900 and i > 799:
            z_test.append(data)
            z_label.append(b)
        else:
            break

print(len(x_train))
print(len(x_label))
print(len(y_verfi))
print(len(y_label))
print(len(z_test))
print(len(z_label))