import tensorflow as tf
import os
from keras.utils import img_to_array, load_img
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator # pyright: ignore[reportMissingImports]
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt


train_path ="fruits-360_100x100/fruits-360/Training/"
test_path ="fruits-360_100x100/fruits-360/Test/"
BatchSize = 64

img = load_img(train_path + "Quince 1/0_100.jpg")


imgA = img_to_array(img)
print(imgA.shape)

model = Sequential()
#blok_1
model.add(Conv2D(filters =16,kernel_size=3,activation="relu",input_shape=(100,100,3)))
model.add(MaxPooling2D())
model.add(Dropout(0.2))
#blok_2
model.add(Conv2D(filters =64,kernel_size=3,activation="relu"))
model.add(MaxPooling2D())
model.add(Dropout(0.3))
#blok_3
model.add(Conv2D(filters =128,kernel_size=3,activation="relu"))
model.add(MaxPooling2D())
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(256,activation="relu"))
model.add(Dropout(0.4))
model.add(Dense(227,activation="softmax"))

print(model.summary())

model.compile(loss = "categorical_crossentropy",optimizer="adam",metrics=["accuracy"])

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range=0.2,
                                   zoom_range=0.1,
                                   horizontal_flip=True,
                                   vertical_flip=True)


test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(train_path,
                                                    target_size=(100,100),
                                                    batch_size=BatchSize,
                                                    class_mode="categorical",
                                                    color_mode="rgb",
                                                    shuffle=True)

test_generator = test_datagen.flow_from_directory(test_path,
                                                  target_size=(100,100),
                                                  batch_size=BatchSize,
                                                  class_mode="categorical",
                                                  color_mode="rgb")

stepsPerEpoch = np.ceil(train_generator.samples / BatchSize).astype(int)
ValidationSteps = np.ceil(test_generator.samples / BatchSize).astype(int)

stop_early = EarlyStopping(monitor="val_accuracy", patience=5)

history = model.fit(train_generator,
                    steps_per_epoch=stepsPerEpoch,
                    epochs=30,
                    validation_data=test_generator,
                    validation_steps=ValidationSteps,
                    callbacks=[stop_early])



