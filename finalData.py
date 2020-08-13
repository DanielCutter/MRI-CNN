
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import warnings
import matplotlib.pyplot as plt
import cv2
import numpy as np

warnings.simplefilter(action='ignore', category=FutureWarning)

os.chdir('C:/Users/dancu/PycharmProjects/firstCNN/data/ad-vs-cn')

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Define paths for image data
train_path = "C:/Users/dancu/PycharmProjects/firstCNN\data/ad-vs-cn/train"
test_path = "C:/Users/dancu/PycharmProjects/firstCNN\data/ad-vs-cn/test"
valid_path = "C:/Users/dancu/PycharmProjects/firstCNN\data/ad-vs-cn/valid"

# Use ImageDataGenerator to create 3 lots of batches
train_batches = ImageDataGenerator(
    rescale=1/255).flow_from_directory(directory=train_path,
        target_size=(160,160), classes=['cn_proc', 'ad_proc'], batch_size=20,
            color_mode="rgb")
valid_batches = ImageDataGenerator(
    rescale=1/255).flow_from_directory(directory=valid_path,
        target_size=(160,160), classes=['cn_proc', 'ad_proc'], batch_size=20,
            color_mode="rgb")
# test_batches = ImageDataGenerator(
#     rescale=1/255).flow_from_directory(directory=test_path,
#         target_size=(224,224), classes=['cn', 'ad'], batch_size=10,
#             color_mode="rgb")

imgs, labels = next(train_batches)

# Test to see normalisation has occurred properly
print(imgs[1][16])

# Define method to plot MRIs
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 10, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# Plot a sample of MRIs
plotImages(imgs)

# # Define the model
# # VGG16
# model = Sequential()
# model.add(Conv2D(input_shape=(160,160,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
# model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
# model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
# model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
# model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
# model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
# model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
# model.add(Flatten())
# model.add(Dense(units=1024,activation="relu"))
# model.add(Dense(units=128,activation="relu"))
# model.add(Dense(units=2, activation="softmax"))

# # This model hits around 70% train acc, 60% val acc
model = Sequential([
    Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding = 'same', input_shape=(160,160,3)),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Flatten(),
    Dense(units=2, activation='softmax')
])

## Basic model with dropouts
# model = Sequential([
#     Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding = 'same', input_shape=(224,224,3)),
#     MaxPool2D(pool_size=(2, 2), strides=2),
#     Dropout(0.1),
#     Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
#     MaxPool2D(pool_size=(2, 2), strides=2),
#     Dropout(0.2),
#     Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'),
#     MaxPool2D(pool_size=(2, 2), strides=2),
#     Dropout(0.3),
#     Flatten(),
#     Dense(units=1, activation='sigmoid')
# ])

# Summarise each layer of the model
print(model.summary())

# Compile and train the model
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x=train_batches,
    steps_per_epoch=len(train_batches),
    validation_data=valid_batches,
    validation_steps=len(valid_batches),
    epochs=20,
    verbose=1
)
