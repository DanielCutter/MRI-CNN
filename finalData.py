
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import warnings
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers import BatchNormalization

warnings.simplefilter(action='ignore', category=FutureWarning)

# OS setup and GPU allocation
os.chdir('C:/Users/dancu/PycharmProjects/firstCNN/data/ad-vs-cn')
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Define paths for image data
train_path = "C:/Users/dancu/PycharmProjects/firstCNN\data/ad-vs-cn/train"
valid_path = "C:/Users/dancu/PycharmProjects/firstCNN\data/ad-vs-cn/valid"

# Use ImageDataGenerator to create 3 lots of batches
train_batches = ImageDataGenerator(
    rescale=1/255, width_shift_range=.05, height_shift_range=.05).flow_from_directory(directory=train_path,
        target_size=(160,160), classes=['cn', 'ad'], batch_size=30,
            color_mode="rgb")
valid_batches = ImageDataGenerator(
    rescale=1/255, width_shift_range=.05, height_shift_range=.05).flow_from_directory(directory=valid_path,
        target_size=(160,160), classes=['cn', 'ad'], batch_size=30,
            color_mode="rgb")

# Store a batch of labelled training data
imgs, labels = next(train_batches)

# Test to see normalisation has occurred properly
# print(imgs[1][8])

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

# Define the model

# VGG16 - 50% accuracy, 50% val accuracy
# model = Sequential([
# Conv2D(input_shape=(160,160,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"),
# Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"),
# MaxPool2D(pool_size=(2,2),strides=(2,2)),
# Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"),
# Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"),
# MaxPool2D(pool_size=(2,2),strides=(2,2)),
# Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"),
# Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"),
# Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"),
# MaxPool2D(pool_size=(2,2),strides=(2,2)),
# Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
# Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
# Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
# MaxPool2D(pool_size=(2,2),strides=(2,2)),
# Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
# Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
# Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
# MaxPool2D(pool_size=(2,2),strides=(2,2)),
# Flatten(),
# Dense(units=1024,activation="relu"),
# Dense(units=128,activation="relu"),
# Dense(units=2, activation="softmax"),
# ])

# Model 1 - 79% accuracy, 57% val accuracy
model = Sequential([
    Conv2D(filters=32, kernel_size=(5, 5), activation='relu', padding = 'same', input_shape=(160,160,3)),
    Conv2D(filters=32, kernel_size=(5, 5), activation='relu', padding='same'),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Flatten(),
    Dense(units=2, activation='softmax')
])

# Another simplistic model - 73% accuracy, 52% val accuracy
# model = Sequential([
#     Conv2D(filters=32, kernel_size=(5, 5), activation='relu', padding = 'same', input_shape=(160,160,3)),
#     MaxPool2D(pool_size=(2, 2), strides=2),
#     Dropout(0.1),
#     Conv2D(filters=32, kernel_size=(5, 5), activation='relu', padding = 'same'),
#     MaxPool2D(pool_size=(2, 2), strides=2),
#     Dropout(0.1),
#     Flatten(),
#     Dense(units=2, activation='softmax')
# ])

# # Simple model with dropout - 80% accuracy, 53% val acc
# model = Sequential([
#     Conv2D(filters=32, kernel_size=(5, 5), activation='relu', padding = 'same', input_shape=(160,160,3)),
#     MaxPool2D(pool_size=(3, 3), strides=3),
#     Dropout(0.2),
#     Flatten(),
#     Dense(units=2, activation='softmax')
# ])

# No dropouts - 75% accuracy, 54% val acc
# model = Sequential([
#     Conv2D(filters=32, kernel_size=(5, 5), activation='relu', padding = 'same', input_shape=(160,160,3)),
#     Conv2D(filters=32, kernel_size=(5, 5), activation='relu', padding='same'),
#     MaxPool2D(pool_size=(2, 2), strides=2),
#     Conv2D(filters=32, kernel_size=(5, 5), activation='relu', padding='same'),
#     Flatten(),
#     Dense(units=2, activation='softmax')
# ])

# First model with dropouts - 75% accuracy, 52% val
# model = Sequential([
#     Conv2D(filters=32, kernel_size=(5, 5), activation='relu', padding = 'same', input_shape=(160,160,3)),
#     MaxPool2D(pool_size=(2, 2), strides=2),
#     Dropout(0.1),
#     Conv2D(filters=64, kernel_size=(5, 5), activation='relu', padding='same'),
#     MaxPool2D(pool_size=(2, 2), strides=2),
#     Dropout(0.2),
#     Conv2D(filters=128, kernel_size=(5, 5), activation='relu', padding='same'),
#     MaxPool2D(pool_size=(2, 2), strides=2),
#     Dropout(0.3),
#     Flatten(),
#     Dense(units=2, activation='softmax')
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
