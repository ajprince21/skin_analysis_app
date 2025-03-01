import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Define parameters
img_size = 224  # MobileNetV2 input size
batch_size = 32
epochs = 10

# Define paths
data_dir = "../train_dataset/train/acne_type/"  # Ensure dataset is organized as dataset/train & dataset/val
# Image augmentation for training
datagen = ImageDataGenerator(
    rescale=1.0 / 255, rotation_range=20, width_shift_range=0.2,
    height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
    horizontal_flip=True, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    data_dir + 'train', target_size=(img_size, img_size), batch_size=batch_size, class_mode='categorical')

val_generator = datagen.flow_from_directory(
    data_dir + 'val', target_size=(img_size, img_size), batch_size=batch_size, class_mode='categorical')

# Load MobileNetV2 as base model
base_model = keras.applications.MobileNetV2(input_shape=(img_size, img_size, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze base model weights

# Build model
model = Sequential([
    base_model,
    keras.layers.GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(train_generator.num_classes, activation='softmax')  # Number of acne types/severity levels
])
# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(train_generator, validation_data=val_generator, epochs=epochs)

# Save model
model.save("../models/acne_classification_model.h5")