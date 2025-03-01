import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau

# Define the improved CNN model
model = Sequential([
    # First Conv Block
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    BatchNormalization(),
    MaxPooling2D(2,2),

    # Second Conv Block
    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    # Third Conv Block (Added an extra layer for better feature learning)
    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    # Flattening
    Flatten(),

    # Fully Connected Layers
    Dense(256, activation='relu'),  # Increased neurons
    Dropout(0.5),  # Dropout to reduce overfitting
    Dense(128, activation='relu'),
    Dropout(0.3),
    
    # Output Layer (4 classes)
    Dense(6, activation='softmax')
])

# Compile the model with Adam optimizer
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Learning rate scheduler to reduce LR when validation loss plateaus
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, verbose=1)

# Define enhanced data augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20, 
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Load training data
train_data = datagen.flow_from_directory(
    '../train_dataset/train', target_size=(128, 128), batch_size=32, class_mode='categorical', subset='training')

# Load validation data
val_data = datagen.flow_from_directory(
    '../train_dataset/train', target_size=(128, 128), batch_size=32, class_mode='categorical', subset='validation')

# Train the model with learning rate scheduler
model.fit(train_data, validation_data=val_data, epochs=10, callbacks=[lr_scheduler])

# Save the trained model
model.save("../models/skin_analysis_model_2.h5")
print("✅ Model Saved: skin_analysis_model_2.h5")