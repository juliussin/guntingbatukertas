import os
import random
import shutil
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def split_data(source, training, testing, split_size):
    list_of_files = os.listdir(source)
    if training[-1] != '/':
        training = training + '/'
    if testing[-1] != '/':
        testing = testing + '/'
    training_list = random.sample(list_of_files, int(len(list_of_files)*split_size))
    testing_list = (x for x in list_of_files if x not in training_list)
    for name in training_list:
        shutil.copyfile(source + name, training + name)
    for name in testing_list:
        shutil.copyfile(source + name, testing + name)


def ask_y_or_n(asking_phrase):
    answer = input('\n\n\n' + asking_phrase + ' (Y/N): ')
    while answer != 'Y' and answer != 'N' and answer != 'y' and answer != 'n':
        answer = input('Input only Y/N! ')
    if answer == 'Y' or answer == 'y':
        answer_bool = True
    elif answer == 'N' or answer == 'n':
        answer_bool = False
    else:
        answer_bool = None
    return answer_bool


gunting_cap = 'capture/gunting/'
batu_cap = 'capture/batu/'
kertas_cap = 'capture/kertas/'

# Check Capture Directory
error_capture_dir = False
if (not os.path.exists(gunting_cap)) or (not os.path.exists(batu_cap)) or (not os.path.exists(kertas_cap)):
    print("\n\n\nCapture directory should contain gunting, batu, and kertas directories!")
    exit()
if len(os.listdir(gunting_cap)) < 5:
    print("\n\n\nGunting directory should contain at least 5 images!")
    error_capture_dir = True
if len(os.listdir(batu_cap)) < 5:
    print("\n\n\nBatu directory should contain at least 5 images!")
    error_capture_dir = True
if len(os.listdir(kertas_cap)) < 5:
    print("\n\n\nKertas directory should contain at least 5 images!")
    error_capture_dir = True
if error_capture_dir:
    exit()

# Print Number of Images
print('Total Gunting Images: ', len(os.listdir(gunting_cap)))
print('Total Batu    Images: ', len(os.listdir(batu_cap)))
print('Total Kertas  Images: ', len(os.listdir(kertas_cap)))

# If there is no images

# List of Directories
training_dir = 'dataset/training/'
testing_dir = 'dataset/testing/'
training_gunting_dir = os.path.join(training_dir, 'gunting/')
testing_gunting_dir = os.path.join(testing_dir, 'gunting/')
training_batu_dir = os.path.join(training_dir, 'batu/')
testing_batu_dir = os.path.join(testing_dir, 'batu/')
training_kertas_dir = os.path.join(training_dir, 'kertas/')
testing_kertas_dir = os.path.join(testing_dir, 'kertas/')

# Create or Overwrite Existing Directories
do_overwrite = ask_y_or_n('Overwrite Existing Training & Testing Directories?')
for directory in [training_gunting_dir, testing_gunting_dir,
                  training_batu_dir, testing_batu_dir,
                  training_kertas_dir, testing_kertas_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)
    elif do_overwrite:
        shutil.rmtree(directory)
        os.makedirs(directory)

# Split Dataset into Training and Testing
do_split = ask_y_or_n('Split Training & Testing? N if already exist!')
if do_split:
    split_size = int(input('Split size (in percent): '))
    split_size = split_size/100.0
    split_data(gunting_cap, training_gunting_dir, testing_gunting_dir, split_size)
    split_data(batu_cap,training_batu_dir,testing_batu_dir, split_size)
    split_data(kertas_cap, training_kertas_dir, testing_kertas_dir, split_size)
    # Print Number of Testing & Training Images
    print('Gunting Training/Testing: ', len(os.listdir(training_gunting_dir)),
          '/', len(os.listdir(testing_gunting_dir)))
    print('Batu    Training/Testing: ', len(os.listdir(training_batu_dir)),
          '/', len(os.listdir(testing_batu_dir)))
    print('Kertas  Training/Testing: ', len(os.listdir(training_kertas_dir)),
          '/', len(os.listdir(testing_kertas_dir)))

# Keras Data Generator
training_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=45,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
training_generator = training_datagen.flow_from_directory(training_dir,
                                                          batch_size=10,
                                                          class_mode='categorical',
                                                          target_size=(150, 150))
validation_datagen = ImageDataGenerator(rescale=1.0/255.0)
validation_generator = validation_datagen.flow_from_directory(testing_dir,
                                                              batch_size=10,
                                                              class_mode='category',
                                                              target_size=(150, 150))

# Define Keras Model to Classify Gunting-Batu-Kertas
input_shape = (150, 150, 3)
kernel_size = (3, 3)
maxpooling_size = (2, 2)
model = tf.keras.models.Sequential([
    # First Convolution
    tf.keras.layers.Conv2D(64, kernel_size, activation='relu', input_shape=input_shape),
    tf.keras.layers.MaxPooling2D(maxpooling_size),
    # Second Convolution
    tf.keras.layers.Conv2D(64, kernel_size, activation='relu'),
    tf.keras.layers.MaxPooling2D(maxpooling_size),
    # Third Convolution
    tf.keras.layers.Conv2D(128, kernel_size, activation='relu'),
    tf.keras.layers.MaxPooling2D(maxpooling_size),
    # Fourth Convolution
    tf.keras.layers.Conv2D(128, kernel_size, activation='relu'),
    tf.keras.layers.MaxPooling2D(maxpooling_size),
    # Flatten the Results to Feed into DNN
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    # Hidden Layers
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    # Output Layers for Categorical Crossentropy
    tf.keras.layers.Dense(3, activation='softmax')
])
model.summary()
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Training Process
history = model.fit(training_generator,
                    epochs=100,
                    steps_per_epoch=20,
                    validation_data=validation_generator,
                    validation_steps=5,
                    verbose=1)

do_save = ask_y_or_n('Save Model?')
if do_save:
    h5name = input('\n\n\nInput H5 File Name: ')
    model.save(h5name+'.h5')



