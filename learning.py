import os
import zipfile
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator


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


def press_any_key():
    answer = input('\n\n\n' + 'Press any key to continue!')
    return True


# Dataset from laurencemoroney.com/rock-paper-scissors-dataset/

do_extract_training = ask_y_or_n('Extract Training Data?')
if do_extract_training:
    local_zip = 'rps.zip'  # Training Data Location
    zip_ref = zipfile.ZipFile(local_zip, 'r')
    zip_ref.extractall('temp/')
    zip_ref.close()

do_extract_testing = ask_y_or_n('Extract Testing Data?')
if do_extract_testing:
    local_zip = 'rps-test-set.zip'  # Testing Data Location
    zip_ref = zipfile.ZipFile(local_zip, 'r')
    zip_ref.extractall('temp/')
    zip_ref.close()

gunting_dir = os.path.join('temp/rps/scissors')
batu_dir = os.path.join('temp/rps/rock')
kertas_dir = os.path.join('temp/rps/paper')

if (not os.path.exists(gunting_dir)) or (not os.path.exists(batu_dir)) or (not os.path.exists(kertas_dir)):
    raise OSError('Training Data gunting-batu-kertas not found!')
    exit()
# if not os.path.exists(os.path.join('/tmp', '/rps-test-set')):
#     raise OSError('Testing Data gunting-batu-kertas not found!')
#     exit()

print('Training gunting images: {0}'.format(len(os.listdir(gunting_dir))))
print('Training batu    images: {0}'.format(len(os.listdir(batu_dir))))
print('Training kertas  images: {0}'.format(len(os.listdir(kertas_dir))))

press_any_key()

gunting_files = os.listdir(gunting_dir)
batu_files = os.listdir(batu_dir)
kertas_files = os.listdir(kertas_dir)

do_preview_training = ask_y_or_n('Preview Training Data?')
if do_preview_training:
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    pic_index = 2

    next_gunting = [os.path.join(gunting_dir, fname)
                    for fname in gunting_files[pic_index-2:pic_index]]
    next_batu = [os.path.join(batu_dir, fname)
                 for fname in batu_files[pic_index - 2:pic_index]]
    next_kertas = [os.path.join(kertas_dir, fname)
                   for fname in kertas_files[pic_index - 2:pic_index]]

    for i, img_path in enumerate(next_gunting+next_batu+next_kertas):
        img = mpimg.imread(img_path)
        plt.imshow(img)
        plt.axis('Off')
        plt.show()

training_dir = os.path.join('temp', 'rps')
training_datagen = ImageDataGenerator(rescale=1.0/255.0,
                                      rotation_range=25,
                                      width_shift_range=0.2,
                                      height_shift_range=0.2,
                                      shear_range=0.2,
                                      zoom_range=0.2,
                                      horizontal_flip=True,
                                      fill_mode='nearest')

validation_dir = os.path.join('temp', 'rps-test-set')
validation_datagen = ImageDataGenerator(rescale=1.0/255.0)

training_generator = training_datagen.flow_from_directory(training_dir,
                                                          target_size=(150, 150),
                                                          class_mode='categorical',
                                                          batch_size=32)
validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                              target_size=(150, 150),
                                                              class_mode='categorical',
                                                              batch_size=32)


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.summary()
press_any_key()

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

do_training = ask_y_or_n('Start for training data?')
if do_training:
    history = model.fit(training_generator,
                        epochs=10,
                        steps_per_epoch=100,
                        validation_data=validation_generator,
                        verbose=1,
                        validation_steps=5)

do_save = ask_y_or_n('Save training model in h5?')
if do_save:
    model.save("guntingbatukertas.h5")

do_plot = ask_y_or_n('Plot the training history?')
if do_plot:
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))
    plt.plot(epochs, acc, 'r', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend(loc=0)
    plt.figure()

    plt.show()
