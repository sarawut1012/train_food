import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.core import Dropout
from keras.layers.core import Flatten, Activation
from keras.layers.core import Dense
from keras.layers import Conv2D, MaxPooling2D
from keras import optimizers


from keras import backend as K

from keras.models import Sequential
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np

import cv2
import os
from matplotlib.backends.backend_pdf import PdfPages
from skimage import io


from keras.utils import plot_model
from keras.layers import BatchNormalization
from keras import regularizers
lb = LabelBinarizer()

batch_size = 30
num_classes = 30
epochs = 200
image_paths = list(paths.list_images(r'data-train'))


image_size = 80
data = []
labels = []
for image_path in image_paths:
    try:
        image = io.imread(image_path)
        image = cv2.resize(image, (image_size, image_size))
        # print(image)
        image.astype('float32')
        image = image / 255
        if (image.shape == (image_size, image_size, 3)):
            # print(image_path)
            data.append(image)
            labels.append(image_path.split('\\')[-2])
            # print(labels)
        else:
            print(image_path)
            os.remove(image_path)
            print("File Removed!")
    except :
        # print(err)
        print("Error on image: ", image_path)
        os.remove(image_path)
        print("File Removed!")

# print(labels)

labels = lb.fit_transform(labels)
# for i in labels:
#     print(i)
data = np.array(data)
labels = np.array(labels)


data = data.reshape(data.shape[0], image_size, image_size, 3)

# for i in labels:
#     print(i)

print(data.shape[0], 'all data')
x_train, x_test, y_train, y_test = train_test_split(data,labels, test_size=0.1)

del data
del labels
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

plt.figure(1, figsize=[5, 6])
plt.subplot(221)
plt.imshow(x_train[0])
plt.title("Ground truth: %s" % y_train[0])
plt.subplot(222)
plt.imshow(x_train[1])
plt.title("Ground truth: %s" % y_train[1])
plt.subplot(223)
plt.imshow(x_train[2])
plt.title("Ground truth: %s" % y_train[2])
plt.subplot(224)
plt.imshow(x_train[3])
plt.title("Ground truth: %s" % y_train[3])
plt.show()

learning_rate = 0.1
lr_decay = 1e-6
lr_drop = 20

if K.image_data_format() == 'channels_first':
    input_shape = (3, image_size, image_size)
else:
    input_shape = (image_size, image_size, 3)

weight_decay = 0.0005
model = Sequential()
model.add(Conv2D(64, (3, 3), padding='same',
                 input_shape=input_shape, kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(512, kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
model.summary()
plot_model(model, show_shapes=True, to_file='multiple_vgg_blocks.png')
def lr_scheduler(epoch):
    return learning_rate * (0.5 ** (epoch // lr_drop))
reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)

# data augmentation
datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=60,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False  # randomly flip images
)
datagen.fit(x_train)

# opt_rms = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size), \
                              steps_per_epoch=x_train.shape[0] // batch_size, epochs=epochs, \
                              verbose=1, validation_data=(x_test, y_test),
                              callbacks=[reduce_lr])

model.save('model-test-new.h5')


def generateEvaluationGraph(history, i):
    # Loss curves
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    f.suptitle('Fold: ' + str(i), fontsize=18)
    ax1.plot(history.history['loss'], 'r', linewidth=3.0)
    ax1.plot(history.history['val_loss'], 'b', linewidth=3.0)
    ax1.legend(['Training set', 'Validation set'], fontsize=16)
    ax1.set_xlabel('Epochs ', fontsize=16)
    ax1.set_ylabel('Loss', fontsize=16)
    ax1.set_title('Loss Curves', fontsize=16)

    # Accuracy Curves
    ax2.plot(history.history['acc'], 'r', linewidth=3.0)
    ax2.plot(history.history['val_acc'], 'b', linewidth=3.0)
    ax2.legend(['Training set', 'Validation set'], fontsize=16)
    ax2.set_xlabel('Epochs ', fontsize=16)
    ax2.set_ylabel('Accuracy', fontsize=16)
    ax2.set_title('Accuracy Curves', fontsize=16)
    return f


fig = generateEvaluationGraph(history, 1)
pp = PdfPages('foo2.pdf')
pp.savefig(fig)
pp.close()
