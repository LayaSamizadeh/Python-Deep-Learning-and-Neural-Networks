import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense

def VGG_16(weights_path=None):
    
    model = Sequential()
    # Convolutional layers and max pooling layers
    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same', input_shape=(224,224,3))) 
    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=2))

    model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same'))
    model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=2))

    model.add(Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding='same'))
    model.add(Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding='same'))  
    model.add(Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding='same'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=2))

    model.add(Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same'))
    model.add(Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same'))
    model.add(Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=2))

    model.add(Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same')) 
    model.add(Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same'))
    model.add(Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=2))
        
    # Fully connected layers
    
    model.add(Flatten())

    model.add(Dense(units=4096, activation='relu'))
    model.add(Dense(units=4096, activation='relu'))
    model.add(Dense(units=1000, activation='softmax'))


    # Load weights if they exists

    if weights_path:
        model.load_weights(weights_path)

    return model


def VGG_16_model():
    return VGG_16('../models/vgg16_weights_tf_dim_ordering_tf_kernels.h5')