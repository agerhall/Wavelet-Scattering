import keras
from keras.layers import Dense, Flatten,BatchNormalization,Input,Conv2D, MaxPooling2D
from keras.models import Sequential, Model,load_model
# from keras.applications import ResNet50
from keras import layers, models, optimizers
from keras.datasets import mnist
from keras import backend as K
import tensorflow as tf
import tensorboard
import numpy as np
from matplotlib import pyplot as plt


def arch1(input_shape,num_classes):

    # Network structure
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(5, 5), strides=(1, 1), padding='same',
                     activation='relu', input_shape=input_shape))
    model.add(BatchNormalization(axis=1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same',
                     activation='relu'))
    model.add(BatchNormalization(axis=1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(300, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.001,amsgrad=False),metrics=['accuracy'])
    return model


def trainCNN(X, Y, input_shape, num_classes, batch_size,epochs):
    CNNModel = arch1(input_shape, num_classes)

    testX = X[:, :, 1:round(0.2*len(X[1, 1, :]))]
    valX = X[:, :, round(0.2*len(X[1, 1, :])):round(0.3*len(X[1, 1, :]))]
    trainX = X[:, :, round(0.3*len(X[1, 1, :])):]

    testY = Y[1:round(0.2 * len(Y))]
    valY = Y[round(0.2 * len(Y)):round(0.3 * len(Y))]
    trainY = Y[round(0.3 * len(Y)):]

    CNNModel.fit(trainX, trainY, batch_size, epochs, validation_data=(valX, valY))



    return CNNModel, testX,testY


def evalModel(model, testX, testY, batch_size):
    score = model.evaluate(testX, testY, batch_size=batch_size, verbose=0)

    print('Results on test set:')
    print(str(model.metrics_names[0]) + ': ' + str(score[0]))
    print(str(model.metrics_names[1]) + ': ' + str(score[1]))

def mnistData():
    (trainX, trainY), (testX, testY) = mnist.load_data()

    if K.image_data_format() == 'channels_first':
        trainX = trainX.reshape(trainX.shape[0], 1, img_y, img_y)
        testX = testX.reshape(testX.shape[0], 1, img_y, img_y)
        input_shape = (1, img_y, img_y)
    else:
        trainX = trainX.reshape(trainX.shape[0], img_y, img_y, 1)
        testX = testX.reshape(testX.shape[0], img_y, img_y, 1)
        input_shape = (img_y, img_y, 1)

    print(trainX.shape)

    # convert class vectors to binary class matrices
    trainY = keras.utils.to_categorical(trainY, num_classes)
    testY = keras.utils.to_categorical(testY, num_classes)

    return trainX, trainY, testX, testY, input_shape



if __name__=='__main__':


    # initialize parameters
    img_x, img_y = 28, 28 # for mnist we have 28x28 images
    batch_size=10
    epochs=5
    num_classes = 10

    trainX,trainY,testX,testY, input_shape = mnistData() # may use other data as well

    plt.imshow(trainX[1,:,:])
    plt.show()

    # Define the model with a given architecture
    CNNModel = arch1(input_shape, num_classes)

    # train the model
    CNNModel.fit(trainX, trainY, batch_size, epochs)

    # evaluate the model performance
    evalModel(CNNModel, testX, testY, batch_size)


