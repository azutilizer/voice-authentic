import os
import numpy as np
import sys
from keras import Sequential
from keras.callbacks import History
from keras.layers import LSTM, Dense, Dropout, Conv2D, Flatten, \
    BatchNormalization, Activation, MaxPooling2D
from keras.utils import np_utils
from tqdm import tqdm
from utilities import get_data

models = ["CNN", "LSTM"]
history = History()


def get_model(class_labels, model_name, input_shape):
    model = Sequential()
    if model_name == 'CNN':
        model.add(Conv2D(8, (13, 13),
                         input_shape=(input_shape[0], input_shape[1], 1)))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation('relu'))
        model.add(Conv2D(8, (13, 13)))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 1)))
        model.add(Conv2D(8, (13, 13)))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation('relu'))
        model.add(Conv2D(8, (2, 2)))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 1)))
        model.add(Flatten())
        model.add(Dense(64))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
    elif model_name == 'LSTM':
        model.add(LSTM(128, input_shape=(input_shape[0], input_shape[1])))
        model.add(Dropout(0.5))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(16, activation='tanh'))
    model.add(Dense(len(class_labels), activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])
    print(model.summary())
    return model


def evaluateModel(model, model_path):
    # Train the epochs
    best_acc = 0
    global x_train, y_train, x_test, y_test
    for i in tqdm(range(50)):
        p = np.random.permutation(len(x_train))
        x_train = x_train[p]
        y_train = y_train[p]
        ret = model.fit(x_train, y_train, batch_size=32, epochs=1, callbacks=[history])
        # loss = ret.history['loss']
        acc = ret.history['acc']

        if acc[0] > best_acc:
            best_acc = acc[0]
            model.save_weights(model_path, overwrite=True)
    model.load_weights(model_path)
    print('Accuracy = ', best_acc)
    model.save(model_path, overwrite=True)
    return best_acc


def training(mode, model_path, dataset_folder, class_labels):
    # Read data
    global x_train, y_train, x_test, y_test
    x_train, x_test, y_train, y_test = get_data(dataset_folder, class_labels, flatten=False)
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    model_id = mode - 1
    if model_id == 0:
        # Model is CNN so have to reshape the data
        in_shape = x_train[0].shape
        x_train = x_train.reshape(x_train.shape[0], in_shape[0], in_shape[1], 1)
        x_test = x_test.reshape(x_test.shape[0], in_shape[0], in_shape[1], 1)
    elif model_id > len(models):
        sys.stderr.write('Model Not Implemented yet')
        sys.exit(-1)

    model = get_model(class_labels, models[model_id], x_train[0].shape)

    accuracy = evaluateModel(model, model_path)
    return accuracy


def training_model(model_path):
    dataset_folder = 'voice'

    class_labels = []
    for sub_folder in os.listdir(dataset_folder):
        class_labels.append(sub_folder)
    return training(2, model_path, dataset_folder, class_labels)


if __name__ == '__main__':
    voice_acc = training_model()
    print('accuracy: {:.2f}%\n'.format(voice_acc*100))
