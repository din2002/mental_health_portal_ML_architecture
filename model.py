import os
import numpy as np
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

def cnn(X_train, y_train, X_test, y_test, batch_size,
        nb_classes, epochs, input_shape):

    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='valid', strides=1,
                     input_shape=input_shape, activation='relu'))

    model.add(MaxPooling2D(pool_size=(4, 3), strides=(1, 3)))

    model.add(Conv2D(32, (1, 3), padding='valid', strides=1,
              input_shape=input_shape, activation='relu'))

    model.add(MaxPooling2D(pool_size=(1, 3), strides=(1, 3)))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                        verbose=1, validation_data=(X_test, y_test))

    # Evaluate accuracy on test and train sets
    score_train = model.evaluate(X_train, y_train, verbose=0)
    print('Train accuracy:', score_train[1])
    score_test = model.evaluate(X_test, y_test, verbose=0)
    print('Test accuracy:', score_test[1])

    return model, history


def model_performance(model, X_train, X_test, y_train, y_test):

    y_test_pred = model.predict_classes(X_test)
    y_train_pred = model.predict_classes(X_train)

    y_test_pred_proba = model.predict_proba(X_test)
    y_train_pred_proba = model.predict_proba(X_train)

    # Converting y_test back to 1-D array for confusion matrix computation
    y_test_1d = y_test[:, 1]

    # Computing confusion matrix for test dataset
    [[tn, fp], [fn, tp]] = confusion_matrix(y_test, y_test_pred)
    conf_matrix = np.array([[tp, fp], [fn, tn]])
    print("Confusion Matrix:")
    print(conf_matrix)

    return y_train_pred, y_test_pred, y_train_pred_proba, \
        y_test_pred_proba, conf_matrix

np.random.seed(15)

if __name__ == '__main__':
    dir_name = r'C:/Dinesh/SEM-VI/RBL/Mental_Health_Portal/data/processed/'
    X_train = os.path(dir_name+'train_samples.npz')
    y_train = os.path(dir_name+'train_labels.npz')
    X_test = os.path(dir_name+'test_samples.npz')
    y_test = os.path(dir_name+'test_labels.npz')

    X_train, y_train, X_test, y_test = \
        X_train['arr_0'], y_train['arr_0'], X_test['arr_0'], y_test['arr_0']

    # CNN parameters
    batch_size = 32
    nb_classes = 2
    epochs = 7

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    X_train = np.array([(X - X.min()) / (X.max() - X.min()) for X in X_train])
    X_test = np.array([(X - X.min()) / (X.max() - X.min()) for X in X_test])

    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    img_rows, img_cols, img_depth = X_train.shape[1], X_train.shape[2], 1

    input_shape = (img_rows, img_cols, 1)

    # run CNN
    print('Fitting model...')
    model, history = cnn(X_train, y_train, X_test, y_test, batch_size,
                         nb_classes, epochs, input_shape)

    # evaluate model
    print('Evaluating model...')
    y_train_pred, y_test_pred, y_train_pred_proba, y_test_pred_proba, \
        conf_matrix = model_performance(model, X_train, X_test, y_train, y_test)

    accuracy = float(conf_matrix[0][0] + conf_matrix[1][1]) / np.sum(conf_matrix)
    precision = float(conf_matrix[0][0]) / (conf_matrix[0][0] + conf_matrix[0][1])
    recall = float(conf_matrix[0][0]) / (conf_matrix[0][0] + conf_matrix[1][0])
    f1_score = 2 * (precision * recall) / (precision + recall)
    print("Accuracy: {}".format(accuracy))
    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("F1-Score: {}".format(f1_score))