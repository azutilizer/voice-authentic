"""
Train a model and display its metrics
"""
import sys
import numpy as np
from utilities import get_data, get_one_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC as SVC
import util

models = ['SVM', 'Random Forest', 'Neural network']


def get_model(model_name):
    """
    Create a model and return it
    :param model_name: name of the model to be created
    :return: created model with fit and predict methods
    """
    if model_name == models[0]:
        return SVC(multi_class='crammer_singer')
    elif model_name == models[1]:
        return RandomForestClassifier(n_estimators=30)
    elif model_name == models[2]:
        return MLPClassifier(solver='lbfgs', alpha=1e-5, activation='logistic', verbose=True,
                             hidden_layer_sizes=(512,), batch_size=32)


def evaluateModel(model_name, model_path):
    clf = get_model(model_name)
    x_train, x_test, y_train, y_test = get_data()
    print('------------- Training Started -------------')
    clf.fit(x_train, y_train)
    print('------------- Training Ended -------------')
    score = clf.score(x_test, y_test)
    print("accuracy: {:.2f}%".format(score * 100.))
    util.save_speaker_model(model_path, clf)


def train(mode, model_path):
    # 1 - SVM
    # 2 - Random Forest
    # 3 - Neural Network

    n = mode - 1
    if n > len(models):
        sys.stderr.write('Invalid Model number')
        return
    print('model given', models[n])
    evaluateModel(models[n], model_path)



