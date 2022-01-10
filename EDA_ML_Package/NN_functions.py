import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import sys
from math import sqrt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.constraints import max_norm
from sklearn.metrics import classification_report, confusion_matrix
import random
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from keras import backend as K

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import r2_score

from numpy.random import seed
seed(1)
import tensorflow
tensorflow.random.set_seed(2)

import warnings

pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 20)
warnings.filterwarnings('ignore')


class ANN_tabular_class():
    def __init__(self, hidden_layers=[[6], [6, 4], [6, 4, 2]],
                 epochs=[15, 75], batch_size=[5, 15], drops=[0.0, 0.1, 0.2, 0.3],
                 optimizer=['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Nadam'],
                 activation=['relu', 'tanh', 'sigmoid']):  # 'softmax', 'softplus', 'softsign', 'hard_sigmoid', 'linear'

        self.hidden_layers = hidden_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.drops = drops
        self.optimizer = optimizer
        self.activation = activation

    def scale_data(self, X_train, X_test):

        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        return X_train, X_test
        '''>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'''
        '''>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'''

    def create_model(self, X, lyrs=[3], opt='Adam', dr=0.2, layers_act='relu'):

        tensorflow.random.set_seed(2)

        model = Sequential()
        # model.add(Dense(X.shape[1], input_dim=X.shape[1], activation='relu'))
        model.add(Dense(lyrs[0], input_dim=X.shape[1], activation=layers_act))
        # self.hidden_layers.pop(0)
        for i in range(1, len(lyrs)):  # if bool(self.hidden_layers) is True:
            # for hl in self.additional_hidden_layers:
            model.add(Dense(lyrs[i], activation=layers_act))
            model.add(Dropout(dr))
        model.add(Dense(units=1, activation='sigmoid'))  # sigmoid for classification

        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])  # , self.f1_m, self.precision_m, self.recall_m])
        '''>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'''

        return model

    def create_model_regression(self, X, lyrs=[3], opt='Adam', dr=0.2, layers_act='relu'):

        tensorflow.random.set_seed(2)

        model = Sequential()
        # model.add(Dense(X.shape[1], input_dim=X.shape[1], activation='relu'))
        model.add(Dense(lyrs[0], input_dim=X.shape[1], activation=layers_act))
        # self.hidden_layers.pop(0)
        for i in range(1, len(lyrs)):  # if bool(self.hidden_layers) is True:
            # for hl in self.additional_hidden_layers:
            model.add(Dense(lyrs[i], activation=layers_act))
            model.add(Dropout(dr))
        model.add(Dense(units=1))
        model.compile(loss='mse', optimizer=opt)  # , self.f1_m, self.precision_m, self.recall_m])
        '''>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'''

        return model

    def create_and_fit_model(self, X, y, batch_size, epochs, lyrs=[3], opt='Adam', dr=0.2, layers_act='relu', verbose=0, NN_problem_type='Classification'):
        if NN_problem_type == 'Classification':
            NN_model = self.create_model(X, lyrs, opt, dr, layers_act)
        else:
            NN_model = self.create_model_regression(X, lyrs, opt, dr, layers_act)

        NN_results = self.fit_model(X, y, NN_model, batch_size, epochs, NN_problem_type, verbose)
        NN_results['layers'] = str(lyrs)
        NN_results['opt'] = opt
        NN_results['dr'] = dr

        NN_results = round(pd.DataFrame(NN_results), 3)
        print('\n ================ NN Analysis Completed ================')
        print(NN_results)
        # self.table_in_PDF(NN_results) to be added

        return NN_results

    def fit_model(self, X, y, model, batch_size, epochs, NN_problem_type, verbose=2):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=101)
        X_train, X_test = self.scale_data(X_train, X_test)

        early_stop = EarlyStopping(monitor='val_loss', patience=5)

        model.fit(x=X_train, y=y_train, validation_data=(X_test, y_test), batch_size=batch_size, epochs=epochs, callbacks=[early_stop], verbose=verbose)

        if NN_problem_type == 'Classification':
            training = model.history
            losses = pd.DataFrame(model.history.history)
            try:
                losses[['loss', 'val_loss']].plot(title='Loss')
            except Exception:
                pass
            plt.show()
            # val_acc = np.mean(training.history['val_accuracy'])
            # print("\n%s: %.2f%%" % ('val_acc', val_acc*100))

            # summarize history for accuracy
            plt.plot(training.history['accuracy'])
            plt.plot(training.history['val_accuracy'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'validation'], loc='upper left')
            plt.show()
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

            # predict probabilities for test set
            y_probs = model.predict(X_test, verbose=0)
            # predict crisp classes for test set
            y_classes = model.predict_classes(X_test, verbose=0)
            # reduce to 1d array
            y_probs = y_probs[:, 0]
            y_classes = y_classes[:, 0]

            # accuracy: (tp + tn) / (p + n)
            accuracy = accuracy_score(y_test, y_classes)
            # precision tp / (tp + fp)
            precision = precision_score(y_test, y_classes)
            # recall: tp / (tp + fn)
            recall = recall_score(y_test, y_classes)
            # f1: 2 tp / (2 tp + fp + fn)
            f1 = f1_score(y_test, y_classes)

            # kappa
            # kappa = cohen_kappa_score(y_test, y_classes)
            # ROC AUC
            auc = roc_auc_score(y_test, y_probs)
            # confusion matrix
            # matrix = confusion_matrix(y_test, y_classes)
            results_dict = {'acc': [accuracy], 'precision': [precision], 'recall': [recall], 'f1': [f1], 'roc_auc': [auc]}

        else:
            losses = pd.DataFrame(model.history.history)
            losses.plot()

            y_pred = model.predict(X_test, verbose=0)
            mae = round(mean_absolute_error(y_test, y_pred), 3)
            mse = round(mean_squared_error(y_test, y_pred), 3)
            rmse = sqrt(mse)
            r2 = round(r2_score(y_test, y_pred), 3)
            results_dict = {'model': "NN",
                            'mae': [mae],
                            'rmse': [rmse],
                            'r2': [r2],
                            # 'roc_auc': [AUC_ROC]
                            }
            results_dict['dtype'] = 'test'

        del model
        K.clear_session()

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        return results_dict

    def optimum_batch_epoch(self, X, y):
        model_keras = KerasClassifier(self.create_model, X=X, verbose=0)
        param_grid = dict(batch_size=self.batch_size, epochs=self.epochs)

        grid_result = self.search_the_grid(X, y, model_keras, param_grid, 'batch_size')

        return grid_result.best_params_['batch_size'], grid_result.best_params_['epochs']

    def optimum_optimizer(self, X, y, batch_size, epoch, dr=0):
        model_keras = KerasClassifier(self.create_model, X=X, dr=dr, epochs=epoch, batch_size=batch_size, verbose=0)
        param_grid = dict(opt=self.optimizer)

        grid_result = self.search_the_grid(X, y, model_keras, param_grid, 'opt')

        return grid_result.best_params_['opt']

    def optimum_hidden_neurons(self, X, y, batch_size, epoch, optimizer='Adam'):
        model_keras = KerasClassifier(self.create_model, X=X, opt=optimizer, dr=0, epochs=epoch, batch_size=batch_size, verbose=0)
        param_grid = dict(lyrs=self.hidden_layers)

        grid_result = self.search_the_grid(X, y, model_keras, param_grid, 'lyrs')

        return grid_result.best_params_['lyrs']

    def optimum_dropout(self, X, y, batch_size, epoch, lyrs=[3], optimizer='Adam'):
        model_keras = KerasClassifier(self.create_model, X=X, lyrs=lyrs, opt=optimizer, epochs=epoch, batch_size=batch_size, verbose=0)
        param_grid = dict(dr=self.drops)

        grid_result = self.search_the_grid(X, y, model_keras, param_grid, 'dr')

        return grid_result.best_params_['dr']

    def optimum_activation(self, X, y, batch_size, epoch, lyrs=[3], optimizer='Adam', dr=0):
        model_keras = KerasClassifier(self.create_model, X=X, lyrs=lyrs, opt=optimizer, dr=dr, epochs=epoch, batch_size=batch_size, verbose=0)
        param_grid = dict(layers_act=self.activation)

        grid_result = self.search_the_grid(X, y, model_keras, param_grid, 'layers_act')

        return grid_result.best_params_['layers_act']

    def chain_optimazation(self, X, y, hidden_layers=[[6], [6, 4], [6, 4, 2]]):
        opt_batch, opt_epoch = self.optimum_batch_epoch(X, y)
        # =======================================================================
        opt_optimizer = self.optimum_optimizer(X, y, opt_batch, opt_epoch)
        # =======================================================================
        opt_layers = self.optimum_hidden_neurons(X, y, opt_batch, opt_epoch, opt_optimizer)
        # =======================================================================
        opt_dropout = self.optimum_dropout(X, y, opt_batch, opt_epoch, opt_layers, opt_optimizer)
        # =======================================================================
        opt_act = self.optimum_activation(X, y, opt_batch, opt_epoch, opt_layers, opt_optimizer, opt_dropout)
        # =======================================================================
        opt_model = self.create_model(X, lyrs=opt_layers, opt=opt_optimizer, dr=opt_dropout, layers_act=opt_act)

        if len(y.unique()) < 10:
            NN_problem_type = "Classification"
        else:
            NN_problem_type = "Regression"

        NN_results = self.fit_model(X, y, opt_model, opt_batch, opt_epoch, NN_problem_type)
        NN_results['layers'] = str(opt_layers)
        NN_results['opt'] = str(opt_dropout)
        NN_results['dr'] = str(opt_dropout)

        return NN_results, dict(opt_batch=opt_batch, opt_optimizer=opt_optimizer, opt_layers=opt_layers, opt_dropout=opt_dropout, opt_act=opt_act)

    def grid_optimazation(self, X, layers):
        model_keras = KerasClassifier(self.create_model, X=X, verbose=0)
        drops = [0.0, 0.1, 0.2, 0.3]
        batch_size = [5, 15]
        epochs = [15, 75]
        optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Nadam']
        activation = ['relu', 'tanh', 'sigmoid']
        param_grid = dict(opt=optimizer, batch_size=batch_size, epochs=epochs, lyrs=layers, dr=drops, layers_act=activation)

        grid_result = self.search_the_grid(X, y, model_keras, param_grid)

        return grid_result.best_params_

    def search_the_grid(self, X, y, model_keras, param_grid, title=''):
        # search the grid
        grid = RandomizedSearchCV(estimator=model_keras,    # GridSearchCV/ RandomizedSearchCV
                                  param_distributions=param_grid,   # param_grid/ param_distributions
                                  cv=3,
                                  verbose=2)  # if you are using CPU  -  , n_jobs=-1

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=101)
        X_train, X_test = self.scale_data(X_train, X_test)

        early_stop = EarlyStopping(monitor='val_loss', patience=5)
        grid_result = grid.fit(X_train, y_train, validation_split=0.2, callbacks=[early_stop], verbose=0)

        print("\n\nBest: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param), "\n\n")

        losses = pd.DataFrame(grid_result.best_estimator_.model.history.history)
        try:
            losses[['loss', 'val_loss']].plot(title=title + ": " + str(grid_result.best_params_[title]))
        except Exception:
            pass
        plt.show()

        return grid_result


if __name__ == '__main__':

    # sys.exit('early exit')

    dataset = np.loadtxt("pima-indians-diabetes.csv", delimiter=",")
    df = pd.read_csv('pima-indians-diabetes.csv')
    # split into input (X) and output (Y) variables
    X = dataset[:, 0:8]
    y = dataset[:, 8]

    ANNtab = ANN_tabular_class()

    reults2 = ANNtab.create_and_fit_model(X, y, 10, 250, verbose=0)

    NN_model1 = ANNtab.create_model(X)
    reults1 = ANNtab.fit_model(X, y, NN_model1, 10, 250, verbose=00)
    print(round(pd.DataFrame(reults1), 3))
    print('------')

    NN_model3 = ANNtab.create_model(X)
    reults3 = ANNtab.fit_model(X, y, NN_model3, 10, 250, verbose=0)
    print('------')
    print(round(pd.DataFrame(reults3), 3))

    print('------')
    # =======================================================================
    sys.exit()
    opt_act, _ = ANNtab.chain_optimazation(X, y, hidden_layers=[[6], [6, 4], [6, 4, 2]])
    # =======================================================================

# predictions = model.predict_classes(X_test)
# print(classification_report(y_test, predictions))
# confusion_matrix(y_test, predictions)
