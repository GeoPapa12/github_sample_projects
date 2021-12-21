import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import os
import io
import warnings
from pprint import pprint
from math import sqrt


# PDF Report building
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer, Table, TableStyle  # import PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib import colors

from IPython import get_ipython

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression

from xgboost import XGBClassifier

from sklearn import svm
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate  # RandomizedSearchCV
from sklearn.model_selection import learning_curve
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import r2_score

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder  # RobustScaler

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer  # , KNNImputer
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


from sklearn.model_selection import RandomizedSearchCV

from sklearn.ensemble import BaggingRegressor, AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesRegressor  # Ensemble methods
from sklearn.linear_model import Lasso, Ridge, ElasticNet, RANSACRegressor, SGDRegressor, HuberRegressor, BayesianRidge # Linear models
from xgboost import XGBRegressor, plot_importance # XGBoost

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

from sklearn.pipeline import make_pipeline
from sklearn.cluster import DBSCAN

# Statistics
from scipy import stats
# from scipy.stats import skew
# from scipy.stats import kurtosis

get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings('ignore')


class ML_models():
    # Story = []

    def __init__(self, Story=[], doc_title="", K_split=5, test_ratio=0.2, path='ML_plots', title_type=''):
        self.K_split = K_split
        self.test_ratio = test_ratio
        self.Story = Story
        self.path = path
        self.doc_title = doc_title
        self.now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

        self.title_type = title_type

        self.styles = getSampleStyleSheet()
        # self.doc = SimpleDocTemplate("ML analysis " + self.nowTitle + "_" + self.doc_title + ".pdf", pagesize=letter,
        #                             rightMargin=inch/2, leftMargin=inch/2,
        #                             topMargin=25.4, bottomMargin=12.7)

        if str(self).find('ML_models') != -1:
            self.add_text("ML Analysis", style="Heading1", fontsize=24)
            self.add_text(self.now)

        self.scoring_metrics_regression = ['r2', 'neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_root_mean_squared_error']
        self.scoring_metrics = ['accuracy', 'precision', 'recall', 'f1']  # , 'roc_auc']
        self.scoring_metrics_multi = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']  # , 'roc_auc_weighted']
        self.scoring_column_names = ['acc', 'precision', 'recall', 'f1']  # , 'roc_auc']
        self.scoring_column_names_regression = ['r2', 'mae', 'mse', 'rmse']  # , 'roc_auc']

        self.random_state = 12
        print('\n--------------------------------------------------------------')

    def directory_one_level_up(self):
        return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    def add_text(self, text, style="Normal", fontsize=12):
        """ Adds text with some spacing around it to  PDF report

        Parameters
        ----------
        text : (str) The string to print to PDF
        style : (str) The reportlab style
        fontsize : (int) The fontsize for the text
        """
        self.Story.append(Spacer(1, 4))
        ptext = "<font size={}>{}</font>".format(fontsize, text)
        self.Story.append(Paragraph(ptext,  self.styles[style]))
        self.Story.append(Spacer(1, 4))

    def table_in_PDF(self, df_results):
        """ Adds style to table to be printed in pdf

        Parameters
        ----------
        table : (list) table to be printed in pdf

        Output: table (list)
        """

        colNames = df_results.columns.to_list()
        table = df_results.values.tolist()
        table.insert(0, colNames)
        table = Table(table, hAlign='LEFT')

        table.setStyle(TableStyle([
            ('FONTSIZE', (0, 0), (-1, 0), 7),
            ('FONTSIZE', (0, 0), (-1, -1), 7),
            ('FONT', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('ALIGN', (0, 0), (0, -1), 'CENTER'),
            ('INNERGRID', (0, 0), (-1, -1), 0.50, colors.black),
            ('BOX', (0, 0), (-1, -1), 0.25, colors.black),
        ]))

        self.add_text("")
        self.Story.append(table)
        self.add_text("")

        return table

    def image_in_PDF(self, plot, x=7, y=2.5):

        buf = io.BytesIO()
        plot.savefig(buf, format='png', dpi=300)
        buf.seek(0)
        # you'll want to close the figure once its saved to buffer
        if 'Figure' in str(type(plot)) is False:
            plot.close()

        self.Story.append(Image(buf, x*inch, y*inch))
        return buf

    def create_directory(self, path):
        """ Creates a folder if the folder does not exist

        Parameters
        ----------
        path: (str)

        Output: -
        """
        try:
            os.mkdir(path)
        except OSError:
            'file aLogRegeady exists'

    def generate_report(self, docTitle):
        """ Buids the PDF report

        Parameters
        ----------
        -

        Output: -
        """
        nowTitle = datetime.now().strftime("%d_%m_%Y %H-%M-%S")

        self.doc = SimpleDocTemplate("ML " + str(docTitle) + "_" + nowTitle + "_" + self.doc_title + ".pdf", pagesize=letter,
                                     rightMargin=inch/2, leftMargin=inch/2,
                                     topMargin=25.4, bottomMargin=12.7)

        self.doc.build(self.Story)

    def confusion_matrix_plot(self, Ytest, Ypred, modelName, percentage=1):

        plt.figure(figsize=(9, 9))
        if percentage == 1:
            sns.heatmap(confusion_matrix(Ytest, Ypred)/len(Ypred), annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Greens_r')
        else:
            sns.heatmap(confusion_matrix(Ytest, Ypred), annot=True, fmt=".2f", linewidths=.5, square=True, cmap='Greens_r')
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        all_sample_title = 'Confusion Matrix'
        plt.title(all_sample_title, size=12)
        # plt.savefig(self.path + 'confusion_matrix_' + self.title_type + '_' + modelName + '.png', dpi=500, bbox_inches='tight')

        buf = self.image_in_PDF(plt, x=3, y=3)
        plt.show()

        return buf

    def ROC_curve(self, names, FRP_list, TRP_list, thresholds_list, best_thres, title=''):

        plt.figure(figsize=(12, 12))

        if len(names) <= 3:
            for x in range(len(names)):
                plt.plot(FRP_list[x], TRP_list[x], label=names[x])
                plt.scatter(FRP_list[x][best_thres[x]], TRP_list[x][best_thres[x]], marker='o', color='black', label='Best')
        else:
            for x in range(len(names)):
                plt.plot(FRP_list[x], TRP_list[x], label=names[x])

        for i, thr in enumerate(best_thres):
            plt.annotate(round(thresholds_list[i][thr], 3), (FRP_list[i][best_thres[i]], TRP_list[i][best_thres[i]] + 0.01))
            pass

        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='random', alpha=.8)
        plt.legend()
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)

        # plt.savefig('ROC_curve' + title + '.png', dpi=500, bbox_inches='tight')
        buf = self.image_in_PDF(plt, x=6, y=6)
        plt.show()

        return buf

    def precision_recall_plot(self, model, X_train, y_train, title):
        y_scores = model.predict_proba(X_train)
        y_scores = y_scores[:, 1]
        precision, recall, threshold = precision_recall_curve(y_train, y_scores)

        plt.figure(figsize=(14, 7))
        plt.title('Precision vs Recal/ Model: ' + title)
        plt.grid()

        plt.plot(threshold, precision[:-1], "r-", label="precision", linewidth=5)
        plt.plot(threshold, recall[:-1], "b", label="recall", linewidth=5)
        plt.xlabel("threshold", fontsize=19)
        plt.legend(loc="upper right", fontsize=19)
        plt.ylim([0, 1])

        buf = self.image_in_PDF(plt, x=5, y=5)
        # plt.savefig(self.path + 'Precision_recal_' + title + '_.png', dpi=500, bbox_inches='tight')
        plt.show()
        return buf

    def base_model_performance(self, df_results, scoring_names):
        plt.figure(facecolor='#07000d', dpi=600)
        dp = df_results[df_results['dtype'] == 'cv'][['model'] + scoring_names]
        dp = dp.set_index('model')
        plt.rcParams["figure.dpi"] = 250
        dp.plot.line()
        plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
        plt.legend(loc='upper center', ncol=5, prop={'size': 8})
        plt.title("Base Models PeRFormance")
        plt.show()
        # fig.savefig(self.path + 'base_peRFormance_' + self.title_type + '.png')  # , facecolor=fig.get_facecolor(), dpi=600, bbox_inches='tight')

    def feature_importances_plot(self, df, model, title):

        featureImportances = pd.DataFrame(model.feature_importances_, index=df.columns.tolist()[0:len(model.feature_importances_)],
                                          columns=['importance']).sort_values('importance', ascending=False)
        num = len(df.columns.tolist()[0:len(model.feature_importances_)])
        ylocs = np.arange(num)
        # get the feature importance for top num and sort in reverse order
        values_to_plot = featureImportances.iloc[:num].values.ravel()[::-1]
        feature_labels = list(featureImportances.iloc[:num].index)[::-1]

        plt.figure(num=None, figsize=(8, 12), dpi=250, facecolor='w', edgecolor='k')
        plt.barh(ylocs, values_to_plot, align='center')
        plt.ylabel('Features')
        plt.xlabel('Importance Score')
        plt.title('Positive Feature Importance Score ' + title)
        plt.yticks(ylocs, feature_labels)
        plt.show()
        # figure = plt.get_figure()
        # featImp.savefig(self.path + 'Positive Feature Importance_' + self.title_type + '.png', dpi=500, bbox_inches='tight')

        buf = self.image_in_PDF(plt, x=8, y=5)

        return buf

    def plot_learning_curve(self, model, title, X, y, ylim=None, cv=10, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 10)):
        """
        Generate a simple plot of the test and training learning curve.

        Parameters
        ----------
        model : object type that implements the "fit" and "predict" methods
            An object of that type which is cloned for each validation.

        title : string
            Title for the chart.

        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples) or (n_samples, n_features), optional
            Target relative to X for classification or regression;
            None for unsupervised learning.

        ylim : tuple, shape (ymin, ymax), optional
            Defines minimum and maximum yvalues plotted.
        cv : int, cross-validation generator or an iterable, optional
            Determines the cross-validation splitting strategy.
            Possible inputs for cv are:
              - None, to use the default 3-fold cross-validation,
              - integer, to specify the number of folds.
              - An object to be used as a cross-validation generator.
              - An iterable yielding train/test splits.

            For integer/None inputs, if ``y`` is binary or multiclass,
            :class:`StratifiedKFold` used. If the model is not a classifier
            or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

            Refer :ref:`User Guide <cross_validation>` for the various
            cross-validators that can be used here.

        n_jobs : integer, optional
            Number of jobs to run in parallel (default 1).
        """
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("Training examples")
        plt.ylabel("F1")
        train_sizes, train_scores, test_scores = learning_curve(
            model, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='f1_weighted')

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1, color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="b")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="b", label="Cross-validation score")

        plt.legend(loc="best")

        # plt.savefig(self.path + 'Learning Curve RF_' + self.title_type + '.png', dpi=500, bbox_inches='tight')

        buf = self.image_in_PDF(plt, x=5, y=5)
        plt.show()
        return buf

    def test_metrics_calculation(self, modelName, y_test, y_pred):
        '''Test metrics are calculated'''
        Accuracy = round(accuracy_score(y_test, y_pred), 3)
        # AUC_ROC = round(roc_auc_score(y_test, y_pred), 3)
        F1score = round(f1_score(y_test, y_pred), 3)
        Precision = round(precision_score(y_test, y_pred), 3)
        Recall = round(recall_score(y_test, y_pred), 3)
        results_dict = {'model': modelName,
                        'acc': [Accuracy],
                        'precision': [Precision],
                        'recall': [Recall],
                        'f1': [F1score],
                        # 'roc_auc': [AUC_ROC]
                        }
        results_dict['dtype'] = 'test'
        return round(pd.DataFrame(results_dict), 3)

    def test_metrics_calculation_Regression(self, modelName, y_test, y_pred):
        '''Test metrics are calculated'''

        mae = round(mean_absolute_error(y_test, y_pred), 3)
        mse = round(mean_squared_error(y_test, y_pred), 3)
        rmse = sqrt(mse)
        r2 = round(r2_score(y_test, y_pred), 3)
        results_dict = {'model': modelName,
                        'mae': [mae],
                        'rmse': [rmse],
                        'r2': [r2],
                        # 'roc_auc': [AUC_ROC]
                        }
        results_dict['dtype'] = 'test'
        return round(pd.DataFrame(results_dict), 3)

    def cv_metrics_calculation(self, scoring_names, modelName, cv_results, scoring_list):
        '''CV metrics are calculated'''
        results_dict = {}
        results_dict['dtype'] = 'cv'
        for i, cv_metric in enumerate(scoring_list):
            results_dict[scoring_names[i]] = [round(cv_results['test_' + cv_metric].mean(), 3)]
        results_dict['fit_tm'] = [round(cv_results['fit_time'].mean(), 3)]
        results_dict['score_tm'] = [round(cv_results['score_time'].mean(), 3)]
        results_dict['model'] = modelName

        return round(pd.DataFrame(results_dict), 3)

    def ML_Basic_Models(self, X, y, threshold=0.5, test_size=0.2, test_case=1):
        '''
        Lightweight script to test many models and find winners
        :param X_train: training split
        :param y_train: training target vector
        :param X_test: test split
        :param y_test: test target vector
        :return: DataFrame of predictions
        '''

        FRP_list = []
        TRP_list = []
        thresholds_list = []
        best_thres = []

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=self.random_state)

        models = [
          ('LogReg', LogisticRegression()),
          ('RF', RandomForestClassifier(random_state=self.random_state)),
          ('KNN', KNeighborsClassifier()),  # ('SVM', SVC()),
          ('GNB', GaussianNB()),
          ('XGB', XGBClassifier())
        ]

        df_cv_results = pd.DataFrame()
        df_test_results = pd.DataFrame()

        for name, model in models:
            kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=self.random_state)
            try:  # Binary classification
                cv_results = model_selection.cross_validate(model, X_train, y_train, cv=kfold, scoring=self.scoring_metrics)
            except Exception:   # Multiclass
                cv_results = model_selection.cross_validate(model, X_train, y_train, cv=kfold, scoring=self.scoring_metrics_multi)

            clf = model.fit(X_train, y_train)
            y_pred = (clf.predict_proba(X_test)[:, 1] > threshold)  # y_pred = clf.predict(X_test)
            y_pred_prob = clf.predict_proba(X_test)[:, 1]

            if test_case == 1:  # If test case equal to 0 do not calculate test metrics
                df_test_results = pd.concat([df_test_results, self.test_metrics_calculation(name, y_test, y_pred)])
            df_cv_results = pd.concat([df_cv_results, self.cv_metrics_calculation(self.scoring_column_names, name, cv_results, self.scoring_metrics)])
            if name == 'RF':  # Save model and predictions for Random forest
                clf_RF = clf
                y_pred_RF = y_pred

            # Data for the ROC curve
            if len(np.unique(y_test)) == 2:
                fpr_log, tpr_log, thresholds = roc_curve(y_test, y_pred_prob)
                FRP_list.append(fpr_log)
                TRP_list.append(tpr_log)
                thresholds_list.append(thresholds)
                gmeans = np.sqrt(tpr_log * (1-fpr_log))
                # locate the index of the largest g-mean
                ix = np.argmax(gmeans)
                best_thres.append(ix)

        df_results = pd.concat([df_test_results, df_cv_results])
        df_results = df_results.reset_index(drop=True)
        df_results = df_results.sort_values(by=['dtype', 'f1'], ascending=False)

        # ROC curve/ Confusion Matrix/ Feature Importance/ Learning Curve
        if len(np.unique(y_test)) == 2:
            self.ROC_curve([x[0] for x in models], FRP_list, TRP_list, thresholds_list, best_thres)
            self.confusion_matrix_plot(y_test, y_pred_RF, 'RF')
            self.confusion_matrix_plot(y_test, y_pred_RF, 'RF', percentage=0)

        '''Visualization of Results'''
        # self.base_model_performance(df_results, self.scoring_column_names)
        # self.precision_recall_plot(clf_RF, X_train, y_train, 'RF')
        self.feature_importances_plot(X_train, clf_RF, 'RF')
        # self.plot_learning_curve(clf_RF, 'RF', X_train, y_train, (0.2, 1.01), kfold, 4, np.linspace(.1, 1.0, 10))
        self.table_in_PDF(df_results)

        print('\n ================ ML Analysis Completed ================')
        print(df_results)
        return df_results

    def ML_Basic_Models_Regression(self, X, y, threshold=0.5, test_size=0.2, test_case=1):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=self.random_state)

        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        models = [
            ('LReg', LinearRegression()),
            ('RF', RandomForestRegressor(random_state=self.random_state)),
            ("Ridge", Ridge(random_state=self.random_state, tol=10)),
            ("Lasso", Lasso(random_state=self.random_state, tol=1)),
            ("BR", BaggingRegressor(random_state=self.random_state)),
            # ("Hub-Reg", HuberRegressor()),
            ("BR", BayesianRidge()),
            ("XGBR", XGBRegressor(seed=self.random_state)),
            # ("GBoost-Reg", GradientBoostingRegressor()),
        ]

        df_cv_results = pd.DataFrame()
        df_test_results = pd.DataFrame()

        for name, model in models:
            kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=self.random_state)
            cv_results = model_selection.cross_validate(model, X_train, y_train, cv=kfold, scoring=self.scoring_metrics_regression)

            clf = model.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            # y_pred_prob = clf.predict_proba(X_test)[:, 1]

            if test_case == 1:  # If test case equal to 0 do not calculate test metrics
                df_test_results = pd.concat([df_test_results, self.test_metrics_calculation_Regression(name, y_test, y_pred)])
            df_cv_results = pd.concat([df_cv_results, self.cv_metrics_calculation(self.scoring_column_names_regression, name, cv_results, self.scoring_metrics_regression)])
            if name == 'RF':  # Save model and predictions for Random forest
                clf_RF = clf
                # y_pred_RF = y_pred

            plt.figure(figsize=(7, 7))
            plt.scatter(y_test, y_pred)
            plt.plot(y_test, y_test, 'r')
            plt.title(name)
            plt.show()

            plt.figure(figsize=(8, 8))
            sns.jointplot(y_test, y_pred, alpha=0.5)
            plt.xlabel('y_test')
            plt.ylabel('y_pred')
            plt.show()

        df_results = pd.concat([df_test_results, df_cv_results])
        df_results = df_results.reset_index(drop=True)
        df_results = df_results.sort_values(by=['dtype', 'r2'], ascending=False)

        df_results['mae%erroor'] = round(df_results['mae']/y.mean(), 3)
        df_results['rmse%erroor'] = round(df_results['rmse']/y.mean(), 3)
        df_results = df_results.apply(lambda x: x.abs() if np.issubdtype(x.dtype, np.number) else x)

        '''Visualization of Results'''
        # self.base_model_performance(df_results, self.scoring_column_names_regression)
        self.feature_importances_plot(X, clf_RF, 'RF')
        # self.plot_learning_curve(clf_RF, 'RF', X_train, y_train, (0.2, 1.01), kfold, 4, np.linspace(.1, 1.0, 10))
        self.table_in_PDF(df_results)

        print('\n ================ ML Analysis Completed ================')
        print(df_results)
        return df_results

    def opt_ramdom_forest(self, X, y, threshold=0.5, test_size=0.2, n_iter=100, verbose=2):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=self.random_state)

        '''
        - n_estimators = number of trees in the foreset
        - max_features = max number of features considered for splitting a node
        - max_depth = max number of levels in each decision tree
        - min_samples_split = min number of data points placed in a node before the node is split
        - min_samples_leaf = min number of data points allowed in a leaf node
        - bootstrap = method for sampling data points (with or without replacement)
        '''

        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start=5, stop=100, num=5)]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(5, 100, num=5)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4, 10]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]
        # Create the random grid
        random_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap}
        # Use the random grid to search for best hyperparameters
        # First create the base model to tune
        rf = RandomForestClassifier()
        # Random search of parameters, using 3 fold cross validation,
        # search across 100 different combinations, and use all available cores
        rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=n_iter, cv=3, verbose=verbose, random_state=self.random_state, n_jobs=-1)
        # Fit the random search model
        rf_random.fit(X_train, y_train)

        # =====================================================================
        best_random = rf_random.best_estimator_
        kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        cv_results = model_selection.cross_validate(best_random, X_train, y_train, cv=kfold, scoring=self.scoring_metrics)
        y_pred = (best_random.predict_proba(X_test)[:, 1] > threshold)
        # y_pred_prob = best_random.predict_proba(X_test)[:, 1]

        df_test_results = pd.concat([pd.DataFrame(), self.test_metrics_calculation('opt RF', y_test, y_pred)])
        df_cv_results = pd.concat([pd.DataFrame(), self.cv_metrics_calculation(self.scoring_column_names, 'opt RF', cv_results, self.scoring_metrics)])

        # =====================================================================
        base_model = RandomForestClassifier(random_state=self.random_state)
        base_model.fit(X_train, y_train)
        y_pred = (base_model.predict_proba(X_test)[:, 1] > threshold)
        cv_results = model_selection.cross_validate(base_model, X_train, y_train, cv=kfold, scoring=self.scoring_metrics)

        df_test_results = pd.concat([df_test_results, self.test_metrics_calculation('bs RF', y_test, y_pred)])
        df_cv_results = pd.concat([df_cv_results, self.cv_metrics_calculation(self.scoring_column_names, 'bs RF', cv_results, self.scoring_metrics)])
        df_results = pd.concat([df_test_results, df_cv_results])
        df_results = df_results.reset_index(drop=True)

        print('\n ================ Opt RF Analysis Completed ================')
        print(df_results)
        self.table_in_PDF(df_results)

        print('Best Parameters:\n')
        print(best_random.get_params())
        # pprint(best_random.get_params())

        # print('Base Parameters:\n')
        # pprint(base_model.get_params())

        return df_results, best_random.get_params(), random_grid

    def PCA_reduction(self, df, no_of_components, story, reduction_method='tSNE', print_option=1):

        df_columns = df.columns.tolist()

        scaler = MinMaxScaler()

        df_values = scaler.fit_transform(df)
        df = pd.DataFrame(df_values, columns=df_columns)

        # scaler = StandardScaler()
        pca = PCA()
        pipeline = make_pipeline(scaler, pca)
        pipeline.fit(df)

        '''
        features = range(pca.n_components_)
        _ = plt.figure(figsize=(22, 5))
        _ = plt.bar(features, pca.explained_variance_)
        _ = plt.xlabel('PCA feature')
        _ = plt.ylabel('Variance')
        _ = plt.xticks(features)
        _ = plt.title("Importance of the Principal Components based on inertia")
        plt.savefig(self.pth_REFIT + 'PCA_n_components_' + str(no_of_components) + '.png')  # , facecolor=plt.get_facecolor(), dpi=600, bbox_inches='tight')
        plt.show()
        '''
        # reduce to 2 important features
        # if reduction_method == "PCA":
        pca = PCA(n_components=no_of_components)
        data = pca.fit_transform(df)  # X
        # else:
        #    data = TSNE(n_components=no_of_components).fit_transform(df)

        # standardize these 2 new features
        np_scaled = StandardScaler().fit_transform(data)
        X = pd.DataFrame(np_scaled).values

        # ======================================================

        n_pcs = pca.components_.shape[0]
        most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]
        initial_feature_names = df.columns.tolist()
        most_important_names = [initial_feature_names[most_important[i]] for i in range(n_pcs)]
        dic = {'PC{}'.format(i+1): most_important_names[i] for i in range(n_pcs)}

        if print_option == 1:
            pc_influnce = pd.DataFrame(pca.components_, columns=df.columns, index=dic.keys())
            # PDFge = PDF_generator()
            # story = PDFge.add_text(story, "Columns influence in PCA")
            # story = PDFge.table_in_PDF(story, round(pc_influnce, 2))
            print(pc_influnce)
            print(pd.DataFrame(sorted(dic.items())))
            print('--------------------------------------------------------------------')

        return X, story, pd.DataFrame(sorted(dic.items()))[1].tolist()

    def fast_DBSCAN(self, X, eps_value, min_samples_value):

        db = DBSCAN(eps=eps_value, min_samples=min_samples_value).fit(X)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        return core_samples_mask, db.labels_

    def dbscan_clustering(self, data, eps_value, min_samples_value, no_of_components, target_Att='', reduction_method='tSNE', synthesis_flag=0, plot_title='', story=[], load_saved_data=0):
        # Generate sample data
        # if load_saved_data == 1:
        #    df_unsuper = pd.read_csv("df_unsuper.csv")
        #    return df_unsuper, []

        df = data.copy()
        y_tar = pd.DataFrame()
        if target_Att != '':
            y_tar = df[target_Att]
            df = df.drop(target_Att, axis=1)

        print('eps_value: ', eps_value)
        print('min_samples_value: ', round(min_samples_value), " out of ", len(df), " data points")
        print('--------------------------------------------------------------------')

        # PCA Analysis
        print("DBCAN colunms: ", df.columns)
        X, story, important_col = self.PCA_reduction(df, no_of_components, story, reduction_method)
        core_samples_mask, dbscan_lbls = self.fast_DBSCAN(X, eps_value, min_samples_value)

        # Number of clusters in dbscan_lbls, ignoring noise if present.
        n_clusters_ = len(set(dbscan_lbls)) - (1 if -1 in dbscan_lbls else 0)
        n_noise_ = list(dbscan_lbls).count(-1)

        print('Estimated number of clusters: %d' % n_clusters_)
        print('Estimated number of noise points: %d' % n_noise_)
        print('--------------------------------------------------------------------')

        story = self.DBSCAN_plot(df, no_of_components, dbscan_lbls, X, y_tar, core_samples_mask, n_clusters_, plot_title, eps_value, min_samples_value, story)

        df['dbscan_lbls'] = dbscan_lbls

        # return df_unsuper, story

    def DBSCAN_plot(self, df_data, no_of_components, dbscan_lbls, X, y_tar, core_samples_mask, n_clusters_, plot_title, eps_value, min_samples_value, story):
        ''' DBSCAN PLOT '''
        # Black removed and is used for noise instead.
        fig, ax = plt.subplots(figsize=(12, 12))
        if no_of_components == 3:
            ax = plt.axes(projection='3d')
        unique_dbscan_lbls = set(dbscan_lbls)
        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_dbscan_lbls))]
        for k, col in zip(unique_dbscan_lbls, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]

            class_member_mask = (dbscan_lbls == k)

            xy = X[class_member_mask & core_samples_mask]
            xy_tar = y_tar[class_member_mask & core_samples_mask]

            if y_tar.empty is False:
                class_label = y_tar.iloc[k]
                if class_label == 1:
                    marker_shape = "o"
                elif class_label == 0:
                    marker_shape = "v"

            if no_of_components == 3:
                plt.plot(xy[:, 0], xy[:, 1], xy[:, 2], 'o', markerfacecolor=tuple(col),
                         markeredgecolor='k', markersize=14, marker=marker_shape)

                xy = X[class_member_mask & ~core_samples_mask]
                plt.plot(xy[:, 0], xy[:, 1], xy[:, 2], 'o', markerfacecolor=tuple(col),
                         markeredgecolor='k', markersize=6, marker=marker_shape)
            else:

                class_pos = (xy_tar == 1)
                class_neg = (xy_tar == 0)
                xy_pos = xy[class_pos]
                xy_neg = xy[class_neg]

                plt.plot(xy_pos[:, 0], xy_pos[:, 1], 'o', markerfacecolor=tuple(col),
                         markeredgecolor='k', markersize=4, marker="o")
                plt.plot(xy_pos[:, 0], xy_pos[:, 1], 'o', markerfacecolor=tuple(col),
                         markeredgecolor='k', markersize=4, marker="v")

                xy = X[class_member_mask & ~core_samples_mask]
                xy_tar = y_tar[class_member_mask & ~core_samples_mask]
                class_pos = (xy_tar == 1)
                class_neg = (xy_tar == 0)
                xy_pos = xy[class_pos]
                xy_neg = xy[class_neg]
                plt.plot(xy_pos[:, 0], xy_pos[:, 1], 'o', markerfacecolor=tuple(col),
                         markeredgecolor='k', markersize=4, marker="o")
                plt.plot(xy_neg[:, 0], xy_neg[:, 1], 'o', markerfacecolor=tuple(col),
                         markeredgecolor='k', markersize=4, marker="v")

        if 2 == 1:
            for i, xy_point in enumerate(X):
                if no_of_components == 3:
                    ax.text(xy_point[0]+0.05, xy_point[1]+0.05, xy_point[2]+0.05, str(i+1))
                else:
                    if y_tar.iloc[i] == 1:
                        # plt.plot(xy[:, 0], xy[:, 1], xy[:, 2], 'o', markerfacecolor='purple',
                        #     markeredgecolor='k', markersize=6)
                        plt.annotate("x", (xy_point[0]+0.025, xy_point[1]+0.025))

        plt.title('Estimated number of clusters: %d' % n_clusters_ + ' - ' + plot_title)
        # PDFge = PDF_generator()
        # story, buf = PDFge.image_in_PDF(story, plt, y=6)
        plt.show()
        # fileName = aggr_appliances.replace(" ", "_")
        # fileName = fileName.replace(":", "-")
        # fig.savefig('plots/dbscan__' + str(len(futures_df)) + '_' + fileName + '.png', facecolor=fig.get_facecolor(), dpi=600, bbox_inches='tight')
        # fig.savefig(self.pth_REFIT + 'dbscan__' + plot_title + '_plusHours'
        #            + '_eps' + str(eps_value)
        #            + 'minSamples' + str(round(min_samples_value)) + '.png', facecolor=fig.get_facecolor(), dpi=600, bbox_inches='tight')
        return story
