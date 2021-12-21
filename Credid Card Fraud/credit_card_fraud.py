import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats.mstats import winsorize
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import os
import math
import sys

sys.path.append("..")

from EDA_ML_Package.PDF_report import PDF_reporting
from EDA_ML_Package.EDA_functions import Data_Analysis
from EDA_ML_Package.ML_functions import ML_models
from EDA_ML_Package.NN_functions import ANN_tabular_class

from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from imblearn.over_sampling import SMOTE

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 15)


# ------------------- Importing Data and classes ------------------------------

ML = ML_models()
DA = Data_Analysis()
ΝΝ = ANN_tabular_class()
PDF = PDF_reporting()

# ----------------------------------------------------------------------------

df = pd.read_csv("creditcard.csv")


df1 = df[df.Class == 0]
df1 = df1.sample(n=20200, random_state=1)

df0 = df[df.Class == 1]
df = pd.concat([df1, df0])

# df = df.drop('Class', axis=1)
# df = df.drop('Time', axis=1)

# ML.dbscan_clustering(df, 0.5, 100, 2, 'Class', reduction_method='PCA')

# sys.exit()
# ----------------- Visuals ----------------------
plt_obj = DA.hist_class_EDA_plots(df, "Class", no_rows=3)
# PDF.image_in_PDF(plt_obj, x=7, y=7)

plt_obj = DA.heatmap_corr_plot(df)
# PDF.image_in_PDF(plt_obj, x=7, y=7)

plt_obj = DA.bar_hist_EDA_plots(df, 'hist', no_rows=3)
# PDF.image_in_PDF(plt_obj, x=7, y=7)

plt_obj = DA.bar_hist_EDA_plots(df, no_rows=3)
# PDF.image_in_PDF(plt_obj, x=7, y=7)


Total_transactions = len(df)
normal = len(df[df.Class == 0])
fraudulent = len(df[df.Class == 1])
fraud_percentage = round(fraudulent/normal*100, 2)
print('\nTotal number of REDUCED Trnsactions are {}'.format(Total_transactions))
print('Number of Normal Transactions are {}'.format(normal))
print('Number of fraudulent Transactions are {}'.format(fraudulent))
print('Percentage of fraud Transactions is {}'.format(fraud_percentage))
# -----------------------------------------------------------------------------

DA.descriptive_analysis(df)
# -----------------------------------------------------------------------------

sc = StandardScaler()
amount = df['Amount'].values
df['Amount'] = sc.fit_transform(amount.reshape(-1, 1))

df = df.drop('Time', axis=1)

df.drop_duplicates(inplace=True)
print('\nTotal number of REDUCED Trnsactions are {}'.format(Total_transactions))

X = df.drop(['Class'], 1)  # [['Pclass', 'Sex', 'Age']]
y = df['Class']

results = ML.ML_Basic_Models(X, y)
results_NN = ΝΝ.create_and_fit_model(X, y, 10, 350, lyrs=[10, 7, 3], verbose=0, dr=0.3)
results_NN = ΝΝ.create_and_fit_model(X, y, 10, 350, lyrs=[5, 2], verbose=0, dr=0)

# PDF.generate_report('Credit Card Fraud')


print("\nBefore OverSampling, counts of label '1': {}".format(sum(y == 1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(y == 0)))

sm = SMOTE(random_state=2)
X_sm, y_sm = sm.fit_resample(X, y)

print("After OverSampling, counts of label '1': {}".format(sum(y_sm == 1)))
print("After OverSampling, counts of label '0': {} \n".format(sum(y_sm == 0)))
results = ML.ML_Basic_Models(X_sm, y_sm)
