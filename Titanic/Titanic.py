import sys

sys.path.append("..")
from EDA_ML_Package.EDA_functions import Data_Analysis
from EDA_ML_Package.ML_functions import ML_models
from EDA_ML_Package.NN_functions import ANN_tabular_class

from pandas.api.types import is_numeric_dtype
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import sys


# ------------------- Importing Data and classes ------------------------------

ML = ML_models()
DA = Data_Analysis()
ΝΝ = ANN_tabular_class()

data = pd.read_csv("train.csv", dtype={"Age": np.float64}, )
data.drop(['PassengerId'], axis=1, inplace=True)

# ------------------- EDA -----------------------------------------------------

DA.descriptive_analysis(data, 'Survived')
DA.nan_heatmap(data)

DA.hist_plot(data)
DA.pie_plot(data, 'Survived', 1, 'Pclass', 'Sex', "Survived/ Sex/ PClass")
DA.pie_plot(data, 'Survived', 0, 'Pclass', 'Sex', "Deceased/ Sex/ PClass")
# DA.bar_plot_count_percentage(data, 'Sex', 'Survived', 'Target attribute distribution based on Sex', 'Both')
DA.heatmap_corr_plot(data)

DA.hist_class_EDA_plots(data, "Survived", no_rows=3)

# ------------------- Data Analysis -------------------------------------------
data["FamilySize"] = data["SibSp"] + data["Parch"]
data["Title"] = data["Name"].apply(lambda x: re.search('([A-Za-z]+)\.', x).group(1))
# data["TitleCat"], title_codes = DA.categorise_codes(data, "Title")

data.loc[(data['Sex'] == 'male') & (data['Age'] < 18), 'Sex'] = 'Male Child'
data.loc[(data['Sex'] == 'female') & (data['Age'] < 18), 'Sex'] = 'Female Child'
# DA.bar_plot_count_percentage(data, 'Sex', 'Survived', 'Target attribute distribution based on Sex', 'Both')

data.drop(['Cabin', 'Ticket'], axis=1, inplace=True)
data.drop(['Title', 'Name'], axis=1, inplace=True)

DA.descriptive_analysis(data, 'Survived')

# ------------------- ML Analysis ---------------------------------------------
# sys.exit()


X = data.drop(['Survived'], 1)  # [['Pclass', 'Sex', 'Age']]
y = data['Survived']

X_tr, pipeline_tr = DA.data_transformation(X)
DA.descriptive_analysis(X_tr)

results_NN, _ = ΝΝ.chain_optimazation(X_tr, y, hidden_layers=[[6], [6, 4], [6, 4, 2]])

sys.exit()

results = ML.ML_Basic_Models(X_tr, y)
results_NN = ΝΝ.create_and_fit_model(X_tr, y, 10, 350, lyrs=[10, 7, 3], verbose=0, dr=0.3)
results_NN = ΝΝ.create_and_fit_model(X_tr, y, 10, 350, lyrs=[5, 2], verbose=0, dr=0)


# results_opt_RF, best_params, random_grid = ML.opt_ramdom_forest(X_tr, y, n_iter=20, verbose=0)

# ------------------- Generating Reports ---------------------------------------------
DA.generate_report('Titanic')
ML.generate_report('Titanic')
