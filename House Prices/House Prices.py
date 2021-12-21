import pandas as pd # conventional alias
import seaborn as sns
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from EDA_ML_Functions.EDA_functions import Data_Analysis
from EDA_ML_Functions.ML_functions import ML_models
from EDA_ML_Functions.NN_functions import ANN_tabular_class


pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 15)

# ------------------- Importing Data and classes ------------------------------

DA = Data_Analysis()
ML = ML_models()
ΝΝ = ANN_tabular_class()

df = pd.read_csv('kc_house_data.csv')

# ------------------- Data Cleansing ------------------------------

df = df.drop('id', axis=1)
df = df.drop('date', axis=1)
df = df.drop('zipcode', axis=1)
DA.descriptive_analysis(df)

X = df.drop(['price'], 1)
y = df['price']

# ------------------- EDA 1 ------------------------------

DA.descriptive_analysis(df)
DA.nan_heatmap(df)
DA.heatmap_corr_plot_2(df)

DA.bar_hist_EDA_plots(df, no_rows=2)
DA.bar_hist_EDA_plots(df, 'hist', no_rows=2)
sns.jointplot(df.sqft_living, df.price, alpha=0.5)

# ------------------- Outliers ------------------------------

df = DA.outlier_winsorize(df, list_exclude=['long', 'lat', 'zipcode', 'yr_renovated', 'yr_built'])

# ------------------- EDA 2 ------------------------------

# sns.pairplot(df[[]], size=2.5)
print(df.describe())

plt.figure(figsize=(8, 8))
sns.jointplot(df.sqft_living, df.price, alpha=0.5)
plt.xlabel('Sqft Living')
plt.ylabel('Sale Price')
plt.show()

non_top_1_perc = df.sort_values('price', ascending=False).iloc[int(len(df)*0.01):]
plt.figure(figsize=(12, 8))
sns.scatterplot(x='long', y='lat',
                data=non_top_1_perc, hue='price',
                palette='RdYlGn', edgecolor=None, alpha=0.2)

X = df.drop(['price'], 1)
y = df['price']

bath = ['bathrooms', 'bedrooms']
cm = sns.light_palette("green", as_cmap=True)
pd.crosstab(df[bath[0]], df[bath[1]]).style.background_gradient(cmap=cm)


# ------------------- ML Analysis ------------------------------
sys.exit()

results_NN = ΝΝ.create_and_fit_model(X, y, batch_size=10, epochs=400, lyrs=[19, 19, 19, 19], dr=0, NNtype='Regression')

resultsML = ML.ML_Basic_Models_Regression(X, y)

