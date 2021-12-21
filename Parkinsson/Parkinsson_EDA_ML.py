import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
from EDA_ML_Functions.EDA_functions import Data_Analysis
from EDA_ML_Functions.ML_functions import ML_models
from EDA_ML_Functions.NN_functions import ANN_tabular_class


def histogram_classification(df, target_att, att):
    # Normalize
    kwargs = dict(alpha=0.5, bins=20, density=True, stacked=True)
    colors = ['b', 'r', 'g']

    # Plot
    plt.subplots(figsize=(10, 8))

    for i, val in enumerate(df.status.unique().tolist()):
        plt.hist(df.loc[df[target_att] == val, att].values, **kwargs, label=str(val), color=colors[i])
    plt.gca().set(title=att, ylabel='Probability')
    plt.legend()
    # plt.title('TBC', fontsize=16)
    # plt.ylabel("Probability")
    
    plt.show()


pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 15)

df = pd.read_csv("Parkinsson/Parkinsson disease.csv")

ML = ML_models()
DA = Data_Analysis()
ΝΝ = ANN_tabular_class()

histogram_classification(df, "status", "spread1")
DA.hist_class_EDA_plots(df, "status", no_rows=3)
plt_obj = DA.bar_hist_EDA_plots(df, 'hist', no_rows=3)
plt_obj = DA.bar_hist_EDA_plots(df, no_rows=3)
DA.heatmap_corr_plot(df)

# sys.exit()

DA.descriptive_analysis(df)

nht = sns.heatmap(df.isnull(), cmap="viridis", yticklabels=False, cbar=False, )
plt.show()
nht.set_xticklabels(nht.get_xmajorticklabels(), fontsize=18)



sns.pairplot(df, x_vars=["spread1"], y_vars=["spread2"],
             hue="status", markers=["o", "x"], height=8, kind="reg")

sns.pairplot(df, x_vars=["MDVP:Fhi(Hz)"], y_vars=["MDVP:Flo(Hz)"],
             hue="status", markers=["o", "x"], height=8, kind="reg")

# ----------------------------------------------------------------------------


X = df.drop('name', axis = 1)
X = X.drop('status', axis = 1)
y = df['status']

results = ML.ML_Basic_Models(X, y)
results = ML.ML_Basic_Models(X, y)
results_NN = ΝΝ.create_and_fit_model(X, y, 10, 350, lyrs=[10, 7, 3], verbose=0, dr=0.3)
results_NN = ΝΝ.create_and_fit_model(X, y, 10, 350, lyrs=[5, 2], verbose=0, dr=0)

results_opt_RF, best_params, random_grid = ML.opt_ramdom_forest(X, y, n_iter=20, verbose=0)
