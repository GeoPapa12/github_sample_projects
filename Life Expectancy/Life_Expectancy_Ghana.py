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
import warnings
import sys

sys.path.append("..")
from EDA_ML_Package.EDA_functions import Data_Analysis
from EDA_ML_Package.PDF_report import PDF_reporting

warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 15)

# ------------------- Functions ------------------------------


def impute_dataset(df):
    for ctr in df.Country.unique():
        df_country = df[df.Country == ctr]
        for clm in df.columns:
            try:
                df.loc[df.Country == ctr, clm] = df_country[clm].fillna(df_country[clm].dropna().mean())
            except Exception:
                pass

    return df


def spec_country_among_others(countryX, category_field, x=6, y=3):
    colors_blue = ["#132C33", "#264D58", '#17869E', '#51C4D3', '#B4DBE9']
    colors_dark = ["#1F1F1F", "#313131", '#636363', '#AEAEAE', '#DADADA']
    colors_red = ["#331313", "#582626", '#9E1717', '#D35151', '#E9B4B4']

    countryX_region = df[df.Country == countryX]['Region'].dropna().iloc[0]
    df_region = df[df['Region'].str.contains(countryX_region).fillna(False)]

    df_region = df_region.reset_index(drop=True)
    df_region = df_region.sort_values(by=[category_field], ascending=False)
    df_region['Rank'] = list(range(1, len(df_region) + 1))
    country_pos = df_region[df_region['Country'] == countryX]['Rank'].iloc[0]
    # df_countryX = df_region.iloc[country_pos-3:country_pos+3]
    mean_score = df_region[category_field].mean()

    top_bottom_pos = 2
    countries_in_middle = 5
    df_region_top = df_region.head(top_bottom_pos)
    df_region_bot = df_region.tail(top_bottom_pos)
    upper_limit = country_pos-countries_in_middle
    if upper_limit < 0:
        upper_limit = 0
    lower_limit = country_pos+countries_in_middle
    df_region_mid = df_region.iloc[upper_limit:lower_limit]
    df_region_mid = df_region_mid[~df_region_mid.index.isin(df_region_top.index)]
    df_region_mid = df_region_mid[~df_region_mid.index.isin(df_region_bot.index)]

    df_region_filtered = pd.concat([df_region_top, df_region_mid, df_region_bot])
    # df_region_filtered = df_region_filtered.drop_duplicates()
    country_region_pos = df_region_filtered.reset_index()[df_region_filtered.reset_index().Country == countryX].index[0]

    fig, ax = plt.subplots(figsize=(14, 8))
    bars0 = ax.bar(df_region_top['Country'], df_region_top[category_field], color=colors_blue[0], alpha=0.6, edgecolor=colors_dark[0])
    bars1 = ax.bar(df_region_mid['Country'], df_region_mid[category_field], color=colors_dark[3], alpha=0.4, edgecolor=colors_dark[0])
    bars2 = ax.bar(df_region_bot['Country'], df_region_bot[category_field], color=colors_red[0], alpha=0.6, edgecolor=colors_dark[0])
    ax.axhline(mean_score, linestyle='--', color=colors_dark[2])

    (bars0 + bars1 + bars2)[country_region_pos].set_alpha(1)
    (bars0 + bars1 + bars2)[country_region_pos].set_color(colors_red[3])
    (bars0 + bars1 + bars2)[country_region_pos].set_edgecolor(colors_dark[0])

    ax.legend(["Average", "Top", "Other", "Bottom"], loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=5, borderpad=1, frameon=False, fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_xlabel("Countries", fontsize=14, labelpad=10, fontweight='bold', color=colors_dark[0])
    ax.set_ylabel(category_field, fontsize=14, labelpad=10, fontweight='bold', color=colors_dark[0])
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    i = 0
    for barX in [bars0, bars1, bars2]:
        for bar in barX:
            x = bar.get_x()
            y = bar.get_height()
            ax.text(
                s=f"{df_region_filtered['Rank'].iloc[i]}th",
                va='center', ha='center',
                x=x+0.38, y=y/2,
                color=colors_blue[0],
                fontsize=14,)
            i += 1

    plot_text = countryX + " is in " + str(df_region_filtered.iloc[country_region_pos].Rank)
    plot_text = plot_text + " position out of " + str(len(df_region)) + " countries in " + countryX_region + "\n"
    plot_text = plot_text + "Category: " + category_field
    plt.text(s=plot_text, ha='left', x=xmin, y=ymax*1.05, fontsize=16, fontweight='bold', color=colors_dark[0])
    # plt.title("How Happy is " + countryX + " in " + countryX_region, loc='left', fontsize=13, color=colors_dark[2])
    plt.tight_layout()
    PDF.image_in_PDF(plt, x=6, y=3)
    plt.show()


# ------------------- Instances ------------------------------


DA = Data_Analysis()
PDF = PDF_reporting()

# ------------------- Loading Data ------------------------------

df = pd.read_csv("Life Expectancy Data.csv")
data = pd.read_csv('2015.csv')
data = data[['Country', 'Region', 'Happiness Rank', 'Happiness Score']]
data['Year'] = 2015
df = pd.merge(df, data, how="left", on=["Country", "Year"])

# Cleansing Column names
for clm in df.columns:
    clm_new = clm.replace("-", "_")
    if clm_new[-1] == " ":
        clm_new = clm_new[:-1]
    df = df.rename(columns={clm: clm_new})


# ------------------- Loading Data ------------------------------

DA.descriptive_analysis(df, describe_opt=0)
DA.nan_heatmap(df)
DA.heatmap_corr_plot_2(df)

# df_mean = df.groupby(['Country']).mean()
# DA.heatmap_corr_plot_2(df_mean)

# ---------------------------------------------------------
df_Ghana = df[df.Country == "Ghana"]

df_sub_africa = df[df.Region == "Sub-Saharan Africa"]
df_sub_africa = df_sub_africa.fillna(0)
df_sub_africa['TarCountry'] = "Not Ghana"
df_sub_africa.loc[df_sub_africa.Country == "Ghana", 'TarCountry'] = "Ghana"

# ---------------------------------------------------------
plt_obj = DA.box_hist_EDA_plots(df_Ghana, no_rows=5)
PDF.add_text("Box Plot", style="Heading1", fontsize=24)
PDF.image_in_PDF(plt_obj, x=7, y=4)

img_hist_obj = DA.box_hist_EDA_plots(df_sub_africa, plot_type='hist', target_att="TarCountry", density_v=True, no_rows=8)
PDF.add_text("Density Histogram Plot", style="Heading3", fontsize=14)
PDF.image_in_PDF(img_hist_obj, x=7, y=4)


sys.exit('')
# ------------------- EDA ------------------------------

df_imp = impute_dataset(df)

df = DA.outlier_winsorize(df, ['GDP'])  # modify outliers
df = impute_dataset(df)
plt_obj = DA.box_hist_EDA_plots(df, no_rows=3)
PDF.add_text("Box Plot", style="Heading1", fontsize=24)
PDF.image_in_PDF(plt_obj, x=7, y=4)
plt_obj = DA.box_hist_EDA_plots(df, 'hist', no_rows=3)
PDF.add_text("Hist Plot", style="Heading1", fontsize=24)
PDF.image_in_PDF(plt_obj, x=7, y=4)

sns.pairplot(df, x_vars=["Hepatitis B"], y_vars=["Life expectancy"],
             hue="Status", markers=["o", "x"], height=8, kind="reg")
sns.pairplot(df, x_vars=["Happiness Score"], y_vars=["Life expectancy"],
             hue="Status", markers=["o", "x"], height=8, kind="reg")
g = sns.jointplot(x=df["Happiness Score"], y=df["Life expectancy"], hue=df["Status"], height=6)
# g = (g.set_axis_labels("Tuition and Fees $","Applications"))

df_region_mean = df.groupby(['Region']).mean()[['Life expectancy', 'Happiness Score']]

fig, axes = plt.subplots(nrows=1, ncols=2, constrained_layout=True, figsize=(14, 8))
sns.barplot(x='Life expectancy', y=df_region_mean.index, data=df_region_mean, ax=axes[0], palette='Spectral')
sns.barplot(x='Happiness Score', y=df_region_mean.index, data=df_region_mean, ax=axes[1], palette='RdYlGn')

PDF.add_text("Ghana In Africa", style="Heading1", fontsize=24, page_break=1)
spec_country_among_others("Ghana", "Alcohol", x=7, y=2)
spec_country_among_others("Ghana", "Happiness Score", x=7, y=2)
spec_country_among_others("Ghana", "Life expectancy", x=7, y=2)
spec_country_among_others("Ghana", "Schooling", x=7, y=2)

PDF.generate_report('Ghana')
