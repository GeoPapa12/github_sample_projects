import pandas as pd
import numpy as np
import math

# ------------------- Loading Data ------------------------------

df = pd.read_csv("Life Expectancy Data - Original.csv")
df_org = pd.read_csv("Life Expectancy Data - Original.csv")
data15 = pd.read_csv('2015.csv')

data15['Year'] = 2015

df = pd.merge(df, data15, how="left", on=["Country", "Year"])
# df['Region'] = df['Region'].fillna(method='ffill')


for yr in []:
    data = pd.read_csv(str(yr) + '.csv')
    data['Year'] = yr

    xx = [[x, yr] for x in df.Country.unique().tolist()]
    xx = pd.DataFrame(xx, columns=['Country', 'Year'])
    df = pd.merge(df, xx, how="outer", on=["Country", "Year"])
    df = pd.merge(df, data, how="left", on=["Country", "Year", 'Region'])


#df = df.drop(df[(df['Region'].isnull()) & (df['Year'] == 2015)].index)
df = df[df.Country.isin(df[df['Year'] == 2015].Country)]
df = df[df.Country.isin(data15.Country.unique())]
df['Region'] = df['Region'].fillna(method='ffill')

df.to_csv('Life Expectancy Data.csv', index=False)