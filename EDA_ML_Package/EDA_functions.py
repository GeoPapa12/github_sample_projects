import pandas as pd
import numpy as np
import math
import copy
import sqlite3

import seaborn as sns
import matplotlib.pyplot as plt

import os

import io
from datetime import datetime

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer, Table, TableStyle
from reportlab.platypus import PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib import colors

from scipy.stats import t
from scipy import stats

from matplotlib.patches import Rectangle
from pandas.api.types import is_numeric_dtype
from pandas.api.types import is_string_dtype

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer  # , KNNImputer
from sklearn.pipeline import Pipeline
from sklearn import preprocessing

from sklearn.preprocessing import OrdinalEncoder


import warnings
from scipy.stats.mstats import winsorize

warnings.filterwarnings('ignore')


class Data_Analysis():

    def __init__(self):

        now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        self.Story = []
        self.styles = getSampleStyleSheet()
        # self.doc = SimpleDocTemplate("DA analysis " + nowTitle + ".pdf", pagesize=letter,
        #                             rightMargin=inch/2, leftMargin=inch/2,
        #                             topMargin=25.4, bottomMargin=12.7)

        self.add_text("Data Analysis", style="Heading1", fontsize=24)
        self.add_text(now)

    def add_text(self, text, style="Normal", fontsize=12):
        """ Adds text with some spacing around it to  PDF report

        Parameters
        ----------
        text : (str) The string to print to PDF
        style : (str) The reportlab style
        fontsize : (int) The fontsize for the text

        Output: -
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
            ('FONT', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('ALIGN', (0, 0), (0, -1), 'CENTER'),
            ('INNERGRID', (0, 0), (-1, -1), 0.50, colors.black),
            ('BOX', (0, 0), (-1, -1), 0.25, colors.black),
        ]))
        self.add_text("")
        self.add_text("")
        self.add_text("")
        self.Story.append(table)

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

    def generate_report(self, docTitle):
        """ Buids the PDF report

        Parameters
        ----------
        -

        Output: -
        """
        nowTitle = datetime.now().strftime("%d_%m_%Y %H-%M-%S")

        self.doc = SimpleDocTemplate("EDA " + str(docTitle) + "_" + nowTitle + ".pdf", pagesize=letter,
                                     rightMargin=inch/2, leftMargin=inch/2,
                                     topMargin=25.4, bottomMargin=12.7)

        self.doc.build(self.Story)

    def story_as_output(self):
        """

        Parameters
        ----------
        -

        Output: -
        """
        return self.Story

    def create_directory(self):
        """ Creates a folder if the folder does not exist

        Parameters
        ----------

        Output: -
        """
        try:
            # os.mkdir(self.path)
            pass
        except OSError:
            'file already exists'

    def size_df(self, df):
        """ Prints the size of a dataframe

        Parameters
        ----------
        -

        Output: -
        """
        print('\nnumber of attributes: ', len(df.columns), '/ number of instances: ', len(df), '\n')
        self.add_text('number of attributes: ' + str(len(df.columns)) + '/ number of instances: ' + str(len(df)))

    def descriptive_analysis(self, data, target_attribute='none'):
        """ A table is created with the characteristics of a dataframe
        The table is printed in the pdf report

        Parameters
        ----------
        df: (dataframe)
        target_attribute: (str) optional

        Output: table
        """
        print(data.describe())
        self.table_in_PDF(data.describe())
        df = round(data.copy(deep=True), 2)

        self.size_df(df)
        # tc: table characteristics
        tc = pd.DataFrame(df.isnull().sum(), columns=['NaN'])
        tc['Unique'] = [len(df[i].unique()) for i in list(df.columns)]
        tc['type'] = df.dtypes

        indexValue = 3
        tc['values'] = [df[clm].unique().tolist()[:indexValue] for clm in list(df.columns)]

        # for i, vl_list in enumerate(df.values):
        #    sm['values'].iloc[i] = [round(vl, 3) for vl in vl_list]

        for x, vl in enumerate(tc['values']):
            if tc['Unique'].iloc[x] > 3:
                vl.append('...')
            for i, y in enumerate(vl):
                if len(str(y)) > 10:
                    try:
                        vl[i] = y[:10] + '..'
                    except Exception:
                        pass
        tc = tc.reset_index()
        tc = tc.astype(str)

        table = self.table_in_PDF(tc)
        print(tc)

        if target_attribute != 'none':
            self.attribute_range(df, target_attribute)

        return table

    def attribute_range(self, data, target_attribute):
        """ A table is created with the range of values of an attribute
        The table is printed in the pdf report

        Parameters
        ----------
        df: (dataframe)
        target_attribute: (str)

        Output: -
        """
        df = data.copy(deep=True)
        dx = self.groupby_count_percentage(df, [target_attribute])
        print('\nValue Distribution of the Target attribute\n')
        print(dx)
        self.table_in_PDF(dx)
        return dx

    '''========================================================================
                                    Data Transformation
    ========================================================================'''

    def data_transformation(self, dfX, numerical_features=[], categorical_features=[], cat_type=0):

        def get_column_names_from_ColumnTransformer(column_transformer, numerical_features):
            col_name = []
            for transformer_in_columns in column_transformer.transformers_[:]:  # the last transformer is ColumnTransformer's 'remainder'
                raw_col_name = transformer_in_columns[2]
                if transformer_in_columns[0] == 'numerical':
                    raw_col_name = numerical_features
                if isinstance(transformer_in_columns[1], Pipeline):
                    transformer = transformer_in_columns[1].steps[-1][1]
                else:
                    transformer = transformer_in_columns[1]
                try:
                    names = transformer.get_feature_names()
                except AttributeError:  # if no 'get_feature_names' function, use raw column name
                    names = raw_col_name
                if isinstance(names, np.ndarray):  # eg.
                    col_name += names.tolist()
                elif isinstance(names, list):
                    col_name += names
                elif isinstance(names, str):
                    col_name.append(names)
            return col_name

        X = dfX.copy(deep=True)
        # Numerical attributes are detected if not defined
        if numerical_features == []:
            numerical_features = [cols for cols in X.columns if is_numeric_dtype(X[cols])]
        num_features_index = [dfX.columns.get_loc(c) for c in numerical_features if c in dfX]
        if categorical_features == []:
            categorical_features = [cols for cols in X.columns if cols not in numerical_features]
        cat_features_index = [dfX.columns.get_loc(c) for c in categorical_features if c in dfX]

        # =====================================================================
        # Applying SimpleImputer and will search for different scalers using GridSearchCV
        numerical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                                                ('scaler', MinMaxScaler())])

        if cat_type == 0:
            # Applying SimpleImputer and then OneHotEncoder
            categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                                                      ('onehot', OneHotEncoder(handle_unknown='ignore'))])
        else:
            categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                                                      ('onehot', OrdinalEncoder())])

        data_transformer = ColumnTransformer(transformers=[
            ('numerical', numerical_transformer, numerical_features),
            ('categorical', categorical_transformer, categorical_features)])

        # =====================================================================
        trasnformer = ('data_transformer', data_transformer)
        pipe_line = Pipeline([trasnformer])

        # =====================================================================

        X_fit = pipe_line.fit_transform(X)
        pipe_line.fit(X)
        col_names_pipeline = get_column_names_from_ColumnTransformer(data_transformer, numerical_features)

        df_fit = round(pd.DataFrame(X_fit, columns=col_names_pipeline), 3)
        return df_fit, pipe_line

    '''========================================================================
                                    Data Cleaning
    ========================================================================'''

    def categorise_codes(self, df, columnName):
        """ Replaces string values with numbers, eg Male->0, Female ->1

        Parameters
        ----------
        df: (dataframe)
        columnName: (str)
        charLimit: (int) How many char to take under consideration

        Output: data in dataframe form
        """
        # self.add_text("> Codes Categories creation for " + columnName, style="Heading3", fontsize=13)

        # df[columnName] = df[columnName].astype(str)
        print(columnName)

        cat_values = df[columnName].astype("category").cat.codes.astype('category')
        map_dict = dict(enumerate(df[columnName].astype('category').cat.categories))

        return cat_values, map_dict

    def place_column_in_bins(self, df, columnName, weight=0.5):
        """ Creates new values by grouping arithmetical data eg [0-2),[2-4)

        Parameters
        ----------
        df: (dataframe)
        columnName: (str)

        Output: df[columnName]
        """
        self.add_text("> Bins creation for " + columnName, style="Heading3", fontsize=13)

        percentge_group = df.groupby(columnName).size() / len(df[columnName]) * 100

        percentge_group_values = percentge_group.values.tolist()
        percentge_group_index = percentge_group.index.tolist()
        max_percentge_group_values = max(percentge_group_values)

        bins = []
        # weight = 0.5
        while len(bins) == 0 or len(bins) > 7:
            bins = []
            cumSum = percentge_group_values[0]
            for j in range(1, len(percentge_group_values)):
                if (cumSum < max_percentge_group_values * weight):
                    cumSum = cumSum + percentge_group_values[j]
                else:
                    bins.append(percentge_group_index[j-1])
                    cumSum = 0
            weight = weight + 0.5

        bins.append(percentge_group_index[-1])
        bins.append(-1)
        bins = list(set(bins))
        bins.sort()

        df[columnName] = pd.cut(df[columnName], bins)
        df[columnName] = df[columnName].astype(str)

        self.add_text('Values from "' + columnName + '" atrribute splited in ' + str(len(bins)-1) + ' categories', fontsize=12)
        print('Values from "' + columnName + '" atrribute splited in ' + str(len(bins)-1) + ' categories')
        return df[columnName]


    '''========================================================================
                                    Visualization
    ========================================================================'''

    def nan_heatmap(self, data):
        """ Creates a bar plot that compares 2 attributes, eg a random att vs
        the target attribute

        Parameters
        ----------
        df: (dataframe)

        Output:
        """
        fig, axes = plt.subplots(figsize=(10, 8))
        clmns = data.columns.tolist()
        if len(clmns) > 10:
            clmns1 = clmns[0:round(len(clmns)/2)]
            nht1 = sns.heatmap(data[clmns1].isnull(), cmap="viridis", yticklabels=False, cbar=False)
            plt.show()
            nht1.set_xticklabels(nht1.get_xmajorticklabels(), fontsize=18)
            figure = nht1.get_figure()
            buf = self.image_in_PDF(figure, y=4)

            fig, axes = plt.subplots(figsize=(10, 8))
            clmns2 = clmns[round(len(clmns)/2):-1]
            nht2 = sns.heatmap(data[clmns2].isnull(), cmap="viridis", yticklabels=False, cbar=False)
            plt.show()
            nht2.set_xticklabels(nht2.get_xmajorticklabels(), fontsize=18)
            figure = nht2.get_figure()
            buf = self.image_in_PDF(figure, y=4)
        else:
            nht = sns.heatmap(data[clmns].isnull(), cmap="viridis", yticklabels=False, cbar=False, )
            plt.show()
            nht.set_xticklabels(nht.get_xmajorticklabels(), fontsize=18)
            figure = nht.get_figure()
            buf = self.image_in_PDF(figure, y=4)

        return buf

    def kde_plot(self, df, att1, att2, value11, value12):
        """ Creates a bar plot that compares 2 attributes, eg a random att vs
        the target attribute

        Parameters
        ----------
        df: (dataframe)

        Output: buf (to be printed in the pdf)
        """
        plt.subplots(figsize=(10, 8))
        sns.kdeplot(df[df[att1] == value11][att2], Label=value11, shade=True, color="r")
        sns.kdeplot(df[df[att1] == value12][att2], Label=value12, shade=True, color="g")
        plt.xlabel(att2)
        plt.ylabel('Probability Density')
        plt.tight_layout()
        plt.show()

    def groupby_count_percentage(self, df, listOfColumns):
        """ Creates a dataset that shows the distribution of attributes in count
        and percentage form. It is used in other functions

        Parameters
        ----------
        data: (dataframe)
        listOfColumns: (list) 1 to multiple attributes

        Output: data in dataframe form
        """

        data = df.copy(deep=True)
        data['count'] = 0
        data['percentage'] = 0

        if len(listOfColumns) == 1:
            df_gb = data.groupby(listOfColumns)[['count', 'percentage']].count().reset_index()
        else:
            df_gb = data.groupby(listOfColumns)[['count', 'percentage']].count().unstack().stack(dropna=False)
            df_gb = df_gb.reset_index()
        df_gb['percentage'] = (df_gb['percentage'] / df_gb['count'].sum() * 100).round(2)

        df_gb[listOfColumns[0]] = df_gb[listOfColumns[0]].astype(str)

        df_gb['Pop Per'] = 0
        ii = 0
        groupList = df_gb[listOfColumns[0]].to_list()
        groupList = [str(i) for i in groupList]

        for y in groupList:

            # df_gb.loc[ii, 'Pop Per'] = df_gb.groupby(listOfColumns[0]).sum().loc[y][0]
            df_gb.loc[ii, 'Pop Per'] = df_gb.groupby(listOfColumns[0]).sum()['count'].loc[y]
            ii = ii + 1

        df_gb['Pop Per'] = (df_gb['count']/df_gb['Pop Per']*100).round(2)

        data = data.drop(['percentage', 'count'], axis=1)
        return df_gb

    def bar_plot_count_percentage(self, data, att, attTarget, title, plotType='Both'):
        """ Creates a bar plot that compares 2 attributes, eg a random att vs
        the target attribute

        Parameters
        ----------
        df: (dataframe)
        att: (str)
        attTarget: (str)
        path: (str) path where the plot will be saved

        Output: buf (to be printed in the pdf)
        """
        self.add_text("Bar Plot: " + att + " vs " + attTarget)

        df = data.copy(deep=True)
        gb = self.groupby_count_percentage(df, [att, attTarget])

        x = gb[att].str.split(",", n=1, expand=True)
        y = x[0].map(lambda x: x.lstrip('(').rstrip('aAbBcC'))
        # #    y = y.astype('float64')
        y = y.sort_values(ascending=True)
        gb.iloc[y.index.tolist()]

        sns.set_style("whitegrid")

        subSize = 2 if plotType == 'Both' else 1

        fig, axes = plt.subplots(subSize, 1, figsize=(14, 8))

        if plotType == 'Both':
            sp = sns.barplot(x=att, y="count", hue=attTarget, data=gb, ax=axes[0])
        if plotType == 'Count':
            sp = sns.barplot(x=att, y="count", hue=attTarget, data=gb)

        if plotType != 'Per':
            _ = plt.setp(sp.get_xticklabels(), rotation=90)  # Rotate labels
            sp.set_yticklabels(sp.get_yticks(), size=15)
            sp.set_xticks([])
            sp.set_xlabel('')
            sp.set_title(title, fontsize=24)
            sp.set_ylabel('Count', fontsize=18)
            sp.legend(prop={'size': 16})

            for p in sp.patches:
                height = p.get_height()
                height = 0 if math.isnan(height) is True else height
                height = int(height)
                sp.text(p.get_x() + p.get_width()/2., height + 3, height, ha="center", fontsize=15)

        if plotType == 'Both':
            sp = sns.barplot(x=att, y="Pop Per", hue=attTarget, data=gb, ax=axes[1])
        elif plotType == 'Per':
            sp = sns.barplot(x=att, y="Pop Per", hue=attTarget, data=gb)

        if plotType != 'Count':
            for ii, p in enumerate(sp.patches):
                height = p.get_height()
                height = round(height, 1)
                sp.text(p.get_x() + p.get_width()/2., height + 3, height, ha="center", fontsize=15)

            sp.set_xlabel(att, fontsize=18)
            sp.set_ylabel('Percentage', fontsize=18)
            sp.set_yticklabels(sp.get_yticks(), size=15)
            # sp.set_xticklabels(sp.get_xticks(), size=15)
            sp.tick_params(labelsize=16)
            sp.legend(prop={'size': 16})

        fig.tight_layout()

        self.create_directory()

        figure = sp.get_figure()
        buf = self.image_in_PDF(figure, y=4)
        plt.show()
        # figure.savefig(self.path + '/' + att + '.png', dpi=500, bbox_inches='tight')
        plt.close(figure)

        return buf

    def hist_plot(self, data):
        """ Creates a histogram of the attributes

        Parameters
        ----------
        data: (dataframe)
        path: (str) path where the plot will be saved

        Output: buf (to be printed in the pdf)
        """
        self.create_directory()

        data.hist(figsize=(16, 20), bins=50, xlabelsize=12, ylabelsize=12)
        # plt.savefig(self.path + '/hist.png', dpi=500, bbox_inches='tight')
        buf = self.image_in_PDF(plt, y=7)
        plt.show()

        return buf

    def heatmap_corr_plot(self, data, listColmRemove=[]):
        """ Creates a heatmap of the attributes

        Parameters
        ----------
        data: (dataframe)
        path: (str) path where the plot will be saved

        Output: buf (to be printed in the pdf)
        """
        self.create_directory()

        self.add_text("Correlation - Heatmap")

        data_columns = data.columns.tolist()

        if listColmRemove != []:
            for colm in listColmRemove:
                data_columns.remove(colm)

        data = data[data_columns]

        corr_matrix = data.corr()
        mask = np.zeros_like(corr_matrix, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True

        if len(data_columns) <= 12:
            fig, ax = plt.subplots(figsize=(20, 10))
        elif len(data_columns) <= 24:
            fig, ax = plt.subplots(figsize=(30, 15))
        elif len(data_columns) > 24:
            fig, ax = plt.subplots(figsize=(50, 25))

        ax = sns.heatmap(round(corr_matrix, 2),
                         mask=mask,
                         square=True,
                         linewidths=.5,
                         cmap='coolwarm',
                         cbar_kws={'shrink': .4, 'ticks': [-1, -.5, 0, 0.5, 1]},
                         vmin=-1,
                         vmax=1,
                         annot=True,
                         annot_kws={'size': 15})
        ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize=14)
        ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize=14)
        plt.yticks(rotation=0)

        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=90,
            horizontalalignment='center')

        # plt.savefig(self.path + '/heatmap_corr.png', dpi=500, bbox_inches='tight')

        buf = self.image_in_PDF(plt, y=7, x=10)

        return fig

    def heatmap_corr_plot_2(self, data, listColmRemove=[]):

        self.create_directory()

        self.add_text("Correlation - Heatmap")

        data_columns = data.columns.tolist()

        if listColmRemove != []:
            for colm in listColmRemove:
                data_columns.remove(colm)

        data = data[data_columns]

        mask = np.triu(data.corr())
        plt.figure(figsize=(15, 6))
        sns.heatmap(round(data.corr(), 2), annot=True, fmt='.2g', vmin=-1, vmax=1, center=0, cmap='coolwarm', mask=mask)
        # plt.ylim(18, 0)
        plt.title('Correlation Matrix Heatmap')
        # plt.show()
        buf = self.image_in_PDF(plt, y=7)
        return buf

    def box_plot(self, df, attXaxis, attYaxis):
        """ Creates a box plot that compares 2 attributes, eg a random att vs
        the target attribute

        Parameters
        ----------
        df: (dataframe)
        attXaxis: (str) attribute on the X axis
        attYaxis: (str) attribute on the Y axis
        path: (str) path where the plot will be saved

        Output: buf (to be printed in the pdf)
        """
        # data = data.astype('float')
        plt.figure(figsize=(12, 5))
        sns.boxplot(x=attXaxis, y=attYaxis, data=df, palette='colorblind')
        plt.tight_layout()

        self.create_directory()
        # featImp.savefig(self.path + '/' + 'box_' + attXaxis + '_' + attYaxis + '.png', dpi=500, bbox_inches='tight')
        return 'nothing'

    def pie_plot(self, data, att1, value1, att2, att3, title):
        """ Creates a box plot that compares 2 attributes, eg a random att vs
        the target attribute

        Parameters
        ----------
        data: data
        att1: (str) main attribute
        value1: value to be ploted from att1
        att2: (str) secondary attribute
        att3: (str) secondary attribute
        title: (str) title

        Output: buf (to be printed in the pdf)
        """
        plt.figure()
        att_1_2_count_per = self.groupby_count_percentage(data, [att1, att2])
        att_1_2_3_count_per = self.groupby_count_percentage(data, [att1, att2, att3])

        group_names = att_1_2_count_per[att_1_2_count_per[att1] == str(value1)][att2].tolist()
        group_size = att_1_2_count_per[att_1_2_count_per[att1] == str(value1)]['Pop Per'].tolist()

        subgroup_names, subgroup_size = [], []
        for att2_i in att_1_2_count_per[att_1_2_count_per[att1] == str(value1)][att2].tolist():
            subgroup_names.append(att_1_2_3_count_per[(att_1_2_3_count_per[att1] == str(value1)) & (att_1_2_3_count_per[att2] == att2_i)][att3].tolist())
            subgroup_size.append(att_1_2_3_count_per[(att_1_2_3_count_per[att1] == str(value1)) & (att_1_2_3_count_per[att2] == att2_i)]['Pop Per'].tolist())

        import math

        subgroup_names = [item[0:10] for sublist in subgroup_names for item in sublist]
        subgroup_size = [item for sublist in subgroup_size for item in sublist]
        subgroup_size = [0 if math.isnan(x) else x for x in subgroup_size]

        clrs, d07set = [], []

        for i in range(0, len(data[att2].unique().tolist())):
            d = [plt.cm.Blues, plt.cm.Reds, plt.cm.Greens]
            start_clr = 0.65
            d07set.append(d[i](0.7))
            for r in data[att3].unique().tolist():
                clrs.append(d[i](start_clr))
                start_clr = start_clr - 0.15

        # First Ring (outside)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.axis('equal')
        mypie, _ = ax.pie(group_size, radius=1.3, labels=group_names, textprops={'fontsize': 14}, colors=d07set)
        plt.setp(mypie, width=0.3, edgecolor='white')

        # Second Ring (Inside)
        mypie, _ = ax.pie(subgroup_size, radius=1.3-0.3, labels=subgroup_names, textprops={'fontsize': 14}, labeldistance=0.7, colors=clrs)
        plt.setp(mypie, width=0.4, edgecolor='white')
        plt.margins(0, 0)

        ax.text(-0.28, 0, title, fontsize=14)
        plt.tight_layout()
        plt.show()

        buf = self.image_in_PDF(fig, y=7)

        fig = mypie[0].get_figure()
        # fig.savefig("donut 3.pdf", bbox_inches='tight')
        return buf

    def bar_hist_EDA_plots(self, df, plot_type='boxplot', no_rows=0):
        if no_rows == 0:
            no_rows = divmod(len(df.columns), 10)[0]
        fig = plt.figure(figsize=(25, 5*no_rows))
        for i, col in enumerate(df.columns, 1):
            if(df[col].dtype == np.float64 or df[col].dtype == np.int64):

                plt.subplot(no_rows, math.ceil(len(df.columns)/no_rows), i)
                if plot_type == 'boxplot':
                    plt.boxplot(df[~df[col].isnull()][col])
                    ax = plt.gca()
                    ax.axes.xaxis.set_ticklabels([])
                else:
                    plt.hist(df[~df[col].isnull()][col])

                plt.title(col, fontsize=20)
            else:
                plt.subplot(no_rows, math.ceil(len(df.columns)/no_rows), i)
                if len(df[col].unique()) <= 5:
                    sp = sns.countplot(x=col, data=df)
                    plt.setp(sp.get_xticklabels(), ha="right", rotation=25, fontsize=8)
                    sp.set(xlabel=None)
                else:
                    ax = plt.gca()
                    ax.axes.xaxis.set_ticklabels([])
                plt.title(col, fontsize=20)
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
        plt.show()
        return fig

    def hist_class_EDA_plots(self, df, target_att, no_rows=0):
        if no_rows == 0:
            no_rows = divmod(len(df.columns), 10)[0]
        fig = plt.figure(figsize=(25, 5*no_rows))

        kwargs = dict(alpha=0.5, bins=20, density=True, stacked=True)
        colors = ['b', 'r', 'g']

        # Plot
        for j, col in enumerate(df.columns, 1):
            plt.subplot(no_rows, math.ceil(len(df.columns)/no_rows), j)
            if(df[col].dtype == np.float64 or df[col].dtype == np.int64):
                for i, val in enumerate(df[target_att].unique().tolist()):
                    plt.hist(df.loc[df[target_att] == val, col].values, **kwargs, label=str(val), color=colors[i])
                # plt.gca().set(title=col, ylabel='Probability')
                plt.xticks(fontsize=12)
                plt.legend()
            plt.title(col, fontsize=20)
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
        plt.show()
        return fig

    '''========================================================================
                                    Statistics
    ========================================================================'''

    def outlier_winsorize(self, data, list_exclude=[]):
        for col in data.columns:
            if col not in list_exclude:
                try:

                    # q75, q25 = np.percentile(data[col], [75, 25])
                    q75, q25 = np.percentile(data[~data[col].isnull()][col], [75, 25])
                    # q95, q05 = np.percentile(data[col], [95, 5])
                    iqr = q75 - q25
                    min_val = q25 - (iqr*1.5)
                    max_val = q75 + (iqr*1.5)

                    outlier_count_lower = len(np.where((data[col] < min_val))[0])
                    outlier_count_upper = len(np.where((data[col] > max_val))[0])
                    outlier_percent_lower = round(outlier_count_lower/len(data[col])*100, 2)
                    outlier_percent_upper = round(outlier_count_upper/len(data[col])*100, 2)
                    print(15*'-' + col + 15*'-')
                    print('Number of outliers lower: {}'.format(outlier_count_lower))
                    print('Number of outliers upper: {}'.format(outlier_count_upper))
                    print('Percent of data that is outlier lower: {}%'.format(outlier_percent_lower))
                    print('Percent of data that is outlier upper: {}%'.format(outlier_percent_upper))

                    wins_data = winsorize(data[col], limits=(outlier_percent_lower/100, outlier_percent_upper/100))
                    data[col] = wins_data
                except Exception:
                    pass
        return data

    def tTest(self, data, binSize, attr1, attr1_Value1, attr1_Value2, attrTarget, attrTarget_Value, vsALLrest=0, alpha=0.05):
        """ Creates a distribution plot and compares the difference between 2
        values of an attribute. Example: This function wil be used when the
        difference in readmission (attrTarget) for >30 days (attrTarget_Value) will
        need to be compared for males (attr1_Value1) and females (attr1_Value2)
        values in gender attribute (attr1). The gender column will be split in
        X samples (binSize). The percentage of readmission will be stored in lists
        for each sample. The mean and stdev will be calculated and ttest will be
        conducted.

        Parameters
        ----------
        data: (dataframe)
        binSize: (int) size of the bins foe the distribution
        attr1: (str) attribute on be tested
        attr1_Value1: (str or number) attribute value 1
        attr1_Value2: (str or number) attribute value 2
        attrTarget: (str)
        attrTarget_Value: (str or number) the value of the attribute target where
            the attr1 values will be tester
        vsALLrest: (str) if 1 then the attr1_Value1 is compared with all the rest
            values of the attribute
        alpha: (int) t test parameter

        Output: buf (to be printed in the pdf)
        """
        data = data.copy(deep=True)
        data = data.sample(frac=1)
        data = data[[attr1, attrTarget]]

        dfSampleSize = int(len(data)/binSize)

        sample1 = []
        sample2 = []

        for x in range(0, binSize):
            start_index = x * dfSampleSize
            end_index = (x+1) * dfSampleSize
            df_sample = data.iloc[start_index:end_index]

            lenOfValue1 = len(df_sample[df_sample[attr1] == attr1_Value1])
            lenOfValue2 = len(df_sample[df_sample[attr1] == attr1_Value2])

            criteria2 = df_sample[attrTarget] == attrTarget_Value

            s1 = round(len(df_sample[(df_sample[attr1] == attr1_Value1) & (criteria2)]) / lenOfValue1*100, 5)

            if vsALLrest == 1:
                s2 = round(len(df_sample[(df_sample[attr1] != attr1_Value1) & (criteria2)]) / (len(df_sample) - lenOfValue1)*100, 5)
                label2 = 'vsALLrest'
            else:
                s2 = round(len(df_sample[(df_sample[attr1] == attr1_Value2) & (criteria2)]) / lenOfValue2*100, 5)
                label2 = attr1_Value2

            sample1.append(s1)
            sample2.append(s2)

        sample1 = np.array(sample1)
        sample2 = np.array(sample2)

        meanS1 = sample1.mean()
        meanS2 = sample2.mean()

        s1_std = np.std(sample1)
        s2_std = np.std(sample2)

        print('-------------------------------------------------------')
        print("Null hypothesis:")
        print("Value1 (", attr1_Value1, ") and Value2 (", attr1_Value2, ") of (", (attr1), ") are not siginificantly different")
        print("Tested in attribute: ", attrTarget, " /Tested in Value: ", attrTarget_Value, "\n")

        print('Sample: 1/ mean: ', meanS1, '/stdev: ', s1_std)
        print('Sample: 2/ mean: ', meanS2, '/stdev: ', s2_std)
        print(' ')

        # stats.ttest_ind(sample1,sample2)
        ttest, pval = stats.ttest_ind(sample1, sample2)

        print(stats.ttest_ind(sample1, sample2))
        if pval < alpha:
            title1 = "we reject null hypothesis"
        else:
            title1 = "we accept null hypothesis"
        print('p: ', round(pval, 2), '/alpha: ', round(alpha, 2), '/ ', title1)

        crVl = t.ppf(1.0 - alpha, len(sample1) + len(sample2) - 2)

        if abs(ttest) <= crVl:
            title2 = "we accept null hypothesis"
        else:
            title2 = "we reject null hypothesis"
        print('t: ', round(ttest, 2), '/crVl: ', round(crVl, 2), '/ ', title2)
        print(title1)
        print('-------------------------------------------------------')

        self.create_directory()

        plt.close()
        plt.hist(sample1, alpha=0.5, label=attr1_Value1)
        plt.hist(sample2, alpha=0.5, label=label2)
        plt.legend(loc='upper right')
        plt.title('p: ' + title1 + '/ t: ' + title2)
        buf = self.image_in_PDF(plt, y=7)
        plt.show()
        # plt.savefig(self.path + '/' + attr1 + "_" + "_" + attrTarget + "_" + 'ttest.png', dpi=500, bbox_inches='tight')

        return buf

    def check_normal_distribution(self, data, columnName, attr1_Value1='None', binSize=0, attrTarget='None', attrTarget_Value='None', path="DA_plots"):

        if attr1_Value1 != 'None':
            data = data.copy(deep=True)
            data = data.sample(frac=1)
            data = data[[columnName, attrTarget]]

            dfSampleSize = int(len(data)/binSize)

            sample1 = []

            for x in range(0, binSize):
                start_index = x * dfSampleSize
                end_index = (x+1) * dfSampleSize
                df_sample = data.iloc[start_index:end_index]

                lenOfValue1 = len(df_sample[df_sample[columnName] == attr1_Value1])

                criteria2 = df_sample[attrTarget] == attrTarget_Value

                s1 = round(len(df_sample[(df_sample[columnName] == attr1_Value1) & (criteria2)]) / lenOfValue1*100, 5)

                sample1.append(s1)

            sample1 = np.array(sample1)

            x = sample1
        else:

            x = np.array(data[columnName])

        k2, p = stats.normaltest(x)
        alpha = 1e-3
        # print("p = {:g}".format(p))
        # p = 3.27207e-11

        title = "null hypothesis: x comes from a normal distribution"

        if p < alpha:  # null hypothesis: x comes from a normal distribution
            title = title + "\nThe null hypothesis can be rejected"
        else:
            title = title + "\nThe null hypothesis cannot be rejected"
        print('-------------------------------------------------------')
        print(title)
        print('-------------------------------------------------------')

        self.create_directory()

        plt.close()
        plt.hist(x, alpha=0.5, label=attr1_Value1)
        plt.legend(loc='upper right')
        plt.title(title)
        buf = self.image_in_PDF(plt, y=7)
        plt.show()
        # plt.savefig(self.path + '/' + columnName + "_distribution" + '.png', dpi=500, bbox_inches='tight')

        return buf


def main():
    # =============================================================================

    # DA = Data_Analysis()

    # =============================================================================
    # cnn = ID.setting_sql_connection('diabeticDB.db')
    # df_sur = ID.run_query('SELECT * FROM survey;', cnn)

    # query = '''SELECT survey.*, medicine.encounter_id as MedEnc, medicine.patient_nbr as MedPatnb, medicine.medicineName, medicine.dose
    # FROM   survey
    #        LEFT JOIN medicine
    #           ON survey.encounter_id = medicine.encounter_id
    #           and survey.patient_nbr = medicine.patient_nbr
    # UNION ALL
    # SELECT survey.*, medicine.*
    # FROM   medicine
    #        LEFT JOIN survey
    #           ON survey.encounter_id = medicine.encounter_id
    #           and survey.patient_nbr = medicine.patient_nbr
    # WHERE  survey.patient_nbr IS NULL;'''

    # df_OUTJOIN = ID.run_query(query, cnn)

    # df_med2 = df_OUTJOIN.set_index(['encounter_id'])
    # df_med2 = df_med2.pivot(columns='medicineName', values='dose')
    # df_med2 = df_med2.reset_index()
    # df_final = pd.merge(df_sur, df_med2, on="encounter_id")
    # print(df_final.head())

    # df = ID.from_csv("diabetic_data.csv")

    # Replacing ? with Nan
    # df = df.replace('?', np.nan)
    # DA.descriptive_analysis(df, 'readmitted')

    # DA.balance_down_sample(df, 'readmitted')
    # DA.generate_report()
    # df = ID.from_csv("diabetic_data.csv")
    # =============================================================================

    # image_buffer = DA.bar_plot_count_percentage(data, 'race', 'readmitted', 'DA_plots')
    # image_buffer = DA.bar_plot_count_percentage(data, 'age', 'readmitted', 'DA_plots')
    # image_buffer = DA.tTest(data, 25, 'A1Cresult', 'None', '>8', 'readmitted', '<30')
    # image_buffer = DA.hist_plot(data, 'DA_plots')
    # DA.generate_report()
    # image_buffer.close()
    pass
