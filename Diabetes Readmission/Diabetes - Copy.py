import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import math

sys.path.append("..")
from EDA_ML_Package.EDA_functions import Data_Analysis
from EDA_ML_Package.ML_functions import ML_models
from EDA_ML_Package.NN_functions import ANN_tabular_class

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 15)
DA = Data_Analysis()


def bar_plot_count_percentage(data, att, attTarget, data2=0, attTarget2=[], plotType='Both', reverse=False):
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
    def annotate_plot(gb, attTarget_X, plotType):
        # Annotating the bar plots with the Per and count
        for jj, p in enumerate(sp.patches):
            # print(gb)
            height = p.get_height()
            height = 0 if math.isnan(height) is True else height
            height = int(height) if height > 100 else round(height, 1)
            gb = gb.sort_values(by=[attTarget_X])
            gb = gb.reset_index(drop=True)
            if plotType == "Per":
                sp.text(p.get_x() + p.get_width()/2., height + 0, str(height) + "\n(" + str(gb['count'][jj]) + ")", ha="center", fontsize=15)
            else:
                sp.text(p.get_x() + p.get_width()/2., height + 0, str(height) + "\n(" + str(gb['Pop Per'][jj]) + ")", ha="center", fontsize=15)

    def plot_details(ii, subSize, attTarget_X, plotType, axis_legend_flag=0):
        # Setting the plot parameters
        # sp.set_yticklabels(sp.get_yticks(), size=15)
        sp.tick_params(labelsize=16)
        if (subSize == ii+1) or ((plotType == "Both") and (axis_legend_flag == 1)):
            sp.set_xlabel(att, fontsize=16)
        else:
            sp.set_xticks([])
            sp.set_xlabel('')
        if plotType == "Per":
            sp.set_ylim([0, 100])
        sp.set_title(att + "/ " + attTarget_X, fontsize=20)
        sp.legend(prop={'size': 16})

    # =========================================================================
    # Making sure attTarget is list so can iterate through it
    if isinstance(attTarget, list) is False:
        attTarget = [attTarget]

    # Setting the size of the plot/ affected by the number of subplots
    subSize = 2*len(attTarget) if plotType == 'Both' else 1*len(attTarget)
    plot_x_size = 3*subSize if subSize > 2 else 14
    plot_y_size = 4*subSize if subSize > 2 else 8
    fig, axes = plt.subplots(subSize, 1, figsize=(plot_x_size, plot_y_size))
    if isinstance(axes, np.ndarray) is False:
        axes = [axes]

    att_original = att

    df = data.copy(deep=True)

    for ii, attTarget_X in enumerate(attTarget):
        if reverse == True:  # Reversing the label with x axis
            att = attTarget_X
            attTarget_X = att_original

        # isolating plot data
        gb = DA.groupby_count_percentage(df, [att, attTarget_X])
        x = gb[att].str.split(",", n=1, expand=True)
        y = x[0].map(lambda x: x.lstrip('(').rstrip('aAbBcC'))
        y = y.sort_values(ascending=True)
        gb.iloc[y.index.tolist()]

        sns.set_style("whitegrid")

        if plotType == 'Both':
            sp = sns.barplot(x=att, y="count", hue=attTarget_X, data=gb, ax=axes[0 + ii*2])
        if plotType == 'Count':
            sp = sns.barplot(x=att, y="count", hue=attTarget_X, data=gb, ax=axes[0 + ii])

        if (plotType == 'Count') or (plotType == 'Both'):
            annotate_plot(gb, attTarget_X, plotType)
            plot_details(ii, subSize, attTarget_X, plotType, axis_legend_flag=0)
            sp.set_ylabel('Count', fontsize=18)

        if plotType == 'Both':
            sp = sns.barplot(x=att, y="Pop Per", hue=attTarget_X, data=gb, ax=axes[1 + ii*2])
        elif plotType == 'Per':
            sp = sns.barplot(x=att, y="Pop Per", hue=attTarget_X, data=gb, ax=axes[0 + ii])

        if (plotType == 'Per') or (plotType == 'Both'):
            annotate_plot(gb, attTarget_X, plotType)
            plot_details(ii, subSize, attTarget_X, plotType, axis_legend_flag=1)
            sp.set_ylabel('Percentage', fontsize=18)

    fig.tight_layout()
    figure = sp.get_figure()
    plt.show()
    plt.close(figure)


def EDA_Analysis():
    '''# =========================== Instances ======================================='''

    DA = Data_Analysis()

    '''# =========================== Importing Data =================================='''

    df = pd.read_csv("diabetic_data.csv")
    df = df.replace('?', np.nan)  # Replacing ? with Nan
    med_keep = ['metformin', 'glipizide', 'glyburide', 'pioglitazone', 'rosiglitazone']

    df['readmitted'] = df.readmitted.replace({'<30': 1, '>30': 0, 'NO': 0})
    bar_plot_count_percentage(df, 'readmitted', med_keep, plotType='Count', reverse=True)
    bar_plot_count_percentage(df, 'readmitted', med_keep, plotType='Per', reverse=True)
    bar_plot_count_percentage(df, 'readmitted', med_keep, plotType='Per', reverse=False)

    sys.exit()

    bar_plot_count_percentage(df, 'change', ['readmitted', 'A1Cresult', 'diabetesMed'], plotType='Per')
    sys.exit()

#    bar_plot_count_percentage(df, 'change', 'readmitted', plotType='Per')
#    bar_plot_count_percentage(df, 'change', 'readmitted', plotType='Both')
#    bar_plot_count_percentage(df, 'change', 'readmitted', plotType='Count')
    #bar_plot_count_percentage(df, 'change', ['readmitted', 'A1Cresult'], plotType='Per')


    medicines1 = ['metformin', 'glimepiride', 'glipizide', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glyburide',
                  'pioglitazone', 'rosiglitazone', 'acarbose', 'glyburide-metformin']
    
    medicines2 = ['glipizide-metformin', 'tolazamide', 'tolbutamide', 'glimepiride-pioglitazone', 'citoglipton', 'examide',
                  'troglitazone', 'acetohexamide', 'miglitol', 'metformin-rosiglitazone', 'metformin-pioglitazone']

    medicines = medicines1 + medicines2
    
    df_med = pd.DataFrame(columns=['medicine', 'Down', 'No', 'Steady', 'Up'])
    
    for i, medic in enumerate(medicines):
        df_med_distr = DA.groupby_count_percentage(df, [medic])
    
        # print(df_med_distr[df_med_distr[medic] == 'Down']['count'][0])
        #print(df_med['Down'].iloc[0])
        medicine_action_list = []
        medicine_action_list.append(medic)
        for xx in ['Down', 'No', 'Steady', 'Up']:
            try:
    
                medicine_action_list.append(df_med_distr[df_med_distr[medic] == xx]['count'].iloc[-1])
            except:
                medicine_action_list.append("-")
    
        df_med.loc[i] = medicine_action_list
    
    print(df_med)
    df_med = df_med.sort_values(by=['No'])
    med_keep = df_med.medicine.to_list()[0:5]

    bar_plot_count_percentage(df, 'readmitted', med_keep, plotType='Per', reverse=True)

    sys.exit()
    '''# ------------------- EDA -----------------------------------------------------'''

    DA.descriptive_analysis(df, 'readmitted')
    DA.bar_plot_count_percentage(df, 'change', 'A1Cresult', 'change vs A1Cresult', plotType='Per')
    DA.bar_plot_count_percentage(df, 'change', 'readmitted', 'change vs readmitted', plotType='Both')
    DA.bar_plot_count_percentage(df, 'A1Cresult', 'readmitted', 'A1Cresult vs readmitted', plotType='Both')
    DA.bar_plot_count_percentage(df, 'race', 'readmitted', 'Race vs Readmitted', plotType='Both')
    DA.bar_plot_count_percentage(df, 'age', 'readmitted', 'Age vs Readmitted', 'Both')
    # DA.kde_plot(df, 'gender', 'time_in_hospital', 'Male', 'Female')
    # DA.box_plot(df, 'race', 'time_in_hospital')

    df = pd.read_csv("Diabetes Data/diabetic_data.csv")
    df = df.replace('?', np.nan)  # Replacing ? with Nan

    replace_dict = {'Infectious': (1, 139),
                    'Neoplasmic': (140, 239),
                    'Hormonal': (240, 279),
                    'Blood': (280, 289),
                    'Mental': (290, 319),
                    'Nervous': (320, 359),
                    'Sensory': (360, 389),
                    'Circulatory': (390, 459),
                    'Respiratory': (460, 519),
                    'Digestive': (520, 579),
                    'Genitourinary': (580, 629),
                    'Childbirth': (630, 679),
                    'Dermatological': (680, 709),
                    'Musculoskeletal': (710, 739),
                    'Congenital': (740, 759),
                    'Perinatal': (760, 779),
                    'Miscellaneous': (780, 799),
                    'Injury': (800, 999)}
    DA.nan_heatmap(df)

    def replace_columnValues_dict(df, columnName, replace_dict):
        def is_number(s):
            try:
                float(s)
                return True
            except ValueError:
                return False
        columnsValues = list(df[columnName].values)
        for i, columnsValue in enumerate(columnsValues):
            for key, tuple_range in replace_dict.items():
                if is_number(columnsValues[i]):
                    if ('str' not in str(type(float(columnsValues[i])))):
                        if (int(float(columnsValues[i])) >= tuple_range[0]) and (int(float(columnsValues[i])) <= tuple_range[1]):
                            columnsValues[i] = key

        return columnsValues

    '''# ========================== Data Cleaning ===================================='''

    # Droping rows that have nan for the following columns
    df = df[~df['race'].isnull()]
    df = df[~df['diag_1'].isnull()]
    df = df[~df['diag_2'].isnull()]
    df = df[~df['diag_3'].isnull()]

    # Droping duplicates from the following column
    df = df.drop_duplicates(subset=['patient_nbr'])

    # Droping rows that have the following values for the X columns
    df = df.loc[~df['discharge_disposition_id'].isin([11, 13, 14, 19, 20, 21, 25, 26])]

    for diag in ['diag_1', 'diag_2', 'diag_3']:
        df.loc[df[diag].str.startswith('250'), diag] = "Diabetes"
        df.loc[df[diag].str.startswith('V' or 'E'), diag] = "Accidental"
        df[diag] = replace_columnValues_dict(df, diag, replace_dict)

    for diag in ['diag_1', 'diag_2', 'diag_3', 'medical_specialty']:
        top_x = df.groupby(diag).size().sort_values(ascending=False).keys().to_list()[:5]
        df.loc[~df[diag].isin(top_x), diag] = 'Other'

    DA.bar_plot_count_percentage(df, 'diag_1', 'A1Cresult', 'diag_1 vs A1Cresult', plotType='Per')
    DA.bar_plot_count_percentage(df, 'diag_1', 'change', 'diag_1 vs change', plotType='Per')
    DA.bar_plot_count_percentage(df, 'diag_1', 'readmitted', 'diag_1 vs readmitted', plotType='Per')

    df = df.drop(['encounter_id', 'patient_nbr', 'payer_code', 'weight'], axis=1)

    df['admission_type_id'] = pd.Series(['Emergency' if val in [1, 2] else 'Not Emergency' for val in df['admission_type_id']], index=df.index)
    df['admission_source_id'] = pd.Series([2 if val in [7] else 1 if val in [1] else 0 for val in df['admission_source_id']], index=df.index)
    df['discharge_disposition_id'] = pd.Series([1 if val in [1] else 0 for val in df['discharge_disposition_id']], index=df.index)

    medicines1 = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'glipizide', 'glyburide',
                  'pioglitazone', 'rosiglitazone', 'acarbose', 'glyburide-metformin']
    df = df.drop(medicines1, axis=1)
    medicines2 = ['glipizide-metformin', 'tolazamide', 'tolbutamide', 'glimepiride-pioglitazone', 'citoglipton', 'examide',
                  'troglitazone', 'acetohexamide', 'miglitol', 'metformin-rosiglitazone', 'metformin-pioglitazone']

    df = df.drop(medicines2, axis=1)

    # colsNumerical = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications',
    #                 'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses']

    def remove_rare_occurences(df, frequeThreshold=30):
        for x in list(df.columns):
            for y in list(df[x].unique()):
                if (len(df[df[x] == y][x]) <= frequeThreshold):
                    print('Attr: ', x, ', /value: ', y, '/size: ', len(df[df[x] == y][x]))
                    df = df[df[x] != y]
        return df

    # for colName in colsNumerical:
    #    df[colName] = DA.place_column_in_bins(df, colName)
    # pd.cut(df.time_in_hospital, 3)

    df = remove_rare_occurences(df)

    df['readmitted'] = df.readmitted.replace({'<30': 1, '>30': 2, 'NO': 0})
    DA.descriptive_analysis(df, 'readmitted')

    X = df.drop(['readmitted'], 1)  # [['Pclass', 'Sex', 'Age']]
    y = df['readmitted']

    # categorize catergoral values

    X_tr, pipeline_tr = DA.data_transformation(X, cat_type=1)
    X_tr.to_csv("X_tr.csv", index=False)
    y.to_csv("y_diabetic.csv", index=False)
    DA.generate_report('diabetes')
    return X, y


def ML_Analusis():
    ML = ML_models()

    X_tr = pd.read_csv("X_tr.csv")
    y = pd.read_csv("y_diabetic.csv")
    df = X_tr.join(y)
    df1 = df[df.readmitted == 1]
    df0 = df[df.readmitted == 0]
    df0 = df0.sample(n=len(df1), random_state=42)
    df = pd.concat([df0, df1])

    X_tr = df.drop(['readmitted'], 1)  # [['Pclass', 'Sex', 'Age']]
    y = df['readmitted']

    # sys.exit()

    for clm in X_tr.columns:
        clm_new = clm.replace("[", "_")
        clm_new = clm_new.replace("]", "_")
        clm_new = clm_new.replace("(", "_")
        clm_new = clm_new.replace(")", "_")
        X_tr = X_tr.rename(columns={clm: clm_new})
    DA.descriptive_analysis(X_tr)

    results = ML.ML_Basic_Models(X_tr, y)
    ML.generate_report('diabetes')

    return results


if __name__ == '__main__':
    X, y = EDA_Analysis()
    results = ML_Analusis()
