from EDA_ML_Functions.EDA_functions import Data_Analysis
from EDA_ML_Functions.ML_functions import ML_models
from EDA_ML_Functions.NN_functions import ANN_tabular_class

import pandas as pd
import numpy as np
import sys

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 15)


def EDA_Analysis():
    '''# =========================== Instances ======================================='''

    DA = Data_Analysis()

    '''# =========================== Importing Data =================================='''

    df = pd.read_csv("Diabetes Data/diabetic_data.csv")
    df = df.replace('?', np.nan)  # Replacing ? with Nan

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
    DA = Data_Analysis()
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
