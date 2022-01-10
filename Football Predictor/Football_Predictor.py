import pandas as pd
import numpy as np
from datetime import datetime
from datetime import datetime as dt
from datetime import date
from sklearn.model_selection import train_test_split
import sys
from scipy.stats import poisson, skellam
import statsmodels.api as sm
import statsmodels.formula.api as smf
from selenium import webdriver
from bs4 import BeautifulSoup
from sklearn.ensemble import RandomForestClassifier
import time

sys.path.append("..")

from EDA_ML_Package.ML_functions import ML_models
from Data_Construction_Functions import *
from Parsing_Functions import *

pd.set_option('display.max_columns', 25)
pd.set_option('display.max_rows', 60)


def prepare_ML_datasets(df, periodX, fixtX):
    ML_historic = previous_data_from_selected_fixture(df, periodX, fixtX)

    if (int(periodX) < int(df.period.iloc[-1])) or (int(fixtX) <= int(df.fixt.iloc[-1])):
        ML_selected_fixture, selected_fixt_results = selected_fixture_data_results(df, periodX, fixtX)
    else:
        Home_Team_Cols = []
        Away_Team_Cols = []
        ML_selected_fixture = parse_next_fixture()

        selected_fixt_results = ML_selected_fixture.copy()

        for col in ML_historic.columns:
            if ("H" in col) and ("365" not in col) and ("HomeTeam" not in col) and ("FT" not in col):
                Home_Team_Cols.append(col)
            elif ("A" in col) and ("365" not in col) and ("AwayTeam" not in col) and ("FT" not in col):
                Away_Team_Cols.append(col)

        ML_selected_fixture[Home_Team_Cols] = 0
        ML_selected_fixture[Away_Team_Cols] = 0
        for Hteam in ML_selected_fixture.HomeTeam:
            ML_selected_fixture.loc[ML_selected_fixture.HomeTeam == Hteam, Home_Team_Cols] = ML_historic[ML_historic.HomeTeam == Hteam][Home_Team_Cols].iloc[-1].values
        for Ateam in ML_selected_fixture.AwayTeam:
            ML_selected_fixture.loc[ML_selected_fixture.AwayTeam == Ateam, Away_Team_Cols] = ML_historic[ML_historic.AwayTeam == Ateam][Away_Team_Cols].iloc[-1].values

    return ML_historic, ML_selected_fixture, selected_fixt_results


def ML_Prediction(periodX, fixtX, League, B365_data=1, ML_train=0, print_selection=1):
    df = pd.read_csv('FootballfullData_' + League + '.csv')
    print("Fixture: ", fixtX)

    ML_historic, X_to_predict, selected_fixt_results = prepare_ML_datasets(df, periodX, fixtX)

    # -----------------  Poisson ----------------------------------------------
    H_odd_Poisson, D_odd_Poisson, A_odd_Poisson, O_odd_Poisson, U_odd_Poisson = odds_by_poisson(ML_historic, X_to_predict)
    selected_fixt_results['poisH'] = np.round(H_odd_Poisson, 3)
    selected_fixt_results['poisD'] = np.round(D_odd_Poisson, 3)
    selected_fixt_results['poisA'] = np.round(A_odd_Poisson, 3)

    # -----------------  Deleting Columns not needed --------------------------
    y = ML_historic['Result']
    X = ML_historic.drop(columns=['Result'])
    X = remove_columns_try(X, ['HomeTeam', 'AwayTeam'])
    X = remove_columns_try(X, ['period', 'fixt', 'Div'])
    X = remove_columns_try(X, ['FTH_Goals', 'FTA_Goals'])
    X = remove_columns_try(X, ["Htm_games_at_H", "Atm_games_at_A", "Htm_Total_games", "Atm_Total_games"])

    X_to_predict = remove_columns_try(X_to_predict, ['HomeTeam', 'AwayTeam'])
    X_to_predict = remove_columns_try(X_to_predict, ['period', 'fixt', 'Div'])
    X_to_predict = remove_columns_try(X_to_predict, ["Htm_games_at_H", "Atm_games_at_A", "Htm_Total_games", "Atm_Total_games"])

    if B365_data == 0:
        X = remove_columns_try(X, ['B365H', 'B365D', 'B365A'])  # 'B365<2.5', 'B365>2.5'
        X_to_predict = remove_columns_try(X_to_predict, ['B365H', 'B365D', 'B365A'])

    # --------- Train Test Datasets --------------------------
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(random_state=0)
    clf.fit(X, y)

    # -----------  Finding the best ML Model --------------------------
    if ML_train == 1:
        ML = ML_models()
        ML.ML_Basic_Models(X, y, test_case=0)
        print("-"*100)
        y_DrawAway = ML_historic['Result'].map({0: 0, 1: 1, 2: 1})
        ML.ML_Basic_Models(X, y_DrawAway, test_case=0)
        print("-"*100)

    # -------------  Constructing the outcome table -----------------
    y_scores = clf.predict_proba(X_to_predict)*100
    predictions = round(pd.DataFrame(100/y_scores, columns=["prH", "prD", "prA"]), 2)

    selected_fixt_results = selected_fixt_results.reset_index(drop=True)
    selected_fixt_results = selected_fixt_results.join(predictions, how='outer')

    selected_fixt_results = check_method_success(selected_fixt_results, "B365H", "B365D", "B365A", "B365")
    selected_fixt_results = check_method_success(selected_fixt_results, "prH", "prD", "prA", "ML")
    selected_fixt_results = check_method_success(selected_fixt_results, "poisH", "poisD", "poisA", "poisson")
    selected_fixt_results = check_ML_lower_B365_success(selected_fixt_results)
    selected_fixt_results = check_method_DA_success(selected_fixt_results, "prH", "prD", "prA", "ML")
    selected_fixt_results = check_method_DA_success(selected_fixt_results, "B365H", "B365D", "B365A", "B365")
    selected_fixt_results = check_method_DA_success(selected_fixt_results, "poisH", "poisD", "poisA", "poisson")

    selected_fixt_results = check_method_DA_focused_success(selected_fixt_results, ["prH", "prD", "prA"], ["B365H", "B365D", "B365A"])

    if print_selection == 1:
        print(selected_fixt_results)

    return selected_fixt_results


def back_testing(first_fixt, last_fixt, periodX, B365_data, League):

    results_total = pd.DataFrame()
    for fixtX in range(first_fixt, last_fixt):
        results = ML_Prediction(periodX, fixtX, League, B365_data=B365_data, print_selection=0)
        results_total = pd.concat([results_total, results])

    print("-"*50)
    print("ML success: ", round(len(results_total[results_total.ML_Sucess == 1])/len(results_total)*100, 2))
    print("ML H_DA success: ", round(len(results_total[results_total.ML_Home_DrawAway == 1])/len(results_total)*100, 2))
    len0_ML_H_DA_0 = len(results_total[results_total.ML_Home_DrawAway_Foc == 0])
    len0_ML_H_DA_1 = len(results_total[results_total.ML_Home_DrawAway_Foc == 1])
    len0_ML_H_DA_m1 = len(results_total[results_total.ML_Home_DrawAway_Foc == -1])
    print("ML H_DA success_foc: ", round(len0_ML_H_DA_1/(len0_ML_H_DA_1 + len0_ML_H_DA_0)*100, 2))
    print("ML B365: ", round(len(results_total[results_total.ML_B365 == 1])/(len(results_total[results_total.ML_B365 == 1]) + len(results_total[results_total.ML_B365 == 0]))*100, 2))
    print("-"*50)
    print("Poisson success: ", round(len(results_total[results_total.poisson_Sucess == 1])/len(results_total)*100, 2))
    print("Poisson H_DA success: ", round(len(results_total[results_total.poisson_Home_DrawAway == 1])/len(results_total)*100, 2))
    print("-"*50)
    print("B365 success: ", round(len(results_total[results_total.B365_Sucess == 1])/len(results_total)*100, 2))
    print("B365 H_DA success: ", round(len(results_total[results_total.B365_Home_DrawAway == 1])/len(results_total)*100, 2))
    print("\n\n")

    unsucc_H_AW = results_total[results_total.ML_Home_DrawAway_Foc == 0]
    unsucc_H_AW.to_csv("unsucc_H_AW_bet" + str(B365_data) + ".csv", index=False)



if __name__ == "__main__":
    League = "E0"
    file_name = "FootballfullData_" + League

    df = pd.read_csv(file_name + '.csv')

    # df = retrieve_data_and_construct_data(League)

    periodX = '2021'
    fixtX = 25

    # ML_historic, ML_selected_fixture, selected_fixt_results = prepare_ML_datasets(df, "2122", fixtX)
    # results = ML_Prediction("2122", fixtX, League, B365_data=0)

    game = "Tottenham/ Arsenal"
    plot_type = "Goals"

    # used in the dashboard
    # homeTeamGoalPlot, awayTeamGoalPlot = plot_factors_dataframes(periodX, fixtX, game, plot_type, League)
    fixtureTable = table_ranking(periodX, fixtX, League)
    factorsTable = factors_current_fixt(periodX, fixtX, League)

    unfold_paired_data(df, League)

    first_fixt = 10
    last_fixt = 11

    back_testing(first_fixt, last_fixt, periodX, 0, League)
    # back_testing(first_fixt, last_fixt, periodX, 1, League)
