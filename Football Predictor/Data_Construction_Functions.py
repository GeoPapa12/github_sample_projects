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

def retrieve_data_and_construct_data(league):

    appended_data = []
    print("League: ", league)
    for yearPeriodXX in range(5, 22):
        if len(str(yearPeriodXX)) == 1:
            start_period = "0" + str(yearPeriodXX)
            if len(str(yearPeriodXX + 1)) == 1:
                end_period = "0" + str(yearPeriodXX + 1)
            else:
                end_period = str(yearPeriodXX + 1)
        else:
            start_period = str(yearPeriodXX)
            end_period = str(yearPeriodXX + 1)

        period = start_period + end_period
        fullData = pd.read_csv("http://www.football-data.co.uk/mmz4281/" + period + "/" + league + ".csv", parse_dates=[1], dayfirst=True)

        period = int(period)

        print("Period: ", period)

        fullData = filter_and_rename_columns(fullData)
        fullData = table_construction(fullData, period)
        fullData = remove_columns(fullData)
        fullData['Div'] = league
        appended_data.append(fullData)

    appended_data = pd.concat(appended_data, sort=False)
    appended_data = appended_data[appended_data['Div'].notna()]  # remove row if div is nan

    appended_data = appended_data.dropna()
    appended_data.to_csv("FootballfullData_" + league + ".csv", index=False)
    return appended_data


def table_construction(df, period):

    def factors_based_on(attribute_type, df, tm):
        ''' Sub function that constracts the Home/ Away/ In favour/ Against factors'''
        dx = df[((df.HomeTeam == tm) | (df.AwayTeam == tm))][['fixt', 'HomeTeam', 'AwayTeam', 'FTH_' + attribute_type, 'FTA_' + attribute_type]]
        dx.loc[(dx.HomeTeam == tm), 'Htm_' + attribute_type + '_favour'] = dx[dx.HomeTeam == tm]['FTH_' + attribute_type].shift(1).rolling(min_periods=1, window=3).sum()
        dx.loc[(dx.AwayTeam == tm), 'Atm_' + attribute_type + '_favour'] = dx[dx.AwayTeam == tm]['FTA_' + attribute_type].shift(1).rolling(min_periods=1, window=3).sum()
        dx.loc[(dx.HomeTeam == tm), 'Htm_' + attribute_type + '_against'] = dx[dx.HomeTeam == tm]['FTA_' + attribute_type].shift(1).rolling(min_periods=1, window=3).sum()
        dx.loc[(dx.AwayTeam == tm), 'Atm_' + attribute_type + '_against'] = dx[dx.AwayTeam == tm]['FTH_' + attribute_type].shift(1).rolling(min_periods=1, window=3).sum()
        df.loc[(df.HomeTeam == tm), 'Htm_' + attribute_type + '_favour'] = round(dx[dx.HomeTeam == tm]['Htm_' + attribute_type + '_favour']/3, 2)
        df.loc[(df.AwayTeam == tm), 'Atm_' + attribute_type + '_favour'] = round(dx[dx.AwayTeam == tm]['Atm_' + attribute_type + '_favour']/3, 2)
        df.loc[(df.HomeTeam == tm), 'Htm_' + attribute_type + '_against'] = round(dx[dx.HomeTeam == tm]['Htm_' + attribute_type + '_against']/3, 2)
        df.loc[(df.AwayTeam == tm), 'Atm_' + attribute_type + '_against'] = round(dx[dx.AwayTeam == tm]['Atm_' + attribute_type + '_against']/3, 2)
        return df

    teamNames = df['HomeTeam'].unique().tolist()
    teamNames = [x for x in teamNames if str(x) != 'nan']
    teamNames = sorted(teamNames)

    df.loc[(df.Result == "H"), 'HomePoints'] = 3
    df.loc[(df.Result == "A"), 'HomePoints'] = 0
    df.loc[(df.Result == "H"), 'AwayPoints'] = 0
    df.loc[(df.Result == "A"), 'AwayPoints'] = 3
    df.loc[(df.Result == "D"), 'HomePoints'] = 1
    df.loc[(df.Result == "D"), 'AwayPoints'] = 1
    df['Result'] = df['Result'].map({'H': 0, 'D': 1, 'A': 2})

    # no_of_fixt = [[x]*10 for x in range(1, (len(teamNames)*2-2) + 1)]
    # no_of_fixt = [item for sublist in no_of_fixt for item in sublist]
    # no_of_fixt = no_of_fixt[:len(df)]

    Top1_tier = ['Man City', 'Liverpool', 'Chelsea', 'Man United']
    Top2_tier = ['Arsenal', 'Tottenham']
    Middle_tier = ['Everton', 'Newcastle', 'Leicester']

    df['Htm_fifa'] = 60
    df['Atm_fifa'] = 50

    for tm in teamNames:

        fixt_len = len(df.loc[((df.HomeTeam == tm) | (df.AwayTeam == tm))])
        df.loc[((df.HomeTeam == tm) | (df.AwayTeam == tm)), 'fixt'] = range(1, fixt_len + 1)  # Needs Improvement
        df.loc[((df.HomeTeam == tm) | (df.AwayTeam == tm)), 'period'] = period

        df.loc[(df.HomeTeam == tm), 'Htm_points_at_H'] = df[df.HomeTeam == tm].groupby(['HomeTeam', 'fixt'])['HomePoints'].sum().groupby(level=0).cumsum().reset_index()['HomePoints'].values
        df.loc[(df.AwayTeam == tm), 'Atm_points_at_A'] = df[df.AwayTeam == tm].groupby(['AwayTeam', 'fixt'])['AwayPoints'].sum().groupby(level=0).cumsum().reset_index()['AwayPoints'].values

        dff = df[((df.HomeTeam == tm) | (df.AwayTeam == tm))][['fixt', 'HomeTeam', 'AwayTeam', 'HomePoints', 'AwayPoints', 'Htm_points_at_H', 'Atm_points_at_A']]
        dff.loc[(dff.HomeTeam == tm), 'TeamsPointsX'] = dff[dff.HomeTeam == tm]['HomePoints']
        dff.loc[(dff.AwayTeam == tm), 'TeamsPointsX'] = dff[dff.AwayTeam == tm]['AwayPoints']
        dff['TeamsPoints'] = dff.TeamsPointsX.cumsum()
        df.loc[(df.HomeTeam == tm), 'Htm_Total_points'] = dff[dff.HomeTeam == tm]['TeamsPoints']
        df.loc[(df.AwayTeam == tm), 'Atm_Total_points'] = dff[dff.AwayTeam == tm]['TeamsPoints']

        df = factors_based_on('Goals', df, tm)
        df = factors_based_on('ShTar', df, tm)
        df = factors_based_on('Sht', df, tm)

        df.loc[(df.HomeTeam == tm), 'Htm_games_at_H'] = [*range(1, len(df[df.HomeTeam == tm])+1)]
        df.loc[(df.AwayTeam == tm), 'Atm_games_at_A'] = [*range(1, len(df[df.AwayTeam == tm])+1)]
 
        if tm in Top1_tier:
            df.loc[df.HomeTeam == tm, 'Htm_fifa'] = 90
            df.loc[df.AwayTeam == tm, 'Atm_fifa'] = 80
        if tm in Top2_tier:
            df.loc[df.HomeTeam == tm, 'Htm_fifa'] = 80
            df.loc[df.AwayTeam == tm, 'Atm_fifa'] = 70
        if tm in Middle_tier:
            df.loc[df.HomeTeam == tm, 'Htm_fifa'] = 70
            df.loc[df.AwayTeam == tm, 'Atm_fifa'] = 60

    return df


def filter_and_rename_columns(df):
    '''Function to remove/ rename columns'''
    df = df[['Date', 'Div', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HS', 'AS',
             'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'B365H', 'B365D', 'B365A']]

    df = df.rename(columns={'FTHG': 'FTH_Goals',
                            'FTAG': 'FTA_Goals',
                            'FTR': 'Result',
                            'HS': 'FTH_Sht',
                            'AS': 'FTA_Sht',
                            'HST': 'FTH_ShTar',
                            'AST': 'FTA_ShTar',
                            'HF': 'HFls',
                            'AF': 'AFls',
                            'HC': 'HCorn',
                            'AC': 'ACorn',
                            'HY': 'HYel',
                            'AY': 'AYel'})
    return df


def remove_columns(df):
    '''Function to remove columns which used for the calcualation of factors'''
    df = df.drop(columns=['Date', 'Div'])  # , 'FTH_Goals', 'FTA_Goals'])
    df = df.drop(columns=['FTH_Sht', 'FTA_Sht', 'FTH_ShTar', 'FTA_ShTar', 'HFls', 'AFls', 'HCorn'])
    df = df.drop(columns=['ACorn', 'HYel', 'AYel', 'HomePoints', 'AwayPoints'])
    return df


def remove_columns_try(df, col_list):
    '''Function to remove columns which used for the calcualation of factors'''
    for col in col_list:
        try:
            df = df.drop(columns=[col])
        except Exception:
            pass
    return df


def previous_data_from_selected_fixture(df, periodX, fixtX):
    '''Retrieve all the data before the selected period and fixture'''
    df.period = df.period.astype(int)
    periodX = int(periodX)

    df1 = df[(df.period < periodX)]
    df2 = df[((df.period == periodX) & (df.fixt < fixtX))]
    ML_historic = pd.concat([df1, df2])
    ML_historic = ML_historic.dropna()

    return ML_historic


def selected_fixture_data_results(df, periodX, fixtX):
    '''Retrieve the data of the selected fixture'''
    df.period = df.period.astype(int)
    periodX = int(periodX)
    selected_fixt = df[((df.period == periodX) & (df.fixt == fixtX))]
    selected_fixt_results = selected_fixt[['HomeTeam', 'AwayTeam', 'FTH_Goals', 'FTA_Goals', 'Result', 'B365H', 'B365D', 'B365A']]

    ML_selected_fixture = selected_fixt.drop(columns=['Result', 'FTH_Goals', 'FTA_Goals'])

    return ML_selected_fixture, selected_fixt_results


def last_fixture_teams(periodX, fixtX, League):
    df = pd.read_csv('FootballfullData_' + League + '.csv')
    selected_fixt_names = selected_fixture_data_results(df, periodX, fixtX)[0]
    selected_fixt_names = selected_fixt_names['HomeTeam'] + "/ " + selected_fixt_names['AwayTeam']
    selected_fixt_names = pd.DataFrame(selected_fixt_names, columns=['label'])
    selected_fixt_names['value'] = selected_fixt_names['label']
    return selected_fixt_names


def factors_current_fixt(periodX, fixtX, League):
    df = pd.read_csv('FootballfullData_' + League + '.csv')
    ML_selected_fixture, _ = selected_fixture_data_results(df, periodX, fixtX)
    return round(ML_selected_fixture.drop(columns=['period', 'fixt', 'B365H', 'B365D', 'B365A']), 2)


def table_ranking(periodX, fixtX, League):
    '''returns the ranking table of the selected fixture'''
    df = pd.read_csv('FootballfullData_' + League + '.csv')
    ML_selected_fixture, _ = selected_fixture_data_results(df, periodX, fixtX)
    ML_selected_fixture_prev, _ = selected_fixture_data_results(df, periodX, fixtX-1)

    ML_selected_fixture = pd.concat([ML_selected_fixture, ML_selected_fixture_prev])

    H_rank = ML_selected_fixture[['HomeTeam', 'Htm_points_at_H', 'Htm_Total_points']]
    A_rank = ML_selected_fixture[['AwayTeam', 'Atm_points_at_A', 'Atm_Total_points']]
    H_rank = H_rank.rename(columns={"HomeTeam": "Team", "Htm_Total_points": "Total", "Htm_points_at_H": "Points@H"})
    A_rank = A_rank.rename(columns={"AwayTeam": "Team", "Atm_Total_points": "Total", "Atm_points_at_A": "Points@A"})

    total_rank = pd.concat([H_rank, A_rank])
    total_rank = total_rank.fillna('-')
    total_rank = total_rank.sort_values(by='Total', ascending=False)
    total_rank['Pos'] = list(range(1, len(total_rank) + 1))

    total_rank = total_rank[['Pos', 'Team', 'Total', 'Points@H', 'Points@A']]

    total_rank.drop_duplicates(subset="Team", inplace=True)
    return total_rank


def plot_factors_dataframes(periodX, fixtX, game, plot_type, League):
    df = pd.read_csv('FootballfullData_' + League + '.csv')
    ML_historic = previous_data_from_selected_fixture(df, periodX, fixtX)
    periodX = int(periodX)
    team1 = game.split("/ ")[0]
    team2 = game.split("/ ")[1]

    if plot_type == "Goals":
        Hlist = ['fixt', 'Htm_Goals_favour', 'Htm_Goals_against']
        Alist = ['fixt', 'Atm_Goals_favour', 'Atm_Goals_against']
    elif plot_type == 'Shoots':
        Hlist = ['fixt', 'Htm_Sht_favour', 'Htm_Sht_against']
        Alist = ['fixt', 'Atm_Sht_favour', 'Atm_Sht_against']
    elif plot_type == 'Shoots on Target':
        Hlist = ['fixt', 'Htm_ShTar_favour', 'Htm_ShTar_against']
        Alist = ['fixt', 'Atm_ShTar_favour', 'Atm_ShTar_against']

    dftm1 = ML_historic[(ML_historic.HomeTeam == team1) & (ML_historic.period == periodX)][Hlist]
    dftm2 = ML_historic[(ML_historic.AwayTeam == team2) & (ML_historic.period == periodX)][Alist]

    return dftm1, dftm2


def unfold_paired_data(df, League):
    H_factorsTable = pd.DataFrame()
    A_factorsTable = pd.DataFrame()
    for col in df.columns:
        # # # print(col)
        if ("Htm" in col) or ("Home" in col):
            new_col = col.replace("Htm_", "")
            new_col = new_col.replace("Home", "")
            # new_col = new_col.replace("at_H", "")
            H_factorsTable[new_col] = 0
            H_factorsTable[new_col] = df[col]
        elif ("Atm" in col) or ("Away" in col):
            new_col = col.replace("Atm_", "")
            new_col = new_col.replace("Away", "")
            # new_col = new_col.replace("at_A", "")
            A_factorsTable[new_col] = 0
            A_factorsTable[new_col] = df[col]
        else:
            H_factorsTable[col] = df[col]
            A_factorsTable[col] = df[col]

    T_factorsTable = pd.concat([H_factorsTable, A_factorsTable])
    T_factorsTable.to_csv(League + "_T_factorsTable.csv", index=False)


def odds_by_poisson(dataInitial, lastFixt):
    reducedData_M1 = dataInitial[['HomeTeam', 'AwayTeam', 'FTH_Goals', 'FTA_Goals']]
    reducedData_M1 = reducedData_M1.rename(columns={'FTH_Goals': 'HomeGoals', 'FTA_Goals': 'AwayGoals'})
    lastFixt = lastFixt[['HomeTeam', 'AwayTeam']]

    goal_model_data = pd.concat([reducedData_M1[['HomeTeam', 'AwayTeam', 'HomeGoals']].assign(home=1).rename(
        columns={'HomeTeam': 'team', 'AwayTeam': 'opponent', 'HomeGoals': 'goals'}),
        reducedData_M1[['AwayTeam', 'HomeTeam', 'AwayGoals']].assign(home=0).rename(
            columns={'AwayTeam': 'team', 'HomeTeam': 'opponent', 'AwayGoals': 'goals'})])

    poisson_model = smf.glm(formula="goals ~ home + team + opponent", data=goal_model_data,
                            family=sm.families.Poisson()).fit()

    # poisson_model.summary()

    def simulate_match(foot_model, homeTeam, awayTeam, max_goals=10):
        home_goals_avg = foot_model.predict(
            pd.DataFrame(data={'team': homeTeam, 'opponent': awayTeam, 'home': 1}, index=[1]))  # .values[0]
        away_goals_avg = foot_model.predict(
            pd.DataFrame(data={'team': awayTeam, 'opponent': homeTeam, 'home': 0}, index=[1]))  # .values[0]
        team_pred = [[poisson.pmf(i, team_avg) for i in range(0, max_goals + 1)] for team_avg in
                     [home_goals_avg, away_goals_avg]]
        return (np.outer(np.array(team_pred[0]), np.array(team_pred[1])))

    TeamH = []
    TeamA = []
    H_per_method_1 = []
    D_per_method_1 = []
    A_per_method_1 = []

    U_per = []

    for x in range(0, len(lastFixt)):
        Team1 = lastFixt.iloc[x, 0]
        Team2 = lastFixt.iloc[x, 1]

        TeamH.append(Team1)
        TeamA.append(Team2)

        poisson_model.predict(pd.DataFrame(data={'team': TeamH[x], 'opponent': TeamA[x],
                                                 'home': 1}, index=[1]))

        poisson_model.predict(pd.DataFrame(data={'team': TeamA[x], 'opponent': TeamH[x],
                                                 'home': 0}, index=[1]))

        H_vs_A = simulate_match(poisson_model, TeamH[x], TeamA[x], max_goals=10)
        # chel_sun = simulate_match(poisson_model, "Chelsea", "Sunderland", max_goals=10)

        Ov_Un = simulate_match(poisson_model, TeamH[x], TeamA[x], max_goals=2)
        Ov_Un[2][2] = 0
        Ov_Un[2][1] = 0
        Ov_Un[1][2] = 0

        U_per = np.append(U_per, np.sum(Ov_Un))

        O_odd = 1 / (1 - U_per)

        U_odd = 1 / U_per

        H_per_method_1 = np.append(H_per_method_1, np.sum(np.tril(H_vs_A, -1)))
        D_per_method_1 = np.append(D_per_method_1, np.sum(np.diag(H_vs_A)))
        A_per_method_1 = np.append(A_per_method_1, np.sum(np.triu(H_vs_A, 1)))

        H_odd_method_1 = 1 / H_per_method_1
        D_odd_method_1 = 1 / D_per_method_1
        A_odd_method_1 = 1 / A_per_method_1

    return (H_odd_method_1, D_odd_method_1, A_odd_method_1, O_odd, U_odd)


def check_method_success(selected_fixt_results, Hcol, Dcol, Acol, method_type):
    try:
        method_chcking = []
        for index, row in selected_fixt_results.iterrows():
            comparsion_flag = 0
            if row[Hcol] == min(row[Hcol], row[Dcol], row[Acol]):
                if row['Result'] == 0:
                    comparsion_flag = 1

            if row[Dcol] == min(row[Hcol], row[Dcol], row[Acol]):
                if row['Result'] == 1:
                    comparsion_flag = 1

            if row[Acol] == min(row[Hcol], row[Dcol], row[Acol]):
                if row['Result'] == 2:
                    comparsion_flag = 1

            method_chcking.append(comparsion_flag)

        selected_fixt_results[method_type + '_Sucess'] = method_chcking
    except Exception:
        pass

    return selected_fixt_results


def check_method_DA_success(selected_fixt_results, Hcol, Dcol, Acol, method_type):
    DrawAway_comparison = []
    try:
        for index, row in selected_fixt_results.iterrows():
            DrawAway_comparison_flag = 0
            if (row[Hcol] == min(row[Hcol], row[Dcol], row[Acol])):
                if row['Result'] == 0:
                    DrawAway_comparison_flag = 1

            if (row[Hcol] != min(row[Hcol], row[Dcol], row[Acol])):
                if row['Result'] != 0:
                    DrawAway_comparison_flag = 1

            DrawAway_comparison.append(DrawAway_comparison_flag)

        selected_fixt_results[method_type + '_Home_DrawAway'] = DrawAway_comparison

    except Exception:
        pass
    return selected_fixt_results


def check_method_DA_focused_success(selected_fixt_results, ML_odds, Bet_odds):
    DrawAway_comparison = []
    try:
        for index, row in selected_fixt_results.iterrows():
            DrawAway_comparison_flag = 0

            if abs(row[Bet_odds[0]] - row[Bet_odds[2]]) > 1:
                if (row[Bet_odds[0]] >= 1.3) and (row[Bet_odds[2]] >= 1.3):
                    if (row[ML_odds[0]] == min(row[ML_odds[0]], row[ML_odds[1]], row[ML_odds[2]])):
                        if row['Result'] == 0:
                            DrawAway_comparison_flag = 1
                    if (row[ML_odds[0]] != min(row[ML_odds[0]], row[ML_odds[1]], row[ML_odds[2]])):
                        if row['Result'] != 0:
                            DrawAway_comparison_flag = 1
                else:
                    DrawAway_comparison_flag = -1
            else:
                DrawAway_comparison_flag = -1

            DrawAway_comparison.append(DrawAway_comparison_flag)

        selected_fixt_results['ML_Home_DrawAway_Foc'] = DrawAway_comparison

    except Exception:
        pass
    return selected_fixt_results


def check_ML_lower_B365_success(selected_fixt_results):
    Bet365_comparison = []
    try:
        for index, row in selected_fixt_results.iterrows():
            Bet365_comparison_flag = "-"
            if (row['prH'] < row['B365H']) and (row['B365H'] == min(row['B365H'], row['B365D'], row['B365A'])):
                if row['Result'] == 0:
                    Bet365_comparison_flag = 1
                else:
                    Bet365_comparison_flag = 0

            if (row['prA'] < row['B365A']) and (row['B365A'] == min(row['B365H'], row['B365D'], row['B365A'])):
                if row['Result'] == 2:
                    Bet365_comparison_flag = 1
                else:
                    Bet365_comparison_flag = 0

            Bet365_comparison.append(Bet365_comparison_flag)

        selected_fixt_results['ML_B365'] = Bet365_comparison

    except Exception:
        pass

    return selected_fixt_results
