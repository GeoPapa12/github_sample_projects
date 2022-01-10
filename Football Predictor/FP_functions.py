import pandas as pd
from datetime import datetime
from datetime import datetime as dt
from datetime import date
from sklearn.model_selection import train_test_split
from EDA_ML_Functions.ML_functions import *
from sklearn.ensemble import RandomForestClassifier

pd.set_option('display.max_columns', 15)

def retrieve_csv_data(league):

    appended_data = []
    print("League: ", league)
    for yearPeriodXX in range(5, 22):
        if len(str(yearPeriodXX)) == 1:
            start_period = "0" + str(yearPeriodXX)
        else:
            start_period = str(yearPeriodXX)

        if len(str(yearPeriodXX+1)) == 1:
            end_period = "0" + str(yearPeriodXX + 1)
        else:
            end_period = str(yearPeriodXX + 1)

        period = start_period + end_period
        fullData = pd.read_csv("http://www.football-data.co.uk/mmz4281/" + period + "/" + league + ".csv", parse_dates=[1], dayfirst=True)

        print("Period: ", period)

        fullData = filter_and_rename_columns(fullData)
        fullData = table_construction(fullData, period)
        appended_data.append(fullData)

    appended_data = pd.concat(appended_data, sort=False)
    appended_data = appended_data[appended_data['Div'].notna()]  # remove row if div is nan
    return appended_data


def table_construction(df, period):

    def factors_based_on(attribute_type, df, tm):
        ''' Sub function that constracts the Home/ Away/ In favour/ Against factors'''
        dx = df[((df.HomeTeam == tm) | (df.AwayTeam == tm))][['fixt', 'HomeTeam', 'AwayTeam', 'FTH_' + attribute_type, 'FTA_' + attribute_type]]
        dx.loc[(dx.HomeTeam == tm), 'HTs_' + attribute_type + '_favour'] = dx[dx.HomeTeam == tm]['FTH_' + attribute_type].shift(1).rolling(min_periods=1, window=3).sum()
        dx.loc[(dx.AwayTeam == tm), 'ATs_' + attribute_type + '_favour'] = dx[dx.AwayTeam == tm]['FTA_' + attribute_type].shift(1).rolling(min_periods=1, window=3).sum()
        dx.loc[(dx.HomeTeam == tm), 'HTs_' + attribute_type + '_against'] = dx[dx.HomeTeam == tm]['FTA_' + attribute_type].shift(1).rolling(min_periods=1, window=3).sum()
        dx.loc[(dx.AwayTeam == tm), 'ATs_' + attribute_type + '_against'] = dx[dx.AwayTeam == tm]['FTH_' + attribute_type].shift(1).rolling(min_periods=1, window=3).sum()
        df.loc[(df.HomeTeam == tm), 'HTs_' + attribute_type + '_favour'] = round(dx[dx.HomeTeam == tm]['HTs_' + attribute_type + '_favour']/3, 2)
        df.loc[(df.AwayTeam == tm), 'ATs_' + attribute_type + '_favour'] = round(dx[dx.AwayTeam == tm]['ATs_' + attribute_type + '_favour']/3, 2)
        df.loc[(df.HomeTeam == tm), 'HTs_' + attribute_type + '_against'] = round(dx[dx.HomeTeam == tm]['HTs_' + attribute_type + '_against']/3, 2)
        df.loc[(df.AwayTeam == tm), 'ATs_' + attribute_type + '_against'] = round(dx[dx.AwayTeam == tm]['ATs_' + attribute_type + '_against']/3, 2)
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

    df['H_fifa'] = 60
    df['A_fifa'] = 50

    for tm in teamNames:

        fixt_len = len(df.loc[((df.HomeTeam == tm) | (df.AwayTeam == tm))])
        df.loc[((df.HomeTeam == tm) | (df.AwayTeam == tm)), 'fixt'] = range(1, fixt_len + 1)  # Needs Improvement
        df.loc[((df.HomeTeam == tm) | (df.AwayTeam == tm)), 'period'] = period

        df.loc[(df.HomeTeam == tm), 'H_csum'] = df[df.HomeTeam == tm].groupby(['HomeTeam', 'fixt'])['HomePoints'].sum().groupby(level=0).cumsum().reset_index()['HomePoints'].values
        df.loc[(df.AwayTeam == tm), 'A_csum'] = df[df.AwayTeam == tm].groupby(['AwayTeam', 'fixt'])['AwayPoints'].sum().groupby(level=0).cumsum().reset_index()['AwayPoints'].values

        dff = df[((df.HomeTeam == tm) | (df.AwayTeam == tm))][['fixt', 'HomeTeam', 'AwayTeam', 'HomePoints', 'AwayPoints', 'H_csum', 'A_csum']]
        dff.loc[(dff.HomeTeam == tm), 'TeamsPointsX'] = dff[dff.HomeTeam == tm]['HomePoints']
        dff.loc[(dff.AwayTeam == tm), 'TeamsPointsX'] = dff[dff.AwayTeam == tm]['AwayPoints']
        dff['TeamsPoints'] = dff.TeamsPointsX.cumsum()
        df.loc[(df.HomeTeam == tm), 'TH_csum'] = dff[dff.HomeTeam == tm]['TeamsPoints']
        df.loc[(df.AwayTeam == tm), 'TA_csum'] = dff[dff.AwayTeam == tm]['TeamsPoints']

        df = factors_based_on('Goals', df, tm)
        df = factors_based_on('ShTar', df, tm)
        df = factors_based_on('Sht', df, tm)

        if tm in Top1_tier:
            df.loc[df.HomeTeam == tm, 'H_fifa'] = 90
            df.loc[df.AwayTeam == tm, 'A_fifa'] = 80
        if tm in Top2_tier:
            df.loc[df.HomeTeam == tm, 'H_fifa'] = 80
            df.loc[df.AwayTeam == tm, 'A_fifa'] = 70
        if tm in Middle_tier:
            df.loc[df.HomeTeam == tm, 'H_fifa'] = 70
            df.loc[df.AwayTeam == tm, 'A_fifa'] = 60

    return df


def filter_and_rename_columns(df):
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
    df = df.drop(columns=['Date', 'Div', 'FTH_Goals', 'FTA_Goals'])
    df = df.drop(columns=['FTH_Sht', 'FTA_Sht', 'FTH_ShTar', 'FTA_ShTar', 'HFls', 'AFls', 'HCorn'])
    df = df.drop(columns=['ACorn', 'HYel', 'AYel', 'HomePoints', 'AwayPoints'])
    return df


def fixture_selection(df, periodX, fixtX):

    df.period = df.period.astype(str)

    def prepare_ML_table():
        '''Retrieve all the data before the selected period and fixture'''
        df1 = df[(df.period < periodX)]
        df2 = df[((df.period == periodX) & (df.fixt < fixtX))]
        df_ML_previous = pd.concat([df1, df2])
        df_ML_previous = df_ML_previous.dropna()
        df_ML_previous = remove_columns(df_ML_previous)
        return df_ML_previous

    def current_fixture():
        '''Retrieve the data of the selected fixture'''
        selected_fixt = df[((df.period == periodX) & (df.fixt == fixtX))]
        df_current_results = selected_fixt[['HomeTeam', 'AwayTeam', 'FTH_Goals', 'FTA_Goals', 'Result', 'B365H', 'B365D', 'B365A']]
        df_ML_current = remove_columns(selected_fixt)
        df_ML_current = df_ML_current.drop(columns=['Result'])

        return df_current_results, df_ML_current

    df_ML_previous = prepare_ML_table()
    # df_ML_previous.to_csv("df_ML_previous.csv", index=False)
    df_current_results, df_ML_current = current_fixture()

    return df_ML_previous, df_current_results, df_ML_current


def last_fixture_teams(periodX, fixtX):
    df = pd.read_csv('fullData.csv')
    selected_fixt_names = fixture_selection(df, periodX, fixtX)[1]
    selected_fixt_names = selected_fixt_names['HomeTeam'] + "/ " + selected_fixt_names['AwayTeam']
    selected_fixt_names = pd.DataFrame(selected_fixt_names, columns=['label'])
    selected_fixt_names['value'] = selected_fixt_names['label']
    return selected_fixt_names


def DA_ML_analysis(periodX, fixtX):
    df = pd.read_csv('fullData.csv')
    df_ML_previous, df_current_results, df_ML_current = fixture_selection(df, periodX, fixtX)

    y = df_ML_previous['Result']
    X = df_ML_previous.drop(columns=['Result'])
    X = X.drop(columns=['HomeTeam', 'AwayTeam', 'period', 'fixt'])
    # X = X.drop(columns=['B365<2.5', 'B365>2.5'])

    df_ML_current = df_ML_current.drop(columns=['HomeTeam', 'AwayTeam', 'period', 'fixt'])
    # df_ML_current = df_ML_current.drop(columns=['B365<2.5', 'B365>2.5'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(random_state=0)
    clf.fit(X, y)

    y_scores = clf.predict_proba(df_ML_current)*100
    predictions = round(pd.DataFrame(100/y_scores, columns=["prH", "prD", "prA"]), 2)
    df_current_results = df_current_results.reset_index(drop=True)
    df_current_results = df_current_results.join(predictions, how='outer')

    comparison = []
    for index, row in df_current_results.iterrows():
        if ((row['prH'] < row['prA']) & (row['prH'] < row['prD'])):
            if row['Result'] == 0:
                comparison.append(1)
            else:
                comparison.append(0)

        if ((row['prD'] < row['prH']) & (row['prD'] < row['prA'])):
            if row['Result'] == 1:
                comparison.append(1)
            else:
                comparison.append(0)

        if ((row['prA'] < row['prH']) & (row['prA'] < row['prD'])):
            if row['Result'] == 2:
                comparison.append(1)
            else:
                comparison.append(0)

    df_current_results['Success/Fail'] = comparison
    print(df_current_results)
    return df_current_results


def factors_current_fixt(periodX, fixtX):
    df = pd.read_csv('fullData.csv')
    _, _, df_ML_current = fixture_selection(df, periodX, fixtX)
    return round(df_ML_current.drop(columns=['period', 'fixt', 'B365H', 'B365D', 'B365A']), 2)


def table_ranking(periodX, fixtX):
    df = pd.read_csv('fullData.csv')
    _, _, df_ML_current = fixture_selection(df, periodX, fixtX)

    H_rank = df_ML_current[['HomeTeam', 'H_csum', 'TH_csum']]
    A_rank = df_ML_current[['AwayTeam', 'A_csum', 'TA_csum']]
    H_rank = H_rank.rename(columns={"HomeTeam": "Team", "TH_csum": "Total", "H_csum": "Points@H"})
    A_rank = A_rank.rename(columns={"AwayTeam": "Team", "TA_csum": "Total", "A_csum": "Points@A"})

    total_rank = pd.concat([H_rank, A_rank])
    total_rank = total_rank.fillna('-')
    total_rank = total_rank.sort_values(by='Total', ascending=False)
    total_rank['Pos'] = list(range(1, len(total_rank) + 1))
    total_rank = total_rank[['Pos', 'Team', 'Total', 'Points@H', 'Points@A']]

    return total_rank


def plot_factors_dataframes(periodX, fixtX, game, plot_type):
    df = pd.read_csv('fullData.csv')
    # selected_fixt_names = fixture_selection(df, periodX, fixtX)[1]
    df_ML_previous, _, _ = fixture_selection(df, periodX, fixtX)

    team1 = game.split("/ ")[0]
    team2 = game.split("/ ")[1]

    if plot_type == "Goals":
        Hlist = ['fixt', 'HTs_Goals_favour', 'HTs_Goals_against']
        Alist = ['fixt', 'ATs_Goals_favour', 'ATs_Goals_against']
    elif plot_type == 'Shoots':
        Hlist = ['fixt', 'HTs_Sht_favour', 'HTs_Sht_against']
        Alist = ['fixt', 'ATs_Sht_favour', 'ATs_Sht_against']
    elif plot_type == 'Shoots on Target':
        Hlist = ['fixt', 'HTs_ShTar_favour', 'HTs_ShTar_against']
        Alist = ['fixt', 'ATs_ShTar_favour', 'ATs_ShTar_against']

    dftm1 = df_ML_previous[(df_ML_previous.HomeTeam == team1) & (df_ML_previous.period == periodX)][Hlist]
    dftm2 = df_ML_previous[(df_ML_previous.AwayTeam == team2) & (df_ML_previous.period == periodX)][Alist]

    return dftm1, dftm2


if __name__ == "__main__":
    df = pd.read_csv('fullData.csv')
    # df = retrieve_csv_data("E0")
    # df.to_csv("fullData.csv", index=False)

    # table_ranking('2021', 33)
    # df_ML_previous, df_current_results, df_ML_current = fixture_selection(df, '2021', 34)
    # ML_results = ML_analysis(df_ML_previous, df_current_results, df_ML_current)

    periodX = '1617'
    fixtX = 34
    plot_type = "Goals"
    game = "Tottenham/ Arsenal"

    x1 = plot_factors_dataframes(periodX, fixtX, game, plot_type)
    x2 = table_ranking(periodX, fixtX)
    x3 = factors_current_fixt(periodX, fixtX)
    x4 = last_fixture_teams(periodX, fixtX)
    x5 = DA_ML_analysis(periodX, fixtX)

    # df_ML_previous, df_current_results, df_ML_current = fixture_selection(df, '2021', 34)
    # ML = ML_models()
    # y = df_ML_previous['Result']
    # y = df_ML_previous['Result'].map({0: 0, 1: 1, 2: 1})

    # X = df_ML_previous.drop(columns=['Result'])
    # X = X.drop(columns=['HomeTeam', 'AwayTeam', 'period', 'fixt'])
    # results = ML.ML_Basic_Models(X, y, test_size=0.25, test_case=0)

    # Buttons to update data
    
    # Add leagues
    
    # Bet365 Parsing