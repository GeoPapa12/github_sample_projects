import pandas as pd
import numpy as np
from datetime import datetime
from datetime import datetime as dt
from datetime import date
from sklearn.model_selection import train_test_split
import sys

sys.path.append("..")

from EDA_ML_Package.ML_functions import ML_models
from scipy.stats import poisson, skellam
import statsmodels.api as sm

import statsmodels.formula.api as smf
from selenium import webdriver
from bs4 import BeautifulSoup
from sklearn.ensemble import RandomForestClassifier
import time
from Data_Construction_Functions import *


def parse_next_fixture():
    url = "https://www.google.com/search?q=premier+league+next+fixtures&oq=premier+league+&aqs=chrome.0.69i59j0i512j69i57j0i512l2j69i60l3.10295j1j7&sourceid=chrome&ie=UTF-8#sie=lg;/g/11p44qhs93;2;/m/02_tc;mt;fp;1;;"
    driver = webdriver.Chrome("C:/Users/user/Documents/chromedriver.exe")
    driver.get(url)

    Home_team = []
    Away_team = []

    soup = BeautifulSoup(driver.page_source, 'html.parser')

    time.sleep(5)

    containers = soup.findAll("div", {"class": "ellipsisize"})
    driver.close()

    for i, container in enumerate(containers):

        teams = [x.get_text() for x in container.findAll("span")]
        try:
            if i % 2 == 1:
                Home_team.append(teams[0])
            else:
                Away_team.append(teams[0])

        except Exception:
            pass

    harmonise_team_names = {
                            "Leeds United": "Leeds",
                            "Norwich City": "Norwich",
                            "Leicester City": "Leicester"
                            }

    parsed_fixt = pd.DataFrame({"HomeTeam": Home_team, "AwayTeam": Away_team})
    for key in harmonise_team_names:
        parsed_fixt["HomeTeam"].replace({key: harmonise_team_names[key]}, inplace=True)
        parsed_fixt["AwayTeam"].replace({key: harmonise_team_names[key]}, inplace=True)

    return parsed_fixt


def load_bet365(league, noGames):

    if league == "E0":
        url = "https://www.bet365.com/#/AC/B1/C1/D13/E37628398/F2/"  # England
        url = "https://www.bet365.gr/#/AC/B1/C1/D1002/E61683472/G40/H^1/"
    elif league == "D1":
        url = "https://www.bet365.com/#/AC/B1/C1/D13/E42422049/F2/"  # Germany
    elif league == "I1":
        url = "https://www.bet365.com/#/AC/B1/C1/D13/E42856517/F2/"  # italy
    elif league == "SP1":
        url = "https://www.bet365.com/#/AC/B1/C1/D13/E42493286/F2/"  # Spain

    if league == "Premier League":
        url = "https://www.bet365.com/#/AC/B1/C1/D13/E37628398/F2/"  # England
    elif league == "Bundesliga":
        url = "https://www.bet365.com/#/AC/B1/C1/D13/E42422049/F2/"  # Germany
    elif league == "Serie A":
        url = "https://www.bet365.com/#/AC/B1/C1/D13/E42856517/F2/"  # italy
    elif league == "Primera Division":
        url = "https://www.bet365.com/#/AC/B1/C1/D13/E42493286/F2/"  # Spain

    driver = webdriver.Chrome("C:/Users/user/Documents/chromedriver.exe")

    driver.get(url)
    driver.get(url)

    time.sleep(5)

    print("============================================================================")
    soup = BeautifulSoup(driver.page_source, 'html.parser')

    teams_vs = []

    containers = soup.findAll("div", {"class": ["sl-CouponParticipantWithBookCloses sl-MarketCouponAdvancedBase_LastChild", "sl-CouponParticipantWithBookCloses"]})
    driver.close()

    for container in containers:
        teams = [x.get_text() for x in container.findAll(
            "div", {"class": "sl-CouponParticipantWithBookCloses_NameContainer"}
        )]

        teams_vs.append(teams[0])
        print(teams)

    all_odds = []
    # count = 0
    count_odds = 0
    containersOdds = soup.findAll("div", {"class": ["gll-ParticipantOddsOnlyDarker gll-Participant_General gll-ParticipantOddsOnly",
                                                    "gll-ParticipantOddsOnlyDarker gll-Participant_General gll-ParticipantOddsOnly sl-MarketCouponAdvancedBase_LastChild"]})

    # gll-ParticipantOddsOnlyDarker gll-Participant_General gll-ParticipantOddsOnly sl-MarketCouponAdvancedBase_LastChild
    for container in containersOdds:
        count_odds = count_odds + 1

        odds = [x.get_text() for x in container.findAll(
            "span", {"class": "gll-ParticipantOddsOnly_Odds"}
        )]

        all_odds.append(odds[0])

    if all_odds[0].find('/') > 0:

        Home = all_odds[:len(all_odds)//3][:noGames]
        Home_dem = [int(x.split('/')[0])/int(x.split('/')[1])+1 for x in Home]
        Draw = all_odds[len(all_odds)//3:][:len(all_odds)//3][:noGames]
        Draw_dem = [int(x.split('/')[0])/int(x.split('/')[1])+1 for x in Draw]
        Away = all_odds[len(all_odds)//3:][len(all_odds)//3:][:noGames]
        Away_dem = [int(x.split('/')[0])/int(x.split('/')[1])+1 for x in Away]

    Home_Team = [x.split(' v ')[0] for x in teams_vs][:noGames]
    Away_Team = [x.split(' v ')[1] for x in teams_vs][:noGames]
    odds = [list(a) for a in zip(Home_dem, Draw_dem, Away_dem)]
    df = pd.DataFrame(list(zip(Home_Team, Away_Team, Home_dem, Draw_dem, Away_dem)))

    print(df)

    if league == "Premier League":
        league = "E0"
    elif league == "Bundesliga":
        league = "D1"
    elif league == "Serie A":
        league = "I1"
    elif league == "Primera Division":
        league = "SP1"

    df.to_excel('last_bet_' + league + '.xlsx')

    return 'nothing'

    # ==================================================================================
    # Odds
    # containersOdds = soup.findAll("div", {"class": "sl-MarketCouponValuesExplicit33 gll-Market_General gll-Market_PWidth-12-3333 sl-MarketCouponAdvancedBase_AdditionalParticipantHeight"})

    # sl-MarketCouponValuesExplicit33 gll-Market_General gll-Market_PWidth-12-3333 sl-MarketCouponAdvancedBase_AdditionalParticipantHeight
    # gll-ParticipantOddsOnlyDarker gll-Participant_General gll-ParticipantOddsOnly
    # span
    # gll-ParticipantOddsOnly_Odds

    # Teams
    # sl-CouponParticipantWithBookCloses_Name
    # sl-CouponParticipantWithBookCloses_NameContainer
    # sl-CouponParticipantWithBookCloses sl-CouponParticipantIPPGBase
