from selenium import webdriver
from bs4 import BeautifulSoup
import time
import pandas as pd


# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
def load_bet365(league, noGames):

    if league == "E0":
        url = "https://www.bet365.com/#/AC/B1/C1/D13/E37628398/F2/"  # England
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

# Read data online
load_bet365("E0", 10)
