# Python 3.6.0 |Anaconda 4.3.1 (64-bit)|

import pandas as pd
from datetime import datetime

def convert_kickoff_to_date(df):
    
    """
    Converts kickoff column from sting to datetime.
    Returns updated dataframe.
    """
    
    from_str_to_date = lambda date: datetime.strptime(date, "%m/%d/%Y")
    df.kickoff = [from_str_to_date(date) for date in list(df.kickoff)]
    
    return df



def scale_stats_per_minute(df):
    
    """
    Scale features in ['CLR','FC','FK','FS','G','GA','GC','GK','PAS','R','S','SAV','TA','Y'] per minute played
    Returns updated df
    """
    
    columns_to_scale = ['CLR', 'FC', 'FK', 'FS', 'G', 'GA', 'GC', 
                    'GK', 'PAS', 'R', 'S', 'SAV', 'TA', 'Y']

    for column in columns_to_scale:
        df[column] = df[column] / df.MINS
        
    return df



def lineup_by_date(df, date, team):
    
    """
    Returns team's lineup for a given date as a dictionary.
    """
    
    
    df = df[df.kickoff == date]
    df = df[df.team == team]
    
    forwards = df[df.position == "forward"].player.tolist()
    defenders = df[df.position == "defender"].player.tolist()
    goalkeepers = df[df.position == "goalkeeper"].player.tolist()
    midfielders = df[df.position == "midfielder"].player.tolist()    

    lineup = {

        "team" : team,
        "date" : date,
        "for" : forwards,
        "mid" : midfielders,
        "def" : defenders,
        "goal" : goalkeepers
    }
    
    return lineup



def attack(df, lineup):
    
    """
    Given lineup, assess team's attacking potential.
    Attack is represented with 'G', 'GA', 'S', 'PAS' featues.
    Each statistic is the sum of players' averages appearing in lineup.
    Returns dictionary.
    """

    forwards = {}
    forwards_avg = []

    for player in lineup["for"]:
    
        date = lineup["date"]
        player_df = df[(df.player == player) & (df.kickoff < date)]

        features = ['G', 'GA', 'S', 'PAS']
        player_df = player_df.loc[:, features] 
        player_avg_performance = player_df.mean().to_dict()
    
        forwards_avg.append(player_avg_performance)
        
    forwards_df = pd.DataFrame(forwards_avg).round(4)
    forwards_df.columns = [key + "_f" for key in forwards_df.keys()]
    
    forwards = forwards_df.sum().to_dict()
    
    
    return forwards



def midfield(df, lineup):
    
    """
    Given lineup, assess team's creative potential and ball control.
    Midfield is represented with 'G', 'GA', 'S', 'FC', 'FS', 'PAS' featues.
    Each statistic is the sum of players' averages appearing in lineup.
    Returns dictionary.
    """

    midfielders = {}
    midfielders_avg = []

    for player in lineup["mid"]:
    
        date = lineup["date"]
        player_df = df[(df.player == player) & (df.kickoff < date)]

        main_features = ['G', 'GA', 'S', 'FC', 'FS']
        midfield_features = ['PAS']
        
        player_main_df = player_df.loc[:, main_features]
        player_avg_performance_main = player_main_df.mean().to_dict()
        midfielder_df = player_df[player_df.position == 'midfielder']
        midfielder_df = midfielder_df.loc[:, midfield_features]
        player_avg_performance_midfield = midfielder_df.mean().to_dict()
        midfielder = {**player_avg_performance_main, **player_avg_performance_midfield}
    
        midfielders_avg.append(midfielder)
        
    midfielders_df = pd.DataFrame(midfielders_avg).round(4)
    midfielders_df.columns = [key + "_m" for key in midfielders_df.keys()]
    
    midfielders = midfielders_df.sum().to_dict()
    
    
    return midfielders



def defence(df, lineup):
    
    """
    Given lineup, assess team's defending potential.
    Defence is represented with 'G', 'GA', 'S', 'FC', 'FS', 'TA', 'CLR' featues.
    Each statistic is the sum of players' averages appearing in lineup.
    Returns dictionary.
    """

    defenders = {}
    defenders_avg = []

    for player in lineup["def"]:
    
        date = lineup["date"]
        player_df = df[(df.player == player) & (df.kickoff < date)]

        main_features = ['G', 'GA', 'S', 'FC', 'FS']
        defender_features = ['TA', 'CLR']
        
        player_main_df = player_df.loc[:, main_features]
        player_avg_performance_main = player_main_df.mean().to_dict()
        defender_df = player_df[player_df.position == 'defender']
        defender_df = defender_df.loc[:, defender_features]
        player_avg_performance_defence = defender_df.mean().to_dict()
        defender = {**player_avg_performance_main, **player_avg_performance_defence}
    
        defenders_avg.append(defender)
        
    defenders_df = pd.DataFrame(defenders_avg).round(4)
    defenders_df.columns = [key + "_d" for key in defenders_df.keys()]
    
    defenders = defenders_df.sum().to_dict()
    
    
    return defenders