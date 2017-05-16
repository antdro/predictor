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