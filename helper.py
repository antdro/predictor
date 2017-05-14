# Python 3.6.0 |Anaconda 4.3.1 (64-bit)|

from importer import pd
from datetime import datetime

def convert_kickoff_to_date(df):
    
    """
    Converts kickoff column from sting to datetime.
    Returns updated dataframe.
    """
    
    from_str_to_date = lambda date: datetime.strptime(date, "%m/%d/%Y")
    df.kickoff = [from_str_to_date(date) for date in list(df.kickoff)]
    
    return df
