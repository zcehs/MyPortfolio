from datetime import datetime
import pandas as pd
from dateutil.relativedelta import relativedelta

class StockDate:
    def __init__(self):

        self.current_date = datetime.now().date()

    def get_current_date(self):

        return self.current_date
    
    def get_past_date(self, years=0, months=0, days=0):

        past_date = self.current_date - relativedelta(years=years, months=months, days=days)

        return pd.Timestamp(past_date)