import pandas as pd
import time
from FinMind.data import DataLoader
import configparser
import requests
import pandas as pd

class StockInfo:
    def __init__(self):
        self.api = self.get_api()
        self.cache = {}
        self.stock_dict = {}

        self.load_stock_list()

    def get_api(self):
        config = configparser.ConfigParser()
        config.read('config.ini')
        finmind_api_key = config.get('FinMind', 'api_key')
        api = DataLoader()
        api.login_by_token(api_token=finmind_api_key)
        return api

    
    def get_daily_data(self, stock_id):

        # 檢查 cache
        if stock_id in self.cache:
            #print("Use Cache")
            return self.cache[stock_id]

        #print("Download Data")

        df = self.api.taiwan_stock_daily(
            stock_id=stock_id,
            start_date="2014-01-01"
        )

        # 日期轉 datetime
        df["date"] = pd.to_datetime(df["date"])

        # 設定 index
        df.set_index("date", inplace=True)

        # 欄位重新命名（給 mplfinance 用）
        df.rename(columns={
            "open": "Open",
            "max": "High",
            "min": "Low",
            "close": "Close",
            "Trading_Volume": "Volume"
        }, inplace=True)

        # 存 cache
        self.cache[stock_id] = df

        return df
    
    def get_data(self, stock_id, start_date):

        df = self.get_daily_data(stock_id)

        return df[df.index >= start_date]
    
    def load_stock_list(self):

        url = "https://api.finmindtrade.com/api/v4/data"

        params = {
            "dataset": "TaiwanStockInfo"
        }

        response = requests.get(
            url,
            params=params
        )

        data = response.json()["data"]

        for item in data:

            stock_id = item["stock_id"]

            stock_name = item["stock_name"]

            self.stock_dict[stock_id] = stock_name

    def get_stock_dict(self):

        return self.stock_dict

    def search_stock(self, keyword):

        result = {}

        keyword = keyword.lower()

        for stock_id, stock_name in self.stock_dict.items():

            if (
                keyword in stock_id.lower()
                or keyword in stock_name.lower()
            ):

                result[stock_id] = stock_name

        return result
    
    def get_stock_name(self, stock_id):

        return self.stock_dict.get(stock_id, "Unknown")
    
    def get_stock_id(self, stock_name):

        for stock_id, name in self.stock_dict.items():

            if name == stock_name:
                return stock_id

        return None
    
    def get_latest_data(self, stock_id, latest = -1):

        df = self.get_daily_data(stock_id)

        return df.iloc[latest]
    
    def get_latest_change(self, stock_id):

        df = self.get_daily_data(stock_id)

        latest_price = df.iloc[-1].Close
        sec_latest_price = df.iloc[-2].Close

        change = 0

        if sec_latest_price != 0:
            change = (latest_price - sec_latest_price) / sec_latest_price * 100

        return change
