import pandas as pd
import mplfinance as mpf


class StockChartAnalyzer:

    def __init__(self):
        pass

    # =========================
    # 
    # =========================

    def add_ma(self, df, windows=[5, 10, 20]):
        for w in windows:
            df[f"MA{w}"] = df["Close"].rolling(window=w).mean()
        return df

    def add_rsi(self, df, period=14):

        delta = df["Close"].diff()

        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()

        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))

        return df

    # =========================
    # =========================

    def plot_candle(self, df, title="Stock K Line"):

        df_plot = df.copy()

        addplots = []

        # MA線
        if "MA5" in df_plot.columns:
            addplots.append(mpf.make_addplot(df_plot["MA5"], color="blue"))
        if "MA10" in df_plot.columns:
            addplots.append(mpf.make_addplot(df_plot["MA10"], color="orange"))
        if "MA20" in df_plot.columns:
            addplots.append(mpf.make_addplot(df_plot["MA20"], color="green"))

        mpf.plot(
            df_plot,
            type="candle",
            volume=True,
            addplot=addplots if addplots else None,
            title=title,
            style="charles"
        )