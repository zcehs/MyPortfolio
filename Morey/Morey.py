import pandas as pd
import data.stock_info as si
import data.stock_list as sl
import analysis.chart as ch
import analysis.drawing_manager as dm
from datetime import datetime
from dateutil.relativedelta import relativedelta
from PySide6.QtWidgets import QApplication

import sys

from gui.main_window import MainWindow

import func.st_date as st_date

#conda activate morey
#python Morey.py


def main():

    # =========================
    # 建立物件
    # =========================

    app = QApplication(sys.argv)


    stock_info_manager = si.StockInfo()

    stock_lst_manager = sl.WatchListManager() 

    stock_lst = stock_lst_manager.get_all_stocks()

    #cache data
    for stock_id in stock_lst:
        stock_info_manager.get_data(stock_id, st_date.StockDate().get_past_date(years=10))

    chart_manager = ch.StockChartAnalyzer()

    drawing_manager = dm.DrawingManager()

    window = MainWindow(
        stock_lst_manager,
        stock_info_manager,
        chart_manager,
        drawing_manager
    )

    window.show()

    sys.exit(app.exec())

    # =========================
    # 加入自選股
    # =========================


if __name__ == "__main__":
    main()