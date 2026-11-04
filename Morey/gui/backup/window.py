from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QListWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import mplfinance as mpf

import sys


class MainWindow(QWidget):

    def __init__(self, watchlist, fetcher, analyzer):

        super().__init__()

        self.watchlist = watchlist
        self.fetcher = fetcher
        self.analyzer = analyzer

        self.setWindowTitle("Stock Analyzer")

        self.resize(1200, 800)

        self.setup_ui()

    def setup_ui(self):

        # =========================
        # 主Layout
        # =========================

        main_layout = QHBoxLayout()

        # =========================
        # 左邊：股票列表
        # =========================

        self.stock_list = QListWidget()

        stocks = self.watchlist.get_all_stocks()

        self.stock_list.addItems(stocks)

        # 點擊事件
        self.stock_list.itemClicked.connect(
            self.on_stock_clicked
        )

        main_layout.addWidget(self.stock_list, 1)

        # =========================
        # 右邊：K線圖
        # =========================

        self.figure = Figure()

        self.canvas = FigureCanvas(self.figure)

        main_layout.addWidget(self.canvas, 4)

        self.setLayout(main_layout)

    def on_stock_clicked(self, item):

        stock_id = item.text()

        print(f"Selected: {stock_id}")

        # 抓資料
        df = self.fetcher.get_daily_data(stock_id)

        # 技術分析
        df = self.analyzer.add_ma(df)

        # 清空舊圖
        self.figure.clear()

        ax = self.figure.add_subplot(111)

        # 畫K線
        mpf.plot(
            df,
            type='candle',
            mav=(5, 10, 20),
            volume=False,
            ax=ax,
            style='charles'
        )

        # 更新畫面
        self.canvas.draw()


def run_app(watchlist, fetcher, analyzer):

    app = QApplication(sys.argv)

    window = MainWindow(
        watchlist,
        fetcher,
        analyzer
    )

    window.show()

    sys.exit(app.exec())