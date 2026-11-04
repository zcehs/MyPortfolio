from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QGridLayout
)

from gui.widgets.stock_list_card import StockListCard
class WatchListPage(QWidget):

    def __init__(self, fetcher, watchlist):

        super().__init__()

        self.watchlist = watchlist

        self.fetcher = fetcher

        layout = QVBoxLayout()

        grid = QGridLayout()

        self.stock_list = self.watchlist.get_all_stocks()

        self.stock_list_card = []

        for stock in self.stock_list:

            card = StockListCard(stock, self.fetcher)
            self.stock_list_card .append(card)
            grid.addWidget(card)

        layout.addLayout(grid)

        self.setLayout(layout)

        self.refresh()

    def refresh(self):

        self.stock_list.clear()

        stocks = self.watchlist.get_all_stocks()

        self.stock_list.append(stocks)