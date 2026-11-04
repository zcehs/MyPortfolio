from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QListWidget,
)


class WatchListWidget(QWidget):

    def __init__(self, watchlist):

        super().__init__()

        self.watchlist = watchlist

        layout = QVBoxLayout()

        self.stock_list = QListWidget()

        layout.addWidget(self.stock_list)

        self.setLayout(layout)

        self.refresh()

    def refresh(self):

        self.stock_list.clear()

        stocks = self.watchlist.get_all_stocks()

        self.stock_list.addItems(stocks)