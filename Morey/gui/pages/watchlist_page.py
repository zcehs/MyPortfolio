from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QListWidget,
    QListWidgetItem
)

from gui.widgets.stock_list_card import StockListCard
class WatchListPage(QWidget):

    def __init__(self,fetcher, watchlist):

        super().__init__()

        self.watchlist = watchlist

        self.fetcher = fetcher

        layout = QVBoxLayout()

        self.stock_list = QListWidget()

        self.stock_list_card = []

        layout.addWidget(self.stock_list)

        self.setLayout(layout)

        self.refresh()

    def refresh(self):

        self.stock_list.clear()

        stocks = self.watchlist.get_all_stocks()

        for stock in stocks:

            # 建立 QListWidgetItem
            item = QListWidgetItem()

            # 建立自訂 Widget
            card = StockListCard(stock, self.fetcher)

            # 設定 item 大小
            item.setSizeHint(card.sizeHint())

            # 加入 list
            self.stock_list.addItem(item)

            # 把 widget 放進 item
            self.stock_list.setItemWidget(item, card)