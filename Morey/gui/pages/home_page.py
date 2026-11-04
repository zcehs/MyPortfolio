from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QGridLayout
)


from gui.widgets.stock_card import StockCard
from PySide6.QtCore import Signal




class HomePage(QWidget):

    stock_clicked = Signal(str)
        
    def __init__(self, fetcher):

        super().__init__()

        self.fetcher = fetcher

        layout = QVBoxLayout()

        grid = QGridLayout()

        main_stock = "TAIEX"

        self.main_card = StockCard(main_stock, self.fetcher)

        self.main_card.clicked.connect(self.stock_clicked.emit)

        layout.addWidget(self.main_card, 1)

        self.stocks = ["2330", "2317", "2454", "0050"]

        self.cards = []

        row = 0
        col = 0

        for stock in self.stocks:

            card = StockCard(stock, self.fetcher)

            card.clicked.connect(self.stock_clicked.emit)

            grid.addWidget(card, row, col)

            self.cards.append(card)

            col += 1

            if col > 1:
                col = 0
                row += 1

        layout.addLayout(grid, 1)

        self.setLayout(layout)
    
