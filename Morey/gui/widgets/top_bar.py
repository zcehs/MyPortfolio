from PySide6.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QCompleter
)
from PySide6.QtGui import QIcon


class TopBar(QWidget):

    def __init__(self, stock_dict):

        super().__init__()

        self.stock_dict = stock_dict

        layout = QHBoxLayout()

        self.search_input = QLineEdit()

        self.search_input.setPlaceholderText(
            "輸入股票代號或名稱"
        )

        # Auto Complete
        completer_list = []

        for stock_id, stock_name in self.stock_dict.items():

            completer_list.append(stock_id)

            completer_list.append(stock_name)

        completer = QCompleter(completer_list)

        self.search_input.setCompleter(completer)

        self.search_button = QPushButton()

        self.search_button.setIcon(QIcon("Icon/search.png"))

        #self.search_button.setIconSize(QSize(24, 24))

        layout.addWidget(self.search_input)

        layout.addWidget(self.search_button)

        self.setLayout(layout)