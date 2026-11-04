from PySide6.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QLabel,
    QSizePolicy
)
from func.st_color import stock_info_color, change_sign

class StockListCard(QWidget):

    def __init__(self, stock_id, fetcher):


        super().__init__()

        self.stock_id = stock_id

        self.fetcher = fetcher

        layout = QHBoxLayout()

        stock_name = fetcher.get_stock_name(stock_id)

        stock_identity = QLabel( f"""
        <span style="
            font-size: 20px;
            font-weight: bold;
        ">
            {stock_name}
        </span>

            <br>

        <span style="
            color: gray;
            font-size: 12px;
            font-weight: bold;
        ">
            {self.stock_id}
        </span>
        """)

        self.cur_stock_data = self.fetcher.get_latest_data(self.stock_id)
        self.sec_cur_stock_data = self.fetcher.get_latest_data(self.stock_id, -2)
        
        self.change_price = self.cur_stock_data.Close - self.sec_cur_stock_data.Close
        self.change = self.fetcher.get_latest_change(self.stock_id)
        self.ch_sign = change_sign(self.change)

        self.background_color, self.font_color = stock_info_color(self.sec_cur_stock_data, self.cur_stock_data, self.change)

        stock_price = QLabel(f"""
        <span style="
            font-size: 20px;
            font-weight: bold;
            color: {self.font_color[0]};
            background-color: {self.background_color[0]};
        ">
            {self.cur_stock_data.Close:.2f} 
        </span>

            <br>

        <span style="
            font-size: 12px;
            font-weight: bold;
            color: {self.font_color[1]};
            background-color: {self.background_color[1]};
            padding: 5px;
        ">
            {self.change_price:.2f}({self.ch_sign}{self.change:.2f}%)
        </span>

        """)

        
        layout.addWidget(stock_identity)

        layout.addWidget(stock_price)

        self.setLayout(layout)

        self.setFixedHeight(80)

        """self.setSizePolicy(
            QSizePolicy.Fixed,
            QSizePolicy.Expanding
        )"""

        self.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Fixed
        )