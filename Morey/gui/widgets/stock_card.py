from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QSizePolicy
)

from gui.widgets.chart_preview import MiniChartWidget
import func.st_date as st_date
from PySide6.QtCore import Signal
from func.st_color import stock_info_color, change_sign

class StockCard(QWidget):

    clicked = Signal(str)

    def __init__(self, stock_id, fetcher):

        super().__init__()

        self.stock_id = stock_id
        self.fetcher = fetcher

        layout = QVBoxLayout()

        stock_name = fetcher.get_stock_name(stock_id)

        self.cur_stock_data = self.fetcher.get_latest_data(self.stock_id)
        self.sec_cur_stock_data = self.fetcher.get_latest_data(self.stock_id, -2)
        
        self.change_price = self.cur_stock_data.Close - self.sec_cur_stock_data.Close
        self.change = self.fetcher.get_latest_change(self.stock_id)
        self.ch_sign = change_sign(self.change)

        self.background_color, self.font_color = stock_info_color(self.sec_cur_stock_data, self.cur_stock_data, self.change)

        info_label = QLabel(
            f"""
            <span style="font-size: 18px; font-weight: bold;">
                {stock_name}
            </span>
            <br>
            <span style="color: gray;font-size: 12px; font-weight: bold;">
                {stock_id}
            </span>
            <br>
            價格：<span style="
            font-weight: bold;
            color: {self.font_color[0]};
            background-color: {self.background_color[0]};">
            {self.cur_stock_data.Close:.2f}元
            </span>
            <br>
            漲跌幅:<span style="
            font-weight: bold;
            color: {self.font_color[1]};
            background-color: {self.background_color[1]};
            padding: 5px;
            ">
            {self.change_price:.2f}({self.ch_sign}{self.change:.2f}%)
            </span>
            """
        )

        layout.addWidget(info_label)

        df = self.fetcher.get_data(self.stock_id, st_date.StockDate().get_past_date(days = 30))


        mini_chart = MiniChartWidget(df)

        layout.addWidget(mini_chart)

        self.setLayout(layout)

        self.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding
        )

    def mousePressEvent(self, event):

        self.clicked.emit(self.stock_id)