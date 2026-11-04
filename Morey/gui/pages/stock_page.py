from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QPushButton
)
from gui.widgets.chart_widget import ChartWidget
import func.st_date as st_date
from func.st_color import stock_info_color, change_sign
from gui.widgets.chart_tools import DrawingToolBar

class StockPage(QWidget):
    def __init__(self, stock_id, fetcher, watchlist_manager, drawing_manager):
        super().__init__()

        self.stock_id = stock_id

        self.fetcher = fetcher

        self.watchlist_manager = watchlist_manager

        self.drawing_manager = drawing_manager

        if (self.fetcher.get_stock_name(self.stock_id) != "Unknown"):
            self.show_stock_page()
        else:
            self.show_error_page()

    def show_stock_page(self):

        layout = QVBoxLayout()
        stock_label = QHBoxLayout()

        #stock info
        self.cur_stock_data = self.fetcher.get_latest_data(self.stock_id)
        self.sec_cur_stock_data = self.fetcher.get_latest_data(self.stock_id, -2)
        
        self.change_price = self.cur_stock_data.Close - self.sec_cur_stock_data.Close
        self.change = self.fetcher.get_latest_change(self.stock_id)
        self.ch_sign = change_sign(self.change)

        self.background_color, self.font_color = stock_info_color(self.sec_cur_stock_data, self.cur_stock_data, self.change)

        stock_label0 = self.show_stock_identity()
        stock_label1 = self.showstock_price()
        stock_label2 = self.show_stock_info()

        if not self.in_watchlist():
            self.watchlist_action_btn_text = "加入自選股"
        else:
            self.watchlist_action_btn_text = "刪除自選股"
        self.watchlist_action_btn = QPushButton(self.watchlist_action_btn_text)
        self.watchlist_action_btn.clicked.connect(self.watchlist_action)
        
        stock_label.addWidget(stock_label0, 4)
        stock_label.addWidget(stock_label1, 8)
        stock_label.addWidget(stock_label2, 2)
        stock_label.addWidget(self.watchlist_action_btn, 1)

        layout.addLayout(stock_label, 1)



        chart = ChartWidget(self.stock_id, self.drawing_manager)
        toolbar = DrawingToolBar(chart)

        layout.addWidget(toolbar)
        df = self.fetcher.get_data(self.stock_id, st_date.StockDate().get_past_date(years=2))
        if df.empty:

            error_label = QLabel("此股票無歷史資料")
            error_label.setStyleSheet("""
                font-size: 40px;
                font-weight: bold;
            """)
            layout.addWidget(error_label)

        else:
            chart.plot_stock(df)
            layout.addWidget(chart, 6)

        self.setLayout(layout)

    def show_error_page(self):
        layout = QVBoxLayout()

        error_label = QLabel("找不到該股票")

        error_label.setStyleSheet("""
            font-size: 40px;
            font-weight: bold;
        """)

        layout.addWidget(error_label)

        self.setLayout(layout)

    def show_stock_identity(self):
        stock_name = self.fetcher.get_stock_name(self.stock_id)
        label = QLabel( f"""
        <span style="
            font-size: 50px;
            font-weight: bold;
        ">
            {stock_name}
        </span>

            <br>

        <span style="
            color: gray;
            font-size: 30px;
            font-weight: bold;
        ">
            {self.stock_id}
        </span>
        """)
        return label
    
    def showstock_price(self):
        label = QLabel(f"""
        <span style="
            font-size: 40px;
            font-weight: bold;
            color: {self.font_color[0]};
            background-color: {self.background_color[0]};
        ">
            {self.cur_stock_data.Close:.2f} 
        </span>

            <br>

        <span style="
            font-size: 20px;
            font-weight: bold;
            color: {self.font_color[1]};
            background-color: {self.background_color[1]};
            padding: 5px;
        ">
            {self.change_price:.2f}({self.ch_sign}{self.change:.2f}%)
        </span>

        """)
        return label

    def show_stock_info(self):

        label = QLabel(f"""
        開<span style="
            font-size: 20px;
            font-weight: bold;
            color: {self.font_color[2]};
            background-color: {self.background_color[2]};
            padding: 5px;
        ">
            {self.cur_stock_data.Open:.2f}  
        </span>

        量<span style="
            font-size: 20px;
            font-weight: bold;
            padding: 5px;
        ">
            {self.cur_stock_data.Volume}
        </span>

            <br>

        高<span style="
            font-size: 20px;
            font-weight: bold;
            color: {self.font_color[3]};
            background-color: {self.background_color[3]};
            padding: 5px;
        ">
            {self.cur_stock_data.High:.2f}
        </span>

        低<span style="
            font-size: 20px;
            font-weight: bold;
            color: {self.font_color[4]};
            background-color: {self.background_color[4]};
            padding: 5px;
        ">
            {self.cur_stock_data.Low:.2f}
        </span>
        """)
        return label
    
    def in_watchlist(self):
        if self.stock_id in self.watchlist_manager.get_all_stocks():
            return True
        else:
            return False
    
    def watchlist_action(self):
        if not self.in_watchlist():
            self.watchlist_manager.add_stock(self.stock_id)
            self.watchlist_action_btn_text = "刪除自選股"
        else:
            self.watchlist_manager.remove_stock(self.stock_id)
            self.watchlist_action_btn_text = "加入自選股"
        self.watchlist_action_btn.setText(
        self.watchlist_action_btn_text
    )


    