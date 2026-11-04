from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel
)

from gui.widgets.chart_widget import ChartWidget


class ChartPage(QWidget):

    def __init__(self):

        super().__init__()

        layout = QVBoxLayout()

        self.stock_label = QLabel()

        self.chart_widget = ChartWidget()

        layout.addWidget(self.stock_label)

        layout.addWidget(self.chart_widget)

        self.setLayout(layout)

    def show_stock(
        self,
        stock_id,
        df
    ):

        self.stock_label.setText(
            f"股票代號：{stock_id}"
        )

        self.chart_widget.plot_stock(df)