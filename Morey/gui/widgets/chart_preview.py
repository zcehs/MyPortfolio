from PySide6.QtWidgets import QWidget, QVBoxLayout

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from func.st_color import only_stock_color

class MiniChartWidget(QWidget):

    def __init__(self, df):

        super().__init__()

        layout = QVBoxLayout()

        layout.setContentsMargins(
            0,
            0,
            0,
            0
        )

        self.figure = Figure(
            figsize=(2, 1),
            facecolor="none"
        )

        self.canvas = FigureCanvas(
            self.figure
        )

        self.canvas.setStyleSheet(
            "background: transparent"
        )

        self.color = only_stock_color(df["Close"].iloc[-2], df["Close"].iloc[-1])[1]

        layout.addWidget(self.canvas)

        self.setLayout(layout)

        self.plot(df)

    def plot(self, df):

        ax = self.figure.add_subplot(111)

        ax.set_facecolor("none")

        ax.plot(
            df.index,
            df["Close"],
            linewidth=1.5,
            color=self.color
        )

        # 不顯示刻度文字
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        # 不顯示刻度線
        ax.tick_params(
            axis='both',
            which='both',
            length=0
        )

        # 上右框線關掉
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(True)

        self.figure.tight_layout(
            pad=0
        )

        self.canvas.draw()