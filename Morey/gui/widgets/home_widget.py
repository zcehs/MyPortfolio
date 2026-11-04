from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel
)

from PySide6.QtCore import Qt


class HomeWidget(QWidget):

    def __init__(self):

        super().__init__()

        layout = QVBoxLayout()

        title = QLabel("Stock Analyzer")

        title.setAlignment(Qt.AlignCenter)

        title.setStyleSheet("""
            font-size: 32px;
            font-weight: bold;
        """)

        layout.addWidget(title)

        self.setLayout(layout)