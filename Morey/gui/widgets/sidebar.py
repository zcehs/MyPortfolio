from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QPushButton
)


class SideBar(QWidget):

    def __init__(self):

        super().__init__()

        layout = QVBoxLayout()

        # 首頁
        self.home_button = QPushButton("🏠 首頁")

        # 自選股
        self.watchlist_button = QPushButton("⭐ 自選股")

        # 市場
        self.market_button = QPushButton("📈 市場")

        # 設定
        self.setting_button = QPushButton("⚙️ 設定")

        layout.addWidget(self.home_button)

        layout.addWidget(self.watchlist_button)

        layout.addWidget(self.market_button)

        layout.addWidget(self.setting_button)

        layout.addStretch()

        self.setLayout(layout)