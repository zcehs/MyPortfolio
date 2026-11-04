from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QStackedWidget
)

from gui.widgets.sidebar import SideBar
from gui.widgets.top_bar import TopBar
from PySide6.QtCore import Signal

from gui.pages.home_page import HomePage
from gui.pages.watchlist_page import WatchListPage
from gui.pages.stock_page import StockPage


class MainWindow(QWidget):

    def __init__(
        self,
        watchlist,
        fetcher,
        analyzer,
        drawing_manager
    ):

        super().__init__()

        self.watchlist = watchlist
        self.fetcher = fetcher
        self.analyzer = analyzer
        self.drawing_manager = drawing_manager

        self.resize(1400, 900)

        self.setWindowTitle("Morey")

        self.setup_ui()

    def setup_ui(self):

        # =========================
        # 主Layout
        # =========================

        main_layout = QVBoxLayout()

        # =========================
        # Top Bar
        # =========================

        self.top_bar = TopBar(self.fetcher.get_stock_dict())

        main_layout.addWidget(self.top_bar)

        # =========================
        # 下方內容區
        # =========================

        content_layout = QHBoxLayout()

        # Sidebar
        self.sidebar = SideBar()

        content_layout.addWidget(
            self.sidebar,
            1
        )

        # 頁面切換
        self.stack = QStackedWidget()

        # 首頁
        self.home_page = HomePage(self.fetcher)

        # 自選股頁
        self.watchlist_page = WatchListPage(
            self.fetcher,
            self.watchlist
        )

        self.stack.addWidget(self.home_page)

        self.stack.addWidget(self.watchlist_page)

        content_layout.addWidget(
            self.stack,
            5
        )

        main_layout.addLayout(content_layout)

        self.setLayout(main_layout)

        # =========================
        # Sidebar事件
        # =========================

        self.sidebar.home_button.clicked.connect(
            self.show_home_page
        )

        self.sidebar.watchlist_button.clicked.connect(
            self.show_watchlist_page
        )

        # =========================
        # Topbar事件
        # =========================

        self.top_bar.search_button.clicked.connect(
            lambda: self.show_stock_page(self.top_bar.search_input.text().strip())
        )

        # =========================
        # homePage事件
        # =========================

        self.home_page.main_card.clicked.connect(
            self.show_stock_page
        )

        for card in self.home_page.cards:
            card.clicked.connect(self.show_stock_page)

        # =========================
        # watchlistPage事件
        # =========================

        """for card in self.watchlist_page.stock_list_card:
            card.clicked.connect(
                lambda checked=False, s=card.stock_id:
                    self.show_stock_page(s)
            )"""

        self.watchlist_page.stock_list.itemClicked.connect(
            self.on_stock_clicked
        )

    def show_home_page(self):

        self.stack.setCurrentWidget(
            self.home_page
        )

    def show_watchlist_page(self):

        self.watchlist_page.refresh()

        self.stack.setCurrentWidget(
            self.watchlist_page
        )

    def show_stock_page(self, text):

        stock_dict = self.fetcher.get_stock_dict()

        stock_idex = None

        for stock_id, stock_name in stock_dict.items():

            if text == stock_id or text == stock_name:
                stock_idex = stock_id
                break

        if stock_idex is None:
            stock_idex = text

        stock_page = StockPage(
            stock_idex,
            self.fetcher,
            self.watchlist,
            self.drawing_manager
        )

        self.stack.addWidget(stock_page)

        self.stack.setCurrentWidget(stock_page)

    def on_stock_clicked(self, item):
        # 取得 item 對應的 widget
        card = self.watchlist_page.stock_list.itemWidget(item)

        # 取得 stock_id
        stock_id = card.stock_id

        self.show_stock_page(stock_id)