from PySide6.QtWidgets import QWidget, QVBoxLayout

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backend_bases import MouseButton

import matplotlib
import mplfinance as mpf
import pandas as pd

matplotlib.rcParams['font.family'] = 'Microsoft JhengHei'

matplotlib.rcParams['axes.unicode_minus'] = False
import analysis.drawing_line as dl


class ChartWidget(QWidget):

    def __init__(self, stock_id, drawing_manager):

        super().__init__()

        self.stock_id = stock_id
        self.drawing_manager = drawing_manager

        layout = QVBoxLayout()

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)

        layout.addWidget(self.canvas)
        self.setLayout(layout)

        self.df = None
        self.ax = None

        # =========================
        # 均線參數
        # =========================

        self.ma_settings = [
            {"enabled": True, "window": 5},
            {"enabled": True, "window": 10},
            {"enabled": True, "window": 20},
        ]

        self.show_volume = True

        # =========================
        # 畫線狀態
        # =========================

        self.points = []
        self.lines = []
        self.preview_artist = None

        self.full_df = None
        self.visible_df = None

        self.visible_days = 365 * 3  # 預設顯示近一年

        self.line_id = self.drawing_manager.get_next_line_id(self.stock_id)

        self.line_types = [
            "Line segment",
            "Straight line",
            "Horizontal line"
        ]

        self.tool = "cursor"  # cursor / draw_line
        self.line_type = "Line segment"
        self.color = "black"

        self.auto_snap = True
        self.snap_threshold = 0.01  # 1%，可調整

        self.dragging_line = None

        # 載入已儲存線
        saved_lines = self.drawing_manager.load_lines(self.stock_id)

        for line in saved_lines:
            self.lines.append(line)

        self.panning = False
        self.pan_start = None
        self.pan_xlim = None
        self.pan_ylim = None

        # =========================
        # matplotlib 事件
        # =========================

        self.canvas.mpl_connect(
            "button_press_event",
            self.on_click
        )

        self.canvas.mpl_connect(
            "pick_event",
            self.on_pick
        )

        self.canvas.mpl_connect(
            "motion_notify_event",
            self.on_motion
        )

        self.canvas.mpl_connect(
            "button_release_event",
            self.on_release
        )

        self.canvas.mpl_connect(
            "scroll_event",
            self.on_scroll
        )

    def plot_stock(self, df, keep_view=False):

        self.full_df = df

        self.df = self.get_visible_df(df)

        if self.df is None or self.df.empty:
            return

        old_xlim = None
        old_ylim = None

        if keep_view and self.ax is not None:
            old_xlim = self.ax.get_xlim()
            old_ylim = self.ax.get_ylim()

        self.figure.clear()

        if self.show_volume:
            gs = self.figure.add_gridspec(
                5,
                1
            )

            self.ax = self.figure.add_subplot(gs[:4, 0])

            self.volume_ax = self.figure.add_subplot(
                gs[4, 0],
                sharex=self.ax
            )

            self.ax.tick_params(
            axis='x',
            which='both',
            labelbottom=False
            )

            
        else:
            self.ax = self.figure.add_subplot(1, 1, 1)
            self.volume_ax = None

        mav = tuple(
            setting["window"]
            for setting in self.ma_settings
            if setting["enabled"]
        )

        if len(mav) == 0:
            mav = None

        mpf.plot(
            self.df,
            type="candle",
            mav=mav,
            volume=self.volume_ax if self.show_volume else False,
            ax=self.ax,
            style="charles"
        )

        

        # =========================
        # Hover 輔助線
        # =========================

        self.vline = self.ax.axvline(
            x=0,
            color="gray",
            linewidth=1,
            linestyle="--",
            visible=False
        )

        self.hline = self.ax.axhline(
            y=0,
            color="gray",
            linewidth=1,
            linestyle="--",
            visible=False
        )

        self.hover_text = self.ax.text(
            0.02,
            0.95,
            "",
            transform=self.ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(
                boxstyle="round",
                facecolor="white",
                alpha=0.8
            )
        )

        self.create_hover_tools()

        self.redraw_lines()

        if keep_view and old_xlim is not None:
            self.ax.set_xlim(old_xlim)
            self.ax.set_ylim(old_ylim)

            if self.volume_ax is not None:
                self.volume_ax.set_xlim(old_xlim)
        else:
            self.set_default_view()

        self.canvas.draw()

    # =========================
    # 工具列呼叫用 method
    # =========================

    def set_tool(self, tool):
        self.tool = tool
        self.points.clear()

    def set_line_type(self, line_type):
        if line_type in self.line_types:
            self.line_type = line_type
            self.points.clear()

    def set_color(self, color):
        self.color = color


    def set_ma_settings(self, ma_settings):

        self.ma_settings = ma_settings

        if self.full_df is not None:
            self.plot_stock(
                self.full_df,
                keep_view=True
            )


    def set_show_volume(self, checked):

        self.show_volume = checked

        if self.full_df is not None:
            self.plot_stock(
                self.full_df,
                keep_view=True
            )

    def set_default_view(self):

        if self.df is None or self.df.empty:
            return

        total = len(self.df)

        x_min = 0
        x_max = total + 5

        self.ax.set_xlim(
            x_min,
            x_max
        )

        self.update_y_range(
            x_min,
            x_max
        )

        if self.volume_ax is not None:
            self.volume_ax.set_xlim(
                x_min,
                x_max
            )

    def set_visible_days(self, days):

        self.visible_days = days

        if self.full_df is not None:
            self.plot_stock(
                self.full_df,
                keep_view=False
            )

    # =========================
    # 滑鼠事件
    # =========================

    def on_click(self, event):

        if event.xdata is None or event.ydata is None:
            return

        # 右鍵按住：準備拖動畫面
        if event.button == MouseButton.RIGHT:
            self.panning = True
            self.pan_start = (event.xdata, event.ydata)
            self.pan_xlim = self.ax.get_xlim()
            self.pan_ylim = self.ax.get_ylim()
            return

        # 沒有選畫線工具就不畫
        if self.tool != "draw_line":
            return
        
        point_counts = 2
        
        if self.line_type == self.line_types[2]:
            point_counts = 1

        point = self.get_point(event)

        self.points.append(point)

        if len(self.points) == point_counts:

            p1 = self.points[0]
            p2 = self.points[point_counts - 1]

            self.line_id += 1

            self.clear_preview_line()

            self.add_line(
                self.line_id,
                self.line_type,
                p1,
                p2,
                self.color
            )

            self.points.clear()

    def on_pick(self, event):

        if self.tool != "erase":
            return

        mouse_event = event.mouseevent

        # 右鍵刪除線
        if mouse_event.button == 1:

            line_artist = event.artist

            line_artist.remove()

            self.drawing_manager.del_line(
                self.stock_id,
                line_artist.db_id
            )

            self.lines = [
                line for line in self.lines
                if line.line != line_artist.db_id
            ]

            self.canvas.draw()

    def on_motion(self, event):

        if event.xdata is None or event.ydata is None:
            return
        
        if self.tool == "draw_line" and len(self.points) == 1:

            p1 = self.points[0]
            p2 = self.get_point(event)

            self.update_preview_line(p1, p2)

        # =========================
        # 右鍵拖動畫面
        # =========================
        if self.panning:

            dx = event.xdata - self.pan_start[0]
            dy = event.ydata - self.pan_start[1]

            x_min, x_max = self.pan_xlim
            y_min, y_max = self.pan_ylim

            # =========================
            # 新X範圍
            # =========================

            new_x_min = x_min - dx
            new_x_max = x_max - dx

            # =========================
            # 邊界限制
            # =========================

            left_boundary = -5

            right_boundary = len(self.df) + 20

            # 左邊超界
            if new_x_min < left_boundary:

                offset = left_boundary - new_x_min

                new_x_min += offset
                new_x_max += offset

            # 右邊超界
            if new_x_max > right_boundary:

                offset = new_x_max - right_boundary

                new_x_min -= offset
                new_x_max -= offset

            # =========================
            # 設定X軸
            # =========================

            self.ax.set_xlim(
                new_x_min,
                new_x_max
            )

            # =========================
            # Y軸平移
            # =========================

            self.update_y_range(
                new_x_min,
                new_x_max
            )

            self.canvas.draw_idle()

            return

        # =========================
        # 拖曳線
        # =========================
        if self.dragging_line is not None:

            xdata = self.dragging_line.get_xdata()
            ydata = self.dragging_line.get_ydata()

            dx = event.xdata - xdata[0]
            dy = event.ydata - ydata[0]

            new_x = [x + dx for x in xdata]
            new_y = [y + dy for y in ydata]

            self.dragging_line.set_xdata(new_x)
            self.dragging_line.set_ydata(new_y)

            self.canvas.draw_idle()
            return

        # hover 資訊
        self.update_hover_info(event)

    def on_release(self, event):

        self.panning = False
        self.pan_start = None
        self.pan_xlim = None
        self.pan_ylim = None

        self.dragging_line = None

    def on_scroll(self, event):

        if event.xdata is None:
            return

        cur_xlim = self.ax.get_xlim()

        x_min, x_max = cur_xlim

        x_range = x_max - x_min

        mouse_x = event.xdata

        # 放大
        if event.button == "up":

            scale_factor = 0.8

        # 縮小
        elif event.button == "down":

            scale_factor = 1.25

        else:
            return

        new_range = x_range * scale_factor

        left_ratio = (
            mouse_x - x_min
        ) / x_range

        right_ratio = (
            x_max - mouse_x
        ) / x_range

        new_x_min = (
            mouse_x
            - new_range * left_ratio
        )

        new_x_max = (
            mouse_x
            + new_range * right_ratio
        )

        # 不要超出太多
        new_x_min = max(
            new_x_min,
            -5
        )

        new_x_max = min(
            new_x_max,
            len(self.df) + 5
        )

        # 最小縮放限制
        if new_x_max - new_x_min < 10:
            return

        self.ax.set_xlim(
            new_x_min,
            new_x_max
        )

        # 自動調整Y軸
        self.update_y_range(
            new_x_min,
            new_x_max
        )

        self.canvas.draw_idle()

    def update_y_range(self, x_min, x_max):

        if self.df is None or self.df.empty:
            return

        start = int(max(0, x_min))
        end = int(min(len(self.df), x_max))

        visible_df = self.df.iloc[start:end]

        if visible_df.empty:
            return

        low = visible_df["Low"].min()
        high = visible_df["High"].max()

        padding = (high - low) * 0.1

        if padding == 0:
            padding = high * 0.01

        self.ax.set_ylim(
            low - padding,
            high + padding
        )

    def update_hover_info(self, event):

        if self.df is None:
            return

        if not hasattr(self, "vline") or not hasattr(self, "hline") or not hasattr(self, "hover_text"):
            return

        idx = round(event.xdata)

        if idx < 0 or idx >= len(self.df):
            self.vline.set_visible(False)
            self.hline.set_visible(False)
            self.hover_text.set_text("")
            self.canvas.draw_idle()
            return

        row = self.df.iloc[idx]

        date = self.df.index[idx]

        open_price = row["Open"]
        high_price = row["High"]
        low_price = row["Low"]
        close_price = row["Close"]

        # 更新十字線
        self.vline.set_xdata([idx, idx])
        self.hline.set_ydata([event.ydata, event.ydata])

        self.vline.set_visible(True)
        self.hline.set_visible(True)

        # 顯示資訊
        text = (
            f"日期：{date.date()}\n"
            f"開：{open_price:.2f}\n"
            f"高：{high_price:.2f}\n"
            f"低：{low_price:.2f}\n"
            f"收：{close_price:.2f}\n"
            f"游標價：{event.ydata:.2f}"
        )

        self.hover_text.set_text(text)

        self.canvas.draw_idle()

    def update_preview_line(self, p1, p2):

        self.clear_preview_line()

        temp_line = dl.DrawingLine(
            -1,
            self.line_type,
            p1,
            p2,
            self.color
        )

        if self.line_type == "Line segment":

            x = [temp_line.p1[0], temp_line.p2[0]]
            y = [temp_line.p1[1], temp_line.p2[1]]

        elif self.line_type == "Straight line":

            x1, y1 = temp_line.p1
            x2, y2 = temp_line.p2

            if x2 == x1:
                return

            slope = (y2 - y1) / (x2 - x1)

            x_start = 0
            x_end = len(self.df) - 1

            y_start = y1 + slope * (x_start - x1)
            y_end = y1 + slope * (x_end - x1)

            x = [x_start, x_end]
            y = [y_start, y_end]

        elif self.line_type == "Horizontal line":

            x1, y1 = temp_line.p1

            x = [0, len(self.df) - 1]
            y = [y1, y1]

        self.preview_artist, = self.ax.plot(
            x,
            y,
            color=self.color,
            linewidth=2,
            linestyle="--",
            alpha=0.6
        )

        self.canvas.draw_idle()

    # =========================
    # 吸附功能
    # =========================

    def get_point(self, event):

        x = event.xdata
        y = event.ydata

        if not self.auto_snap:
            return (x, y)

        idx = round(x)

        if idx < 0:
            idx = 0

        if idx >= len(self.df):
            idx = len(self.df) - 1

        row = self.df.iloc[idx]

        candidates = {
            "Open": row["Open"],
            "High": row["High"],
            "Low": row["Low"],
            "Close": row["Close"]
        }

        nearest_name = None
        nearest_price = None
        nearest_distance = float("inf")

        for name, price in candidates.items():

            distance = abs(y - price)

            if distance < nearest_distance:
                nearest_distance = distance
                nearest_name = name
                nearest_price = price

        # 用價格比例判斷距離，避免高價股和低價股尺度不同
        allowed_distance = nearest_price * self.snap_threshold

        if nearest_distance <= allowed_distance:
            print(f"Snap to {nearest_name}: {nearest_price}")
            return (idx, nearest_price)

        return (x, y)
    
    def get_visible_df(self, df):

        if df is None or df.empty:
            return df

        last_date = df.index[-1]

        start_date = last_date - pd.Timedelta(
            days=self.visible_days
        )

        visible_df = df[df.index >= start_date]

        return visible_df

    # =========================
    # 新增與繪製線
    # =========================

    def add_line(self, line_id, line_type, p1, p2, color):

        drawing_line = dl.DrawingLine(
            line_id,
            line_type,
            p1,
            p2,
            color
        )

        self.lines.append(drawing_line)

        self.drawing_manager.save_line(
            self.stock_id,
            drawing_line
        )

        self.draw_drawing_line(drawing_line)


    def draw_drawing_line(self, drawing_line):

        if drawing_line.type == "Line segment":
            self.draw_line_segment(drawing_line)

        elif drawing_line.type == "Straight line":
            self.draw_straight_line(drawing_line)

        elif drawing_line.type == "Horizontal line":
            self.draw_horizontal_line(drawing_line)

    def draw_line_segment(self, drawing_line):

        x = [
            drawing_line.p1[0],
            drawing_line.p2[0]
        ]

        y = [
            drawing_line.p1[1],
            drawing_line.p2[1]
        ]

        line_artist, = self.ax.plot(
            x,
            y,
            color=drawing_line.color,
            linewidth=2,
            picker=5
        )

        line_artist.db_id = drawing_line.line

        self.canvas.draw()

    def draw_straight_line(self, drawing_line):

        x1, y1 = drawing_line.p1
        x2, y2 = drawing_line.p2

        if x2 == x1:
            return

        slope = (y2 - y1) / (x2 - x1)

        x_start = 0
        x_end = len(self.df) - 1

        y_start = y1 + slope * (x_start - x1)
        y_end = y1 + slope * (x_end - x1)

        line_artist, = self.ax.plot(
            [x_start, x_end],
            [y_start, y_end],
            color=drawing_line.color,
            linewidth=2,
            picker=5
        )

        line_artist.db_id = drawing_line.line

        self.canvas.draw()

    def draw_horizontal_line(self, drawing_line):

        x1, y1 = drawing_line.p1

        x_start = 0
        x_end = len(self.df) - 1

        line_artist, = self.ax.plot(
            [x_start, x_end],
            [y1, y1],
            color=drawing_line.color,
            linewidth=2,
            picker=5
        )

        line_artist.db_id = drawing_line.line

        self.canvas.draw()

    def redraw_lines(self):

        for drawing_line in self.lines:
            self.draw_drawing_line(drawing_line)

    

    def clear_preview_line(self):

        if self.preview_artist is not None:
            self.preview_artist.remove()
            self.preview_artist = None
            self.canvas.draw_idle()

    def create_hover_tools(self):

        self.vline = self.ax.axvline(
            x=0,
            color="gray",
            linewidth=1,
            linestyle="--",
            visible=False
        )

        self.hline = self.ax.axhline(
            y=0,
            color="gray",
            linewidth=1,
            linestyle="--",
            visible=False
        )

        self.hover_text = self.ax.text(
            0.02,
            0.95,
            "",
            transform=self.ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(
                boxstyle="round",
                facecolor="white",
                alpha=0.8
            )
        )