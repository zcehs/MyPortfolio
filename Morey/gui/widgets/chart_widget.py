from PySide6.QtWidgets import QWidget, QVBoxLayout

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backend_bases import MouseButton

import matplotlib
import mplfinance as mpf
import pandas as pd

import analysis.drawing_line as dl


matplotlib.rcParams["font.family"] = "Microsoft JhengHei"
matplotlib.rcParams["axes.unicode_minus"] = False


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

        self.full_df = None      # 完整資料，例如 10 年
        self.df = None           # 實際畫出來的資料，例如近 1 年
        self.ax = None
        self.volume_ax = None

        # =========================
        # 顯示設定
        # =========================

        self.visible_days = 365 * 2

        self.ma_settings = [
            {"enabled": True, "window": 5},
            {"enabled": True, "window": 10},
            {"enabled": True, "window": 20},
        ]

        self.show_volume = True

        # =========================
        # 畫線功能
        # =========================

        self.points = []
        self.lines = []
        self.preview_artist = None

        self.line_id = self.drawing_manager.get_next_line_id(
            self.stock_id
        )

        self.line_types = [
            "Line segment",
            "Straight line",
            "Horizontal line"
        ]

        self.tool = "cursor"  # cursor / draw_line / erase
        self.line_type = "Line segment"
        self.color = "black"

        self.auto_snap = True
        self.snap_threshold = 0.01

        self.dragging_line = None

        saved_lines = self.drawing_manager.load_lines(self.stock_id)

        for line in saved_lines:
            self.lines.append(line)

        # =========================
        # 平移功能
        # =========================

        self.panning = False
        self.pan_start = None
        self.pan_xlim = None
        self.pan_ylim = None

        # =========================
        # 事件
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

    # =========================
    # 主繪圖
    # =========================

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

            self.ax = self.figure.add_subplot(
                gs[:4, 0]
            )

            self.volume_ax = self.figure.add_subplot(
                gs[4, 0],
                sharex=self.ax
            )

        else:

            self.ax = self.figure.add_subplot(
                1,
                1,
                1
            )

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
            style="charles",
            warn_too_much_data=10000,
            update_width_config=dict(
                candle_linewidth=0.6,
                candle_width=0.6,
                volume_width=0.6
            )
        )

        self.hide_x_axis_labels()

        self.create_hover_tools()

        self.redraw_lines()

        if keep_view and old_xlim is not None:

            self.ax.set_xlim(old_xlim)
            self.ax.set_ylim(old_ylim)

            if self.volume_ax is not None:
                self.volume_ax.set_xlim(old_xlim)

        else:

            self.set_default_view()

        self.canvas.draw_idle()

    def get_visible_df(self, df):

        if df is None or df.empty:
            return df

        last_date = df.index[-1]

        start_date = last_date - pd.Timedelta(
            days=self.visible_days
        )

        return df[df.index >= start_date]

    def set_default_view(self):

        if self.df is None or self.df.empty:
            return

        total = len(self.df)

        visible_count = 20  # 約一個月交易日

        x_max = total + 5
        x_min = total - visible_count

        if x_min < 0:
            x_min = 0

        self.ax.set_xlim(
            x_min,
            x_max
        )

        if self.volume_ax is not None:
            self.volume_ax.set_xlim(
                x_min,
                x_max
            )

        self.update_y_range(
            x_min,
            x_max
        )

    def hide_x_axis_labels(self):

        self.ax.tick_params(
            axis="x",
            which="both",
            labelbottom=False
        )

        if self.volume_ax is not None:
            self.volume_ax.tick_params(
                axis="x",
                which="both",
                labelbottom=False
            )

    # =========================
    # 工具列呼叫
    # =========================

    def set_tool(self, tool):

        self.tool = tool
        self.points.clear()
        self.clear_preview_line()

    def set_line_type(self, line_type):

        if line_type in self.line_types:
            self.line_type = line_type
            self.points.clear()
            self.clear_preview_line()

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

    # =========================
    # 滑鼠事件
    # =========================

    def on_click(self, event):

        if event.xdata is None or event.ydata is None:
            return

        if event.button == MouseButton.RIGHT:

            self.panning = True
            self.pan_start = (
                event.xdata,
                event.ydata
            )

            self.pan_xlim = self.ax.get_xlim()
            self.pan_ylim = self.ax.get_ylim()

            return

        if self.tool != "draw_line":
            return

        point_count = 2

        if self.line_type == "Horizontal line":
            point_count = 1

        point = self.get_point(event)

        self.points.append(point)

        if len(self.points) == point_count:

            p1 = self.points[0]
            p2 = self.points[point_count - 1]

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

            self.canvas.draw_idle()

    def on_motion(self, event):

        if event.xdata is None or event.ydata is None:
            return

        if self.panning:

            self.handle_pan(event)
            return

        if self.dragging_line is not None:

            self.handle_drag_line(event)
            return

        if self.tool == "draw_line" and len(self.points) == 1:

            p1 = self.points[0]
            p2 = self.get_point(event)

            self.update_preview_line(
                p1,
                p2
            )

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

        x_min, x_max = self.ax.get_xlim()

        x_range = x_max - x_min

        mouse_x = event.xdata

        if event.button == "up":
            scale_factor = 0.8

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

        new_x_min, new_x_max = self.clamp_x_range(
            new_x_min,
            new_x_max
        )

        if new_x_max - new_x_min < 10:
            return

        self.ax.set_xlim(
            new_x_min,
            new_x_max
        )

        if self.volume_ax is not None:
            self.volume_ax.set_xlim(
                new_x_min,
                new_x_max
            )

        self.update_y_range(
            new_x_min,
            new_x_max
        )

        self.canvas.draw_idle()

    # =========================
    # 平移 / 縮放輔助
    # =========================

    def handle_pan(self, event):

        dx = event.xdata - self.pan_start[0]

        x_min, x_max = self.pan_xlim

        new_x_min = x_min - dx
        new_x_max = x_max - dx

        if new_x_min < 10:
            self.expand_visible_range()
            return

        new_x_min, new_x_max = self.clamp_x_range(
            new_x_min,
            new_x_max
        )

        self.ax.set_xlim(
            new_x_min,
            new_x_max
        )

        if self.volume_ax is not None:
            self.volume_ax.set_xlim(
                new_x_min,
                new_x_max
            )

        self.update_y_range(
            new_x_min,
            new_x_max
        )

        self.canvas.draw_idle()

    def clamp_x_range(self, x_min, x_max):

        left_boundary = -5
        right_boundary = len(self.df) + 5

        width = x_max - x_min

        if width >= right_boundary - left_boundary:
            return left_boundary, right_boundary

        if x_min < left_boundary:
            x_min = left_boundary
            x_max = x_min + width

        if x_max > right_boundary:
            x_max = right_boundary
            x_min = x_max - width

        return x_min, x_max

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

    # =========================
    # Hover
    # =========================

    def create_hover_tools(self):

        self.vline = self.ax.axvline(
            x=0,
            color="gray",
            linewidth=1,
            linestyle="--",
            visible=False,
            zorder=100
        )

        self.hline = self.ax.axhline(
            y=0,
            color="gray",
            linewidth=1,
            linestyle="--",
            visible=False,
            zorder=100
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
            ),
            zorder=101
        )

    def update_hover_info(self, event):

        if self.df is None:
            return

        if not hasattr(self, "vline"):
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

        self.vline.set_xdata([idx, idx])
        self.hline.set_ydata([event.ydata, event.ydata])

        self.vline.set_visible(True)
        self.hline.set_visible(True)

        text = (
            f"日期：{date.date()}\n"
            f"開：{row['Open']:.2f}\n"
            f"高：{row['High']:.2f}\n"
            f"低：{row['Low']:.2f}\n"
            f"收：{row['Close']:.2f}\n"
            f"游標價：{event.ydata:.2f}"
        )

        self.hover_text.set_text(text)

        self.canvas.draw_idle()

    # =========================
    # 吸附
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

        nearest_price = None
        nearest_distance = float("inf")

        for price in candidates.values():

            distance = abs(y - price)

            if distance < nearest_distance:
                nearest_distance = distance
                nearest_price = price

        allowed_distance = nearest_price * self.snap_threshold

        if nearest_distance <= allowed_distance:
            return (idx, nearest_price)

        return (x, y)

    # =========================
    # 畫線
    # =========================

    def add_line(
        self,
        line_id,
        line_type,
        p1,
        p2,
        color
    ):

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

    def draw_horizontal_line(self, drawing_line):

        x1, y1 = drawing_line.p1

        line_artist, = self.ax.plot(
            [0, len(self.df) - 1],
            [y1, y1],
            color=drawing_line.color,
            linewidth=2,
            picker=5
        )

        line_artist.db_id = drawing_line.line

    def redraw_lines(self):

        for drawing_line in self.lines:
            self.draw_drawing_line(drawing_line)

    # =========================
    # 預覽線
    # =========================

    def update_preview_line(self, p1, p2):

        if self.line_type == "Line segment":

            x = [p1[0], p2[0]]
            y = [p1[1], p2[1]]

        elif self.line_type == "Straight line":

            x1, y1 = p1
            x2, y2 = p2

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

            x = [0, len(self.df) - 1]
            y = [p1[1], p1[1]]

        else:
            return

        if self.preview_artist is None:

            self.preview_artist, = self.ax.plot(
                x,
                y,
                color=self.color,
                linewidth=2,
                linestyle="--",
                alpha=0.6
            )

        else:

            self.preview_artist.set_xdata(x)
            self.preview_artist.set_ydata(y)
            self.preview_artist.set_color(self.color)

        self.canvas.draw_idle()

    def clear_preview_line(self):

        if self.preview_artist is not None:
            self.preview_artist.remove()
            self.preview_artist = None
            self.canvas.draw_idle()

    # =========================
    # 拖曳線，暫時保留
    # =========================

    def handle_drag_line(self, event):

        xdata = self.dragging_line.get_xdata()
        ydata = self.dragging_line.get_ydata()

        dx = event.xdata - xdata[0]
        dy = event.ydata - ydata[0]

        new_x = [x + dx for x in xdata]
        new_y = [y + dy for y in ydata]

        self.dragging_line.set_xdata(new_x)
        self.dragging_line.set_ydata(new_y)

        self.canvas.draw_idle()

    def expand_visible_range(self):

        if self.full_df is None:
            return

        max_days = 365 * 10

        if self.visible_days >= max_days:
            return

        self.visible_days = min(
            self.visible_days + 365,
            max_days
        )

        self.plot_stock(
            self.full_df,
            keep_view=False
        )