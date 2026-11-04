from PySide6.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QPushButton,
    QComboBox,
    QCheckBox,
    QSpinBox,
    QLabel
)


class DrawingToolBar(QWidget):

    def __init__(self, chart_widget):

        super().__init__()

        self.chart_widget = chart_widget

        layout = QHBoxLayout()

        self.cursor_button = QPushButton("游標")
        self.erase_button = QPushButton("橡皮擦")
        self.line_button = QPushButton("畫線")

        self.line_type_box = QComboBox()
        self.line_type_box.addItems([
            "Line segment",
            "Straight line",
            "Horizontal line"
        ])

        self.color_box = QComboBox()
        self.color_box.addItems([
            "black",
            "red",
            "green",
            "blue",
            "orange",
            "purple",
            "white"
        ])

        #self.snap_check = QCheckBox("吸附")

        """self.snap_target_box = QComboBox()
        self.snap_target_box.addItems([
            "Open",
            "High",
            "Low",
            "Close"
        ])"""

        """self.mav_line = QComboBox()
        self.mav_line.addItems([
            
        ])"""

        self.ma1_check = QCheckBox("MA1")
        self.ma1_check.setChecked(True)

        self.ma1_spin = QSpinBox()
        self.ma1_spin.setRange(1, 300)
        self.ma1_spin.setValue(5)


        self.ma2_check = QCheckBox("MA2")
        self.ma2_check.setChecked(True)

        self.ma2_spin = QSpinBox()
        self.ma2_spin.setRange(1, 300)
        self.ma2_spin.setValue(10)


        self.ma3_check = QCheckBox("MA3")
        self.ma3_check.setChecked(True)

        self.ma3_spin = QSpinBox()
        self.ma3_spin.setRange(1, 300)
        self.ma3_spin.setValue(20)

        self.volume_check = QCheckBox("成交量")
        self.volume_check.setChecked(True)


        layout.addWidget(self.cursor_button)
        layout.addWidget(self.erase_button)
        layout.addWidget(self.line_button)
        layout.addWidget(self.line_type_box)
        layout.addWidget(self.color_box)
        #layout.addWidget(self.snap_check)
        #layout.addWidget(self.snap_target_box)
        layout.addWidget(self.ma1_check)
        layout.addWidget(self.ma1_spin)

        layout.addWidget(self.ma2_check)
        layout.addWidget(self.ma2_spin)

        layout.addWidget(self.ma3_check)
        layout.addWidget(self.ma3_spin)
        layout.addWidget(self.volume_check)

        self.setLayout(layout)

        self.cursor_button.clicked.connect(
            self.use_cursor
        )

        self.erase_button.clicked.connect(
            self.use_erase
        )

        self.line_button.clicked.connect(
            self.use_line_tool
        )

        self.line_type_box.currentTextChanged.connect(
            self.chart_widget.set_line_type
        )

        self.color_box.currentTextChanged.connect(
            self.chart_widget.set_color
        )


        self.ma1_check.toggled.connect(self.update_ma_settings)
        self.ma1_spin.valueChanged.connect(self.update_ma_settings)

        self.ma2_check.toggled.connect(self.update_ma_settings)
        self.ma2_spin.valueChanged.connect(self.update_ma_settings)

        self.ma3_check.toggled.connect(self.update_ma_settings)
        self.ma3_spin.valueChanged.connect(self.update_ma_settings)

        self.volume_check.toggled.connect(
            self.chart_widget.set_show_volume
        )

    def use_cursor(self):
        self.chart_widget.set_tool("cursor")

    def use_line_tool(self):
        self.chart_widget.set_tool("draw_line")

    def use_erase(self):
        self.chart_widget.set_tool("erase")

    def update_ma_settings(self):

        ma_settings = [
            {
                "enabled": self.ma1_check.isChecked(),
                "window": self.ma1_spin.value()
            },
            {
                "enabled": self.ma2_check.isChecked(),
                "window": self.ma2_spin.value()
            },
            {
                "enabled": self.ma3_check.isChecked(),
                "window": self.ma3_spin.value()
            },
        ]

        self.chart_widget.set_ma_settings(ma_settings)