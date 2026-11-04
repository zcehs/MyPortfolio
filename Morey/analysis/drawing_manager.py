import sqlite3

import analysis.drawing_line as dl


class DrawingManager:

    def __init__(self):

        self.conn = sqlite3.connect(
            "drawings.db"
        )

        self.create_table()

    def create_table(self):

        cursor = self.conn.cursor()

        cursor.execute("""

        CREATE TABLE IF NOT EXISTS chart_lines (
            stock_id TEXT,
            line INTEGER,
            type TEXT,
            x1 REAL,
            y1 REAL,
            x2 REAL,
            y2 REAL,
            color TEXT,
            PRIMARY KEY (stock_id, line)
        )

        """)

        self.conn.commit()

    def save_line(self, stock_id, drawing_line):

        cursor = self.conn.cursor()

        cursor.execute("""

        INSERT INTO chart_lines (

            stock_id,
                       
            line,
                       
            type,

            x1, y1,

            x2, y2,
                       
            color

        )

        VALUES (?, ?, ?, ?, ?, ?, ?, ?)

        """, (

            stock_id,

            drawing_line.line,

            drawing_line.type,

            drawing_line.p1[0],
            drawing_line.p1[1],

            drawing_line.p2[0],
            drawing_line.p2[1],

            drawing_line.color
        ))

        self.conn.commit()

    def del_line(self, stock_id, line):

        cursor = self.conn.cursor()

        cursor.execute("""
        DELETE FROM chart_lines
        WHERE stock_id = ?
        AND line = ?
        """, (stock_id, line))

        self.conn.commit()


    def load_lines(self, stock_id):

        cursor = self.conn.cursor()

        cursor.execute("""
        SELECT line, type, x1, y1, x2, y2, color
        FROM chart_lines
        WHERE stock_id = ?
        ORDER BY line
        """, (stock_id,))

        rows = cursor.fetchall()

        lines = []

        for row in rows:
            line, type_, x1, y1, x2, y2, color = row

            drawing_line = dl.DrawingLine(
                line=line,
                type=type_,
                p1=(x1, y1),
                p2=(x2, y2),
                color=color
            )

            lines.append(drawing_line)

        return lines
    
    def clear_all(self, stock_id):
        cursor = self.conn.cursor()

        cursor.execute("""
        DELETE FROM chart_lines
        WHERE stock_id = ?
        """, (stock_id,))

        self.conn.commit()

    def get_next_line_id(self, stock_id):

        cursor = self.conn.cursor()

        cursor.execute("""
        SELECT COALESCE(MAX(line), -1) + 1
        FROM chart_lines
        WHERE stock_id = ?
        """, (stock_id,))

        return cursor.fetchone()[0]