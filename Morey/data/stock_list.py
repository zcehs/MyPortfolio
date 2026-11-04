import sqlite3


class WatchListManager:

    def __init__(self, db_name="watchlist.db"):

        self.conn = sqlite3.connect(db_name)

        self.create_table()


    def create_table(self):

        cursor = self.conn.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS watchlist (
            stock_id TEXT PRIMARY KEY
        )
        """)

        self.conn.commit()

    def add_stock(self, stock_id):

        cursor = self.conn.cursor()

        cursor.execute("""
        INSERT OR IGNORE INTO watchlist(stock_id)
        VALUES(?)
        """, (stock_id,))

        self.conn.commit()

    def remove_stock(self, stock_id):

        cursor = self.conn.cursor()

        cursor.execute("""
        DELETE FROM watchlist
        WHERE stock_id = ?
        """, (stock_id,))

        self.conn.commit()

    def get_all_stocks(self):

        cursor = self.conn.cursor()

        cursor.execute("""
        SELECT stock_id FROM watchlist
        """)

        rows = cursor.fetchall()

        return [row[0] for row in rows]
        