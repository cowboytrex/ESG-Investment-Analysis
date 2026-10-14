import sqlite3

DATABASE = 'stocks.db'

def create_database():
    """Create the database and the necessary tables."""
    conn = sqlite3.connect(DATABASE)
    

    # Create a table for stocks
    conn.execute("""
        CREATE TABLE IF NOT EXISTS stock_prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            date TEXT NOT NULL,
            closing_price REAL NOT NULL,
            UNIQUE(ticker, date)
        )
    """)

    conn.commit()
    conn.close()

if __name__ == "__main__":
    create_database()
    print("Database created successfully!")
