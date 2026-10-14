
import sqlite3
from fastapi import FastAPI, HTTPException
from database import DATABASE

# Create the FastAPI application
app = FastAPI(title="Canadian Banking Financial API")


# Home endpoint
@app.get("/")
def home():
    return {"message": "Canadian Banking API is running!"}


# Return all banks stored in SQL
@app.get("/banks")
def get_banks():
    conn = sqlite3.connect(DATABASE)

    try:
        rows = conn.execute(
            "SELECT DISTINCT ticker FROM stock_prices ORDER BY ticker"
        ).fetchall()
    finally:
        conn.close()

    return {"banks": [row[0] for row in rows]}


# Return historical prices for a specific bank
@app.get("/stocks/{ticker}")
def get_stock_prices(ticker: str, limit: int = 30):

    if limit < 1 or limit > 1000:
        raise HTTPException(
            status_code=400,
            detail="Limit must be between 1 and 1000"
        )

    conn = sqlite3.connect(DATABASE)

    try:
        rows = conn.execute("""
            SELECT date, closing_price
            FROM stock_prices
            WHERE ticker = ?
            ORDER BY date DESC
            LIMIT ?
        """, (ticker.upper(), limit)).fetchall()
    finally:
        conn.close()

    if not rows:
        raise HTTPException(
            status_code=404,
            detail="Stock ticker not found"
        )

    return {
        "ticker": ticker.upper(),
        "prices": [
            {"date": row[0], "closing_price": row[1]}
            for row in rows
        ]
    }

@app.get("/compare")
def compare_banks(bank1: str, bank2: str):

    conn = sqlite3.connect(DATABASE)

    try:
        def calculate_return(ticker):
            rows = conn.execute("""
                SELECT closing_price
                FROM stock_prices
                WHERE ticker = ?
                ORDER BY date ASC
            """, (ticker.upper(),)).fetchall()

            if len(rows) < 2:
                raise HTTPException(
                    status_code=404,
                    detail=f"Not enough data for {ticker}"
                )

            first_price = rows[0][0]
            last_price = rows[-1][0]

            return round(
                ((last_price - first_price) / first_price) * 100,
                2
            )

        return1 = calculate_return(bank1)
        return2 = calculate_return(bank2)

    finally:
        conn.close()

    return {
        "bank1": bank1.upper(),
        "bank1_return_percent": return1,
        "bank2": bank2.upper(),
        "bank2_return_percent": return2,
        "better_performer": (
            bank1.upper() if return1 > return2
            else bank2.upper() if return2 > return1
            else "Tie"
        )
    }
