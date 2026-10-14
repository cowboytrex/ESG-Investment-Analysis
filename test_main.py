
import sqlite3
import pytest
from fastapi.testclient import TestClient
import main

client = TestClient(main.app)


@pytest.fixture(autouse=True)
def test_database(tmp_path, monkeypatch):
    # Create a temporary database for each test
    db_path = tmp_path / "test_stocks.db"

    conn = sqlite3.connect(db_path)

    conn.execute("""
        CREATE TABLE stock_prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            date TEXT NOT NULL,
            closing_price REAL NOT NULL,
            UNIQUE(ticker, date)
        )
    """)

    # Add predictable sample prices
    banks = [
        "RY.TO", "TD.TO", "BNS.TO",
        "BMO.TO", "CM.TO", "NA.TO"
    ]

    for ticker in banks:
        for day in range(1, 6):
            conn.execute("""
                INSERT INTO stock_prices
                (ticker, date, closing_price)
                VALUES (?, ?, ?)
            """, (
                ticker,
                f"2026-01-{day:02d}",
                100.0 + day
            ))

    conn.commit()
    conn.close()

    # Redirect API database queries to test database
    monkeypatch.setattr(main, "DATABASE", str(db_path))

    yield


def test_home():
    response = client.get("/")
    assert response.status_code == 200


def test_get_banks():
    response = client.get("/banks")
    assert response.status_code == 200
    assert len(response.json()["banks"]) == 6


def test_get_stock_prices():
    response = client.get("/stocks/RY.TO?limit=5")
    assert response.status_code == 200
    assert len(response.json()["prices"]) == 5


def test_invalid_stock():
    response = client.get("/stocks/INVALID")
    assert response.status_code == 404


def test_invalid_limit():
    response = client.get("/stocks/RY.TO?limit=2000")
    assert response.status_code == 400


def test_compare_banks():
    response = client.get(
        "/compare?bank1=RY.TO&bank2=TD.TO"
    )

    assert response.status_code == 200

    data = response.json()
    assert data["better_performer"] == "Tie"
