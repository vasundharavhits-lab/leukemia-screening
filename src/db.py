"""Scan log and users. SQLite by default; set DATABASE_URL to use SQL Server / PostgreSQL.

SQL Server example (PowerShell):
  $env:DATABASE_URL = "mssql+pyodbc://USER:PASS@SERVER/DBNAME?driver=ODBC+Driver+17+for+SQL+Server"
"""
import datetime
import os

import pandas as pd
from sqlalchemy import (Column, DateTime, Float, Integer, MetaData, String, Table,
                        create_engine, inspect, select, text, update)

DB_URL = os.getenv("DATABASE_URL", "sqlite:///scans.db")
CSV_EXPORT = os.getenv("CSV_EXPORT", "exports/scans.csv")   # lets Power BI Desktop read the data on day one

engine = create_engine(DB_URL)
meta = MetaData()

scans = Table(
    "scans", meta,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("scanned_at", DateTime, nullable=False),
    Column("patient_id", String(64)),
    Column("age", Integer),
    Column("gender", String(16)),
    Column("lab", String(128)),
    Column("image_name", String(256)),
    Column("prob_leukemia", Float),
    Column("threshold", Float),
    Column("prediction", String(32)),
    Column("uncertain", Integer),
    Column("nucleus_area", Float),
    Column("nucleus_circularity", Float),
    Column("nucleus_solidity", Float),
    Column("nc_ratio", Float),
    Column("review_status", String(32)),
    Column("scanned_by", String(64)),      # username who ran the scan
    Column("reviewed_by", String(64)),     # username who saved the review decision
)

users = Table(
    "users", meta,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("username", String(64), unique=True, nullable=False),
    Column("password_hash", String(256), nullable=False),
    Column("created_at", DateTime, nullable=False),
    Column("active", Integer, nullable=False),
)


def _add_missing_columns():
    """An older scans table (from before logins) gets the new columns added in place."""
    have = {c["name"] for c in inspect(engine).get_columns("scans")}
    for col in scans.columns:
        if col.name not in have:
            coltype = col.type.compile(engine.dialect)
            with engine.begin() as con:
                con.execute(text(f"ALTER TABLE scans ADD COLUMN {col.name} {coltype}"))


def init_db():
    meta.create_all(engine)
    _add_missing_columns()


def load_scans() -> pd.DataFrame:
    with engine.connect() as con:
        return pd.read_sql(select(scans).order_by(scans.c.id), con)


def export_csv():
    try:
        os.makedirs(os.path.dirname(CSV_EXPORT) or ".", exist_ok=True)
        load_scans().to_csv(CSV_EXPORT, index=False)
    except Exception:
        pass                      # the CSV is a convenience, never block a scan on it


def insert_scan(**r) -> int:
    values = {
        "scanned_at": datetime.datetime.now(),
        "patient_id": r["patient_id"], "age": r["age"], "gender": r["gender"], "lab": r["lab"],
        "image_name": r.get("image_name"),
        "prob_leukemia": r["prob_leukemia"], "threshold": r["threshold"],
        "prediction": r["prediction"], "uncertain": int(r["uncertain"]),
        "nucleus_area": r.get("nucleus_area"), "nucleus_circularity": r.get("nucleus_circularity"),
        "nucleus_solidity": r.get("nucleus_solidity"), "nc_ratio": r.get("nc_ratio"),
        "review_status": "Pending",
        "scanned_by": r.get("scanned_by"),
    }
    with engine.begin() as con:
        res = con.execute(scans.insert().values(**values))
        scan_id = int(res.inserted_primary_key[0])
    export_csv()
    return scan_id


def update_review(scan_id: int, status: str, reviewer: str = None):
    with engine.begin() as con:
        con.execute(update(scans).where(scans.c.id == scan_id)
                    .values(review_status=status, reviewed_by=reviewer))
    export_csv()
