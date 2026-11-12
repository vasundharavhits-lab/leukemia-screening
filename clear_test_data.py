"""Delete ALL scans from the database (use before giving the app to a client).

Safe steps:
  1. shows how many scans exist
  2. makes a backup copy of scans.db (SQLite) before deleting anything
  3. asks you to type DELETE to confirm
  4. deletes the rows and refreshes exports/scans.csv

Run from the project folder (the one that contains app.py and src/):
    python clear_test_data.py
"""
import datetime
import os
import shutil
import sys

from sqlalchemy import delete, func, select, text

from src import db


def main():
    with db.engine.connect() as con:
        n = con.execute(select(func.count()).select_from(db.scans)).scalar()
    print(f"Database: {db.DB_URL}")
    print(f"Scans stored: {n}")
    if n == 0:
        print("Nothing to delete.")
        return

    # backup (SQLite file only; for SQL Server / PostgreSQL use your database's own backup)
    if db.DB_URL.startswith("sqlite:///"):
        path = db.DB_URL.replace("sqlite:///", "", 1)
        if os.path.exists(path):
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup = f"{os.path.splitext(path)[0]}_backup_{stamp}.db"
            shutil.copy(path, backup)
            print(f"Backup saved: {backup}")
    else:
        print("This is not a SQLite database. Make your own backup first.")

    answer = input("Type DELETE to remove ALL scans: ").strip()
    if answer != "DELETE":
        print("Cancelled. Nothing was deleted.")
        return

    with db.engine.begin() as con:
        con.execute(delete(db.scans))
        if db.DB_URL.startswith("sqlite"):
            try:
                con.execute(text("DELETE FROM sqlite_sequence WHERE name = 'scans'"))
            except Exception:
                pass                       # table may not exist; ids restart anyway
    db.export_csv()
    print("Done. All scans deleted. The next scan will be #1.")
    print("Now reload the app page (F5).")


if __name__ == "__main__":
    sys.exit(main())
