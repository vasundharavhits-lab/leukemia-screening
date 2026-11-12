import shutil, datetime
from sqlalchemy import delete, or_, select
from src import db

cond = or_(db.scans.c.patient_id.like("T-%"), db.scans.c.patient_id.like("TEST-%"))

with db.engine.connect() as con:
    rows = con.execute(select(db.scans.c.id, db.scans.c.patient_id, db.scans.c.prediction).where(cond)).fetchall()
print(f"Test scans to delete: {len(rows)}")
for r in rows:
    print(r)
if not rows:
    raise SystemExit("Nothing to delete.")

path = db.DB_URL.replace("sqlite:///", "", 1)
backup = path + "." + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".bak"
shutil.copy(path, backup)
print("Backup saved:", backup)

if input("Type DELETE to remove only these rows: ").strip() != "DELETE":
    raise SystemExit("Cancelled. Nothing was deleted.")

with db.engine.begin() as con:
    con.execute(delete(db.scans).where(cond))
db.export_csv()
print("Done. Reload the app page (F5).")
