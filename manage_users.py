"""Manage the people who can sign in to the dashboard.

Run from the project folder (the one that contains app.py and src/):
    python manage_users.py add dr.sharma        (asks for the password, typing is hidden)
    python manage_users.py list
    python manage_users.py passwd dr.sharma     (set a new password)
    python manage_users.py delete dr.sharma
"""
import argparse
import getpass
import sys

from src import auth, db


def ask_password():
    p1 = getpass.getpass("Password (at least 8 characters): ")
    p2 = getpass.getpass("Repeat password: ")
    if p1 != p2:
        sys.exit("The two passwords do not match. Nothing was changed.")
    return p1


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("add", "passwd", "delete"):
        sub.add_parser(name).add_argument("username")
    sub.add_parser("list")
    args = ap.parse_args()

    db.init_db()
    try:
        if args.cmd == "add":
            auth.create_user(args.username, ask_password())
            print(f"User '{args.username.strip().lower()}' created.")
        elif args.cmd == "passwd":
            auth.set_password(args.username, ask_password())
            print("Password changed.")
        elif args.cmd == "delete":
            auth.delete_user(args.username)
            print("User deleted.")
        elif args.cmd == "list":
            users = auth.list_users()
            if not users:
                print("No users yet. Add one with:  python manage_users.py add USERNAME")
            for name, created in users:
                print(f"{name}    (created {created:%Y-%m-%d %H:%M})")
    except ValueError as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
