"""
Create the SQLite database and apply the third-normal-form schema.

Safe to run repeatedly: every statement in schema.sql is IF NOT EXISTS, so
this creates what is missing and leaves existing data alone.

Usage:
    python scripts/init_db.py
    python scripts/init_db.py --status
"""

from __future__ import annotations

import argparse

from wildlife_monitor.db import DB_PATH, init_db, table_counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", action="store_true",
                        help="only report what is already in the database")
    args = parser.parse_args()

    if not args.status:
        path = init_db()
        print(f"[OK] Schema applied -> {path}")
    elif not DB_PATH.exists():
        print(f"[INFO] No database yet at {DB_PATH}. "
              f"Run 'python scripts/init_db.py' to create it.")
        return

    counts = table_counts()
    if not counts:
        print("[INFO] Database is empty.")
        return

    print(f"\n{'table':<22}{'rows':>10}")
    print("-" * 32)
    for table, count in counts.items():
        print(f"{table:<22}{count:>10,}")


if __name__ == "__main__":
    main()
