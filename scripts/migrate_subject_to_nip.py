from __future__ import annotations

import argparse
import json
import re
import shutil
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

SUBJECT_TABLES = (
    "enrollment_sessions",
    "biometric_templates",
    "verification_sessions",
    "biometric_audit_logs",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Migrate one biometric subject from an employee ID to an 18-digit NIP."
    )
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--old-subject", required=True)
    parser.add_argument("--nip", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    database = args.database.resolve()
    normalized_nip = re.sub(r"\D", "", args.nip)
    if not re.fullmatch(r"employee-[0-9]+", args.old_subject):
        raise SystemExit("--old-subject must use the legacy employee-<id> format")
    if len(normalized_nip) != 18:
        raise SystemExit("--nip must contain exactly 18 digits")
    if not database.is_file():
        raise SystemExit(f"Database not found: {database}")

    new_subject = f"nip-{normalized_nip}"
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    backup = database.with_name(f"{database.name}.backup-before-nip-{timestamp}")
    shutil.copy2(database, backup)

    connection = sqlite3.connect(database)
    try:
        connection.execute("PRAGMA foreign_keys = OFF")
        connection.execute("BEGIN IMMEDIATE")
        old_profile = connection.execute(
            "SELECT status, active_template_id FROM biometric_profiles WHERE subject_id = ?",
            (args.old_subject,),
        ).fetchone()
        if old_profile is None:
            raise RuntimeError(f"Legacy subject not found: {args.old_subject}")
        if connection.execute(
            "SELECT 1 FROM biometric_profiles WHERE subject_id = ?", (new_subject,)
        ).fetchone():
            raise RuntimeError(f"Target subject already exists: {new_subject}")

        updates: dict[str, int] = {}
        for table in SUBJECT_TABLES:
            cursor = connection.execute(
                f"UPDATE {table} SET subject_id = ? WHERE subject_id = ?",  # noqa: S608
                (new_subject, args.old_subject),
            )
            updates[table] = cursor.rowcount

        profile_cursor = connection.execute(
            "UPDATE biometric_profiles SET subject_id = ? WHERE subject_id = ?",
            (new_subject, args.old_subject),
        )
        if profile_cursor.rowcount != 1:
            raise RuntimeError("Expected exactly one biometric profile to be migrated")
        updates["biometric_profiles"] = profile_cursor.rowcount
        connection.commit()

        print(
            json.dumps(
                {
                    "old_subject": args.old_subject,
                    "new_subject": new_subject,
                    "profile_status": old_profile[0],
                    "active_template_id": old_profile[1],
                    "updated_rows": updates,
                    "backup": str(backup),
                },
                indent=2,
            )
        )
    except Exception:
        connection.rollback()
        backup.replace(database)
        raise
    finally:
        connection.close()


if __name__ == "__main__":
    main()
