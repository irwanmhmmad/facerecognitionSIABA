from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class Base(DeclarativeBase):
    pass


class Database:
    def __init__(self, url: str) -> None:
        connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
        self.engine = create_engine(url, pool_pre_ping=True, connect_args=connect_args)
        self.session_factory = sessionmaker(bind=self.engine, expire_on_commit=False, class_=Session)

    def create_all(self) -> None:
        Base.metadata.create_all(self.engine)
        inspector = inspect(self.engine)
        columns = {column["name"] for column in inspector.get_columns("enrollment_sessions")}
        if "struktur_org" not in columns:
            with self.engine.begin() as connection:
                connection.execute(
                    text("ALTER TABLE enrollment_sessions ADD COLUMN struktur_org VARCHAR(1000)")
                )

        inspector = inspect(self.engine)
        indexes = {index["name"] for index in inspector.get_indexes("enrollment_sessions")}
        if "ix_enrollment_sessions_struktur_org" not in indexes:
            with self.engine.begin() as connection:
                connection.execute(
                    text(
                        "CREATE INDEX ix_enrollment_sessions_struktur_org "
                        "ON enrollment_sessions (struktur_org)"
                    )
                )

        inspector = inspect(self.engine)
        profile_columns = {column["name"] for column in inspector.get_columns("biometric_profiles")}
        if "template_version" not in profile_columns:
            with self.engine.begin() as connection:
                connection.execute(
                    text(
                        "ALTER TABLE biometric_profiles "
                        "ADD COLUMN template_version INTEGER NOT NULL DEFAULT 0"
                    )
                )
                connection.execute(
                    text(
                        "UPDATE biometric_profiles SET template_version = 1 "
                        "WHERE active_template_id IS NOT NULL"
                    )
                )

        inspector = inspect(self.engine)
        template_columns = {column["name"] for column in inspector.get_columns("biometric_templates")}
        if "template_version" not in template_columns:
            with self.engine.begin() as connection:
                connection.execute(
                    text(
                        "ALTER TABLE biometric_templates "
                        "ADD COLUMN template_version INTEGER NOT NULL DEFAULT 1"
                    )
                )

    def session(self) -> Generator[Session, None, None]:
        db = self.session_factory()
        try:
            yield db
        finally:
            db.close()
