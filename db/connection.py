import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import (
    sessionmaker,
    DeclarativeBase
)

load_dotenv()


DATABASE_URL = os.getenv("SUPABASE_DB_URL")

if not DATABASE_URL:
    raise ValueError(
        "SUPABASE_DB_URL is not set."
    )


# ==========================================
# SQLAlchemy Base
# ==========================================

class Base(DeclarativeBase):
    pass


# ==========================================
# Database Engine
# ==========================================

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True
)


# ==========================================
# Database Session
# ==========================================

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False
)


# ==========================================
# Dependency
# ==========================================

def get_db():

    db = SessionLocal()

    try:

        yield db

    finally:

        db.close()