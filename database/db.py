from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

DATABASE_URL = "sqlite:///./salary_predictions.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id       = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    password = Column(String)
    role     = Column(String, default="user")

class Prediction(Base):
    __tablename__ = "predictions"
    id          = Column(Integer, primary_key=True, index=True)
    username    = Column(String, nullable=True)
    role        = Column(String)
    industry    = Column(String)
    location    = Column(String)
    education   = Column(String)
    experience  = Column(Integer)
    num_skills  = Column(Integer)
    predicted_salary = Column(Float)
    created_at  = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# Migration: check if username column exists in predictions, if not, add it
try:
    with engine.connect() as conn:
        conn.execute("SELECT username FROM predictions LIMIT 1")
except Exception:
    try:
        with engine.connect() as conn:
            conn.execute("ALTER TABLE predictions ADD COLUMN username VARCHAR")
    except Exception as e:
        print(f"Migration error (username column): {e}")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

