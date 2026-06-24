from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from app.database import Base, engine
from app.models import patient, session

# Routes
from app.routes.patient import router as patient_router
from app.routes.speech import router as speech_router

# -----------------------------------
# APP
# -----------------------------------
app = FastAPI(
    title="VaakSuddhi AI Backend",
    version="1.0.0"
)

# -----------------------------------
# CORS
# -----------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------
# CREATE TABLES
# -----------------------------------
# Base.metadata.create_all(bind=engine)
@app.on_event("startup")
def startup():
    Base.metadata.create_all(bind=engine)


def ensure_database_schema():
    statements = [
        "ALTER TABLE patients ADD COLUMN IF NOT EXISTS gender VARCHAR",
        "ALTER TABLE patients ADD COLUMN IF NOT EXISTS diagnosis VARCHAR",
        "ALTER TABLE patients ADD COLUMN IF NOT EXISTS therapist_name VARCHAR",
        "ALTER TABLE patients ADD COLUMN IF NOT EXISTS parent_contact VARCHAR",
        "ALTER TABLE patients ADD COLUMN IF NOT EXISTS is_active BOOLEAN DEFAULT TRUE",
        "ALTER TABLE patients ADD COLUMN IF NOT EXISTS created_at TIMESTAMP",
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS f0_mean FLOAT",
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS mpt FLOAT",
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS jitter FLOAT",
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS shimmer FLOAT",
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS hnr FLOAT",
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS target_word VARCHAR",
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS spoken_word VARCHAR",
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS accuracy INTEGER",
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS feedback VARCHAR",
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS stars INTEGER",
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS audio_file VARCHAR",
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS session_type VARCHAR",
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS trs_score INTEGER",
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS created_at TIMESTAMP",
    ]

    with engine.begin() as connection:
        for statement in statements:
            connection.execute(text(statement))


ensure_database_schema()

# -----------------------------------
# INCLUDE ROUTES
# -----------------------------------
app.include_router(patient_router)
app.include_router(speech_router)
<<<<<<< HEAD
# app.include_router(image_router)
=======
>>>>>>> dd04105877102bd7ee5d7fce179fdfbe4e4e0f43

# -----------------------------------
# ROOT
# -----------------------------------
@app.get("/")
def home():
    return {
        "message": "VaakSuddhi V1 Running 🚀"
    }
