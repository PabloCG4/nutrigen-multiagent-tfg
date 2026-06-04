from __future__ import annotations

import math
import os
from contextlib import asynccontextmanager
import secrets
import shutil
import psycopg2
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from datetime import date as date_type
from typing import Any, Iterator, Literal, Optional
from urllib.parse import quote, urlencode
from uuid import uuid4
import json
import queue
import threading

import bcrypt
import httpx
from authlib.integrations.httpx_client import OAuth2Client
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from pydantic import BaseModel, ConfigDict, Field, field_validator

# Load environment variables from src/backend/.env as early as possible,
# before importing celery_app (producer) to avoid missing REDIS_URL/CELERY_*.
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

from src.backend import config
from src.backend.allergy_ids import validate_allergies_for_save
from src.backend.database_manager import DatabaseManager
from src.backend.redis_client import get_redis
from src.backend.state_store import (
    consume_oauth_state,
    create_menu_job_hash,
    get_menu_job_hash,
    set_oauth_state,
    update_menu_job_hash,
)
from src.backend.utils import (
    autocomplete_top_matches,
    calculate_dynamic_targets,
    calculate_exercise_burned_calories,
    display_product_name_for_lang,
    get_autocomplete_product_mapping,
    get_nutritionist_agent,
    get_product_macros_mapping,
    get_product_name_mapping,
    normalize_product_text,
    parse_dietary_style,
)
from src.backend.celery_app import celery_app
from src.backend.tasks.vision import process_vision_image_task
from src.common.nutrition_targets import compute_meal_targets_from_daily_summary

_MENU_JOB_TTL_SECONDS = 2 * 60 * 60


def _get_job_or_404(redis_client, job_id: str, user_id: int) -> dict[str, Any]:
    job = get_menu_job_hash(redis_client, job_id)
    if job is None or int(job.get("user_id") or 0) != int(user_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Menu job not found (expired or invalid).",
        )
    return job

def _output_language_from_profile(profile: dict | None) -> Literal["es", "en"]:
    """Maps stored profile language to pipeline output language (default Spanish)."""
    if not profile:
        return "es"
    lang = profile.get("language")
    return lang if lang in ("es", "en") else "es"


@asynccontextmanager
async def _app_lifespan(_app: FastAPI):
    """Run DB migrations once at startup (not per HTTP request) to avoid DDL/DML deadlocks."""
    db_url = os.environ.get("DATABASE_URL")
    if db_url:
        DatabaseManager.ensure_schema_once(db_url)
    # Initialize Redis client/pool once (supports multi-worker by externalizing state).
    try:
        _app.state.redis = get_redis()
        _app.state.redis.ping()
    except Exception:
        _app.state.redis = None
    yield


app = FastAPI(
    title="AI Culinary Assistant Backend",
    version="0.1.0",
    lifespan=_app_lifespan,
)

FRONTEND_URL_VALUE = (os.getenv("FRONTEND_URL") or "").strip()
if FRONTEND_URL_VALUE == "":
    FRONTEND_URL_ORIGINS = ["http://localhost:5173"]
else:
    FRONTEND_URL_ORIGINS = [FRONTEND_URL_VALUE]

# middleware is used to allow the frontend to access the backend. CORS means Cross-Origin Resource Sharing.
app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_URL_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SECRET_KEY = os.getenv("SECRET_KEY")


ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7

GOOGLE_AUTHORIZATION_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"

def _resolve_google_redirect_uri(request: Request) -> str:
    backend_url_value = (os.getenv("BACKEND_URL") or "").strip().rstrip("/")
    if backend_url_value != "":
        return f"{backend_url_value}/api/auth/callback"
    return f"{str(request.base_url).rstrip('/')}/api/auth/callback"

FRONTEND_BASE_URL = FRONTEND_URL_VALUE if FRONTEND_URL_VALUE != "" else "http://localhost:5173"

_OAUTH_STATE_TTL_SECONDS = 600.0

# OAuth2PasswordBearer is a FastAPI dependency that validates the JWT token.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/login")

def get_db():
    """
    FastAPI dependency that yields a safe DB session. "yields" means that the function will return a generator, and a generator is a function that returns an iterator, and an iterator is an object that can be iterated over, so it will return a database manager object.
    The context manager guarantees connection close on request completion.
    """
    with DatabaseManager(db_path=str(config.APP_DB_PATH)) as db:
        yield db

# StrictBaseModel is a Pydantic model that is used to validate the data that is sent to the backend.
class StrictBaseModel(BaseModel):
    # extra="forbid" means that the model will raise an error if any extra fields are sent to the backend.
    model_config = ConfigDict(extra="forbid")

class UserRegister(BaseModel):
    email: str = Field(..., description="User's primary email address used for login and notifications")
    password: str = Field(..., min_length=8, description="Plaintext password. Must be at least 8 characters long.")

    @field_validator("email")
    @classmethod
    def validate_commercial_email_domain(cls, value: str) -> str:
        """
        Validates that the email belongs to an authorized commercial email provider
        supporting both global (.com) and regional (.es) extensions.
        """
        email_clean = value.strip().lower()
        
        if "@" not in email_clean:
            raise ValueError("Invalid email format: missing '@' character.")
            
        _, domain = email_clean.split("@", 1)
        
        # Tuple of allowed commercial providers and extensions
        ALLOWED_DOMAINS = (
            "gmail.com",
            "hotmail.com",
            "hotmail.es",
            "outlook.com",
            "outlook.es",
            "yahoo.com",
            "yahoo.es"
        )
        
        if not domain.endswith(ALLOWED_DOMAINS):
            raise ValueError(
                "Registration restricted to standard commercial email providers "
                "(Gmail, Hotmail, Outlook, Yahoo) with valid extensions."
            )
            
        return email_clean


class Token(StrictBaseModel):
    access_token: str
    token_type: str


class UpdateEmailRequest(StrictBaseModel):
    new_email: str = Field(min_length=3)
    current_password: str = Field(default="", max_length=200)


class ChangePasswordRequest(StrictBaseModel):
    current_password: str = Field(default="", max_length=200)
    new_password: str = Field(min_length=8, max_length=100)


class DeleteAccountRequest(StrictBaseModel):
    current_password: str = Field(default="", max_length=200)


class PreferencesUpdateRequest(StrictBaseModel):
    language: Literal["es", "en"] = Field(description="UI language preference.")
    units: Literal["metric", "imperial"] = Field(description="Units system preference.")


class ProfileUpdate(StrictBaseModel):
    name: str = Field(min_length=1)
    location: str = Field(min_length=1)
    age: int = Field(ge=0, le=130)
    gender: str = Field(min_length=1)
    weight: float = Field(gt=0)
    height: float = Field(gt=0)
    activity_level: str = Field(min_length=1)
    allergies: list[str] = Field(default_factory=list)
    physical_goal: str = Field(min_length=1)
    weight_goal_rate: float = Field(
        ge=-1.5,
        le=1.5,
        description="Kg/week to lose (negative) or gain (positive).",
    )
    # Number of user meals per day configuration constraint
    meals_per_day: int = Field(
        default=3,
        ge=1,
        le=6,
        description="Configured number of target meals per day.",
    )

    @field_validator("allergies")
    @classmethod
    def _validate_allergies(cls, value: list[str]) -> list[str]:
        return validate_allergies_for_save(list(value))


class RecipeLog(StrictBaseModel):
    recipe_name: str = Field(min_length=1)
    recipe_json: dict[str, Any] = Field(
        description="Full recipe payload to display later on the dashboard.",
    )
    calories: float = Field(ge=0)
    protein: float = Field(ge=0)
    carbs: float = Field(ge=0)
    fat: float = Field(ge=0)
    date: str = Field(
        description="Date in YYYY-MM-DD format",
        pattern=r"^\d{4}-\d{2}-\d{2}$",
    )


class RemoveConsumedRecipeRequest(StrictBaseModel):
    """Remove one logged recipe from a day by index (same order as GET /api/summary)."""

    date: str = Field(
        description="Date in YYYY-MM-DD format",
        pattern=r"^\d{4}-\d{2}-\d{2}$",
    )
    index: int = Field(ge=0, description="0-based index into consumed_recipes_json for that day.")


class ExerciseLog(StrictBaseModel):
    date: str = Field(
        description="Date in YYYY-MM-DD format",
        pattern=r"^\d{4}-\d{2}-\d{2}$",
    )
    category: str = Field(
        description="Allowed values: 'cardio' or 'strength'.",
        min_length=1,
    )
    type: Optional[str] = Field(
        default=None,
        description="Cardio type: bike, walk, run, swim.",
    )
    duration_minutes: int = Field(ge=1, le=600)
    intensity: Optional[str] = Field(
        default=None,
        description="Strength intensity: medium, high, very_high.",
    )
    manual_burned_calories: Optional[float] = Field(
        default=None,
        description="If set, stored as burned_calories without MET calculation; weight not required.",
        ge=1.0,
        le=10000.0,
    )


class GenerateMenuRequest(StrictBaseModel):
    barcodes: list[str] = Field(min_length=1)
    diners: int = Field(ge=1, le=20)
    time_available: int = Field(ge=1, le=600)
    dish_type: str = Field(min_length=1)
    dietary_style: Optional[str] = None
    special_requests: Optional[str] = None
    priority_barcode: Optional[str] = Field(default=None, min_length=1)


class CheatMealRequest(StrictBaseModel):
    description: str = Field(min_length=1)

# Used to store the top-5 alternatives for a product.
class VisionAlternative(StrictBaseModel):
    name: str
    barcode: str


class VisionDetectedProduct(StrictBaseModel):
    detected_name: str
    detected_barcode: str
    alternatives: list[VisionAlternative]


class VisionProcessResponse(StrictBaseModel):
    products: list[VisionDetectedProduct]


class VisionEnqueueResponse(StrictBaseModel):
    status: str
    task_id: str


class VisionStatusResponse(StrictBaseModel):
    status: str
    task_id: str
    state: str
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None


class VerifiedProduct(StrictBaseModel):
    barcode: str = Field(min_length=1)
    name: str = Field(min_length=1)


class VisionConfirmRequest(StrictBaseModel):
    verified_products: list[VerifiedProduct]


class ProductAutocompleteSuggestion(StrictBaseModel):
    name: str
    barcode: str
    kcal_per_100g: Optional[float] = None
    protein_g_per_100g: Optional[float] = None
    fat_g_per_100g: Optional[float] = None
    carbs_g_per_100g: Optional[float] = None


class ProductAutocompleteResponse(StrictBaseModel):
    suggestions: list[ProductAutocompleteSuggestion]


class ProductMacrosBatchRequest(StrictBaseModel):
    barcodes: list[str] = Field(default_factory=list, max_length=50)
    lang: Optional[str] = Field(
        default=None,
        description="UI language for product name: 'es' or 'en' (uses bilingual catalog when available).",
    )


class ProductMacrosBatchResponse(StrictBaseModel):
    """Only includes barcodes found in the local macros catalog."""
    macros: dict[str, ProductAutocompleteSuggestion]


class PortionedProductConsumeRequest(StrictBaseModel):
    product_name: str = Field(min_length=1)
    barcode: Optional[str] = None
    grams: float = Field(gt=0, le=10000)


class PortionedProductConsumeResponse(StrictBaseModel):
    status: str
    message: str
    data: dict[str, Any]


# Compares the plain password that the user enters with the hashed password in the database.
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(
        plain_password.encode("utf-8"),
        hashed_password.encode("utf-8"),
    )


def get_password_hash(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def user_has_password_credential(user: dict) -> bool:
    """True if the user registered with email/password or has set a local password; False for OAuth-only accounts."""
    return bool(user.get("password_credential", True))


# Creates a JWT token for the user, it is used to authenticate the user when they are logged in to navigate the backend.
def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: DatabaseManager = Depends(get_db),
) -> int:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        # Decodes the JWT token and returns the payload.
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        # sub is the subject of the token, which is the email of the user. Why email? Because the user's email is used to identify the user in the database.
        email = payload.get("sub")
        if not isinstance(email, str) or not email:
            raise credentials_exception
    except JWTError as error:
        raise credentials_exception from error

    user = db.get_user_by_email(email)
    if user is None:
        raise credentials_exception
    # ID used to identify the user in the database.
    return int(user["id"])


# ─────────────────────────────────────────────────────────────────────────────
# MENU JOBS ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/menu-jobs")
def create_menu_job(
    payload: GenerateMenuRequest,
    user_id: int = Depends(get_current_user),
    db: DatabaseManager = Depends(get_db),
):
    # Endpoint: start an asynchronous "menu generation" job.
    # The API returns immediately with a job_id. The frontend can poll:
    #   - GET /api/menu-jobs/{job_id}         for progress/status
    #   - GET /api/menu-jobs/{job_id}/result  for the final payload

    # Load the user's saved profile from the DB (required to compute targets and constraints).
    profile = db.get_user_profile(user_id)
    if profile is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User profile is required before generating a menu.",
        )

    # Validate and parse optional dietary style string into the internal enum used by the pipeline.
    parsed_dietary_style = parse_dietary_style(payload.dietary_style)
    if payload.dietary_style and parsed_dietary_style is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Invalid dietary_style. Allowed values: "
                "vegan, vegetarian, celiac, mediterranean, asian, arabic, latin."
            ),
        )

    if payload.priority_barcode and payload.priority_barcode not in payload.barcodes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="priority_barcode must be one of the submitted barcodes.",
        )

    # Fetch today's summary to compute remaining calories and macronutrients
    today_string = date_type.today().isoformat()
    summary = db.get_daily_summary(user_id, today_string)

    # Extract user profile tracking properties
    meals_per_day = int(profile.get("meals_per_day") or 3)
    meals_logged_today = len(summary.get("consumed", {}).get("recipes", []))
    remaining_budget = summary.get("remaining", {})

    # Compute the dynamic meal budget allocations for this single session
    budget = compute_meal_targets_from_daily_summary(
        meals_per_day=meals_per_day,
        meals_logged_today=meals_logged_today,
        remaining=remaining_budget,
    )

    # Hydrate profile payload with session-specific non-persistent overrides for Celery
    profile["meals_per_day"] = meals_per_day
    profile["session_meal_caloric_target"] = budget["meal_caloric_target"]
    profile["session_meal_protein_g"] = budget["meal_protein_g"]
    profile["session_meal_carb_g"] = budget["meal_carb_g"]
    profile["session_meal_fat_g"] = budget["meal_fat_g"]
    profile["session_meals_remaining"] = budget["meals_remaining"]

    # Log operational details for diagnostic tracking
    print(
        f"[MenuJob] Meal budget: {budget['meal_caloric_target']} kcal "
        f"({budget['meals_remaining']} meals left, {remaining_budget.get('calories', 0.0)} kcal remaining)."
    )

    job_id = str(uuid4())
    redis_client = getattr(app.state, "redis", None) or get_redis()
    create_menu_job_hash(
        redis_client,
        job_id=job_id,
        user_id=int(user_id),
        created_at=time.time(),
        status="queued",
        progress=0,
        message="Queued",
        ttl_seconds=_MENU_JOB_TTL_SECONDS,
    )
    output_language = _output_language_from_profile(profile)
    try:
        celery_app.send_task(
            "menu.generate",
            args=[
                job_id,
                int(user_id),
                payload.model_dump(),
                profile,
                output_language,
            ],
            task_id=job_id,
        )
    except Exception as exc:
        update_menu_job_hash(
            redis_client,
            job_id,
            status="error",
            progress=0,
            message="Error",
            error=f"Failed to enqueue menu task: {exc}",
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to enqueue menu generation task.",
        ) from exc

    # Return immediately; the client will poll using the returned job_id.
    return {"status": "success", "job_id": job_id}


@app.get("/api/menu-jobs/{job_id}")
def get_menu_job_status(
    job_id: str,
    user_id: int = Depends(get_current_user),
):
    redis_client = getattr(app.state, "redis", None) or get_redis()
    job = _get_job_or_404(redis_client, job_id, user_id)
    return {
        "status": "success",
        "data": {
            "job_id": job.get("job_id"),
            "state": job.get("status"),
            "progress": job.get("progress"),
            "message": job.get("error") or job.get("message"),
        },
    }


@app.get("/api/menu-jobs/{job_id}/result")
def get_menu_job_result(
    job_id: str,
    user_id: int = Depends(get_current_user),
):
    redis_client = getattr(app.state, "redis", None) or get_redis()
    job = _get_job_or_404(redis_client, job_id, user_id)
    if job.get("status") == "error":
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=job.get("error") or "Menu job failed.",
        )
    if job.get("status") != "done" or job.get("result") is None:
        raise HTTPException(
            status_code=status.HTTP_202_ACCEPTED,
            detail="Menu job still running.",
        )
    return {"status": "success", "data": job.get("result")}


@app.post("/api/generate-menu/macros")
def generate_menu_macros(
    user_id: int = Depends(get_current_user),
):
    raise HTTPException(
        status_code=status.HTTP_410_GONE,
        detail="Fast mode is disabled; macros are computed during menu generation.",
    )


@app.post("/api/register")
def register_user(payload: UserRegister, db: DatabaseManager = Depends(get_db)):
    try:
        hashed_password = get_password_hash(payload.password)
        user_id = db.create_user(payload.email, hashed_password, "")
        return {"status": "success", "message": "User registered successfully.", "user_id": user_id}
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error
    except psycopg2.Error as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error while registering user: {error}",
        ) from error
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error while registering user: {error}",
        ) from error

# post difference with put is that post is used to create a resource and put is used to update a resource.
@app.post("/api/login", response_model=Token)
def login_user(
    # OAuth2PasswordRequestForm is a FastAPI dependency that validates the username and password.
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: DatabaseManager = Depends(get_db),
):
    email = form_data.username
    user = db.get_user_by_email(email)
    if user is None or not verify_password(form_data.password, str(user["hashed_password"])):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(data={"sub": str(user["email"])})
    return Token(access_token=access_token, token_type="bearer")


@app.put("/api/settings/email")
def update_email(
    payload: UpdateEmailRequest,
    user_id: int = Depends(get_current_user),
    db: DatabaseManager = Depends(get_db),
):
    try:
        user = db.get_user_by_id(user_id)
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found.",
            )

        if user_has_password_credential(user):
            if not payload.current_password.strip():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Current password is required.",
                )
            if not verify_password(payload.current_password, str(user["hashed_password"])):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Incorrect password.",
                )

        db.update_user_email(user_id, new_email=payload.new_email)
        return {"status": "success", "message": "Email updated successfully."}
    except HTTPException:
        raise
    except psycopg2.IntegrityError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Email update failed: {error}",
        ) from error
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error while updating email: {error}",
        ) from error


@app.post("/api/settings/password")
def change_password(
    payload: ChangePasswordRequest,
    user_id: int = Depends(get_current_user),
    db: DatabaseManager = Depends(get_db),
):
    try:
        user = db.get_user_by_id(user_id)
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found.",
            )

        new_hashed_password = get_password_hash(payload.new_password)
        if user_has_password_credential(user):
            if not payload.current_password.strip():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Current password is required.",
                )
            if not verify_password(payload.current_password, str(user["hashed_password"])):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Incorrect password.",
                )
            db.update_user_password(user_id, new_hashed_password=new_hashed_password)
        else:
            db.update_user_password(
                user_id,
                new_hashed_password=new_hashed_password,
                password_credential=True,
            )
        return {"status": "success", "message": "Password updated successfully."}
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error while changing password: {error}",
        ) from error


@app.delete("/api/settings/account")
def delete_account(
    payload: DeleteAccountRequest,
    user_id: int = Depends(get_current_user),
    db: DatabaseManager = Depends(get_db),
):
    try:
        user = db.get_user_by_id(user_id)
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found.",
            )

        if user_has_password_credential(user):
            if not payload.current_password.strip():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Current password is required.",
                )
            if not verify_password(payload.current_password, str(user["hashed_password"])):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Incorrect password.",
                )

        db.delete_user_account(user_id)
        return {"status": "success", "message": "Account deleted successfully."}
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error while deleting account: {error}",
        ) from error


@app.put("/api/settings/preferences")
def update_preferences(
    payload: PreferencesUpdateRequest,
    user_id: int = Depends(get_current_user),
    db: DatabaseManager = Depends(get_db),
):
    try:
        db.update_user_preferences(
            user_id,
            language=payload.language,
            units=payload.units,
        )
        return {"status": "success", "message": "Preferences updated successfully."}
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error while updating preferences: {error}",
        ) from error


@app.get("/api/auth/google")
def google_oauth_start(request: Request) -> RedirectResponse:
    """Redirects the browser to Google's OAuth consent screen."""
    client_id = os.getenv("GOOGLE_CLIENT_ID")
    google_redirect_uri = _resolve_google_redirect_uri(request)
    
    redis_client = getattr(app.state, "redis", None) or get_redis()
    state = secrets.token_urlsafe(32)
    set_oauth_state(redis_client, state, ttl_seconds=int(_OAUTH_STATE_TTL_SECONDS))

    # urlencode is used to encode the query parameters so they can be sent in the Google OAuth consent screen.
    # client_id is the client ID of the application.
    # redirect_uri is the URL to which Google will redirect the user after they have authenticated.
    # response_type is the type of response we want from Google. We want a code because we will use it to exchange for tokens.
    query = urlencode(
        {
            "client_id": client_id,
            "redirect_uri": google_redirect_uri,
            "response_type": "code",
            "scope": "openid email profile",
            "state": state,
            "access_type": "offline",
            "prompt": "select_account",
        }
    )
    authorization_url = f"{GOOGLE_AUTHORIZATION_URL}?{query}"
    return RedirectResponse(url=authorization_url)


@app.get("/api/auth/callback")
def google_oauth_callback(
    request: Request,
    code: Optional[str] = Query(default=None),
    state: Optional[str] = Query(default=None),
    error: Optional[str] = Query(default=None),
    db: DatabaseManager = Depends(get_db),
) -> RedirectResponse:
    """
    Handles Google's redirect: exchanges the code for tokens, ensures the user
    exists in SQLite, issues a JWT (needed to authenticate the user when they are logged in to navigate the backend), and redirects to the frontend with the token.
    """
    if error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Google OAuth error: {error}",
        )
    if not code or not state:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing authorization code or state.",
        )

    redis_client = getattr(app.state, "redis", None) or get_redis()
    if not consume_oauth_state(redis_client, state):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired OAuth state.",
        )

    client_id = os.getenv("GOOGLE_CLIENT_ID")
    client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
    google_redirect_uri = _resolve_google_redirect_uri(request)

    # Authlib: exchange authorization code for tokens.
    with OAuth2Client(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=google_redirect_uri,
        scope="openid email profile",
    ) as oauth_client:
        token_response = oauth_client.fetch_token(
            GOOGLE_TOKEN_URL,
            grant_type="authorization_code",
            code=code,
            redirect_uri=google_redirect_uri,
        )
        access_token_oauth = str(token_response.get("access_token", ""))

    if not access_token_oauth:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Token exchange did not return an access token.",
        )

    # httpx: fetch verified email from Google userinfo.
    with httpx.Client() as http_client:
        userinfo_response = http_client.get(
            GOOGLE_USERINFO_URL,
            headers={"Authorization": f"Bearer {access_token_oauth}"},
            timeout=30.0,
        )
    if userinfo_response.status_code != status.HTTP_200_OK:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to fetch Google user profile.",
        )
    userinfo = userinfo_response.json()
    email = userinfo.get("email")
    if not isinstance(email, str) or not email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Google account did not return an email address.",
        )

    user_row = db.get_user_by_email(email)
    created_new_user = False
    # If the user does not exist in the database, we create a new user.
    if user_row is None:
        google_name = userinfo.get("name") or f"{userinfo.get('given_name', '')} {userinfo.get('family_name', '')}".strip()
        if not isinstance(google_name, str) or google_name == "":
            # If the user does not have a name, we use the email to generate a random name.
            google_name = str(email).split("@")[0]
        random_password = secrets.token_urlsafe(48)
        # We create a new user in the database.
        db.create_user(
            email,
            get_password_hash(random_password),
            google_name,
            password_credential=False,
        )
        user_row = db.get_user_by_email(email)
        if user_row is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create user after Google sign-in.",
            )
        created_new_user = True
    
    # We create a JWT token for the user. 
    jwt_token = create_access_token(data={"sub": str(user_row["email"])})
    token_query = quote(jwt_token, safe="")
    # If the user is a new user, we add a new_user=1 query parameter to the redirect URL. Used in the frontend to show a page for the user to complete their profile.
    new_user_suffix = "&new_user=1" if created_new_user else ""
    redirect_target = (
        f"{FRONTEND_BASE_URL.rstrip('/')}/login?token={token_query}{new_user_suffix}"
    )
    return RedirectResponse(url=redirect_target)


@app.get("/api/profile")
def get_profile(
    user_id: int = Depends(get_current_user),
    db: DatabaseManager = Depends(get_db),
):
    try:
        profile = db.get_user_profile(user_id)
        if profile is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User profile not found.",
            )
        user = db.get_user_by_id(user_id)
        if user is not None and "name" in user:
            profile["name"] = user["name"]
            profile["password_credential"] = user_has_password_credential(user)
        return profile
    except psycopg2.Error as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error while retrieving profile: {error}",
        ) from error
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error while retrieving profile: {error}",
        ) from error


@app.post("/api/profile")
def save_profile(
    payload: ProfileUpdate,
    user_id: int = Depends(get_current_user),
    db: DatabaseManager = Depends(get_db),
):
    try:
        target_calories, target_protein_g, target_carb_g, target_fat_g = calculate_dynamic_targets(
            age=payload.age,
            gender=payload.gender,
            weight=payload.weight,
            height=payload.height,
            activity_level=payload.activity_level,
            physical_goal=payload.physical_goal,
            weight_goal_rate=payload.weight_goal_rate,
        )

        save_payload = payload.model_dump()
        db.set_user_name(user_id, payload.name)
        save_payload["target_calories"] = target_calories
        save_payload["target_protein_g"] = target_protein_g
        save_payload["target_carb_g"] = target_carb_g
        save_payload["target_fat_g"] = target_fat_g

        db.save_user_profile(user_id, save_payload)
        return {"status": "success", "message": "Profile saved successfully."}
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid profile data: {error}",
        ) from error
    except HTTPException:
        raise
    except psycopg2.Error as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error while saving profile: {error}",
        ) from error
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error while saving profile: {error}",
        ) from error

# Used when the user clicks on the dashboard to see the daily summary.
@app.get("/api/summary")
def get_summary(
    date: Optional[str] = Query(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
    user_id: int = Depends(get_current_user),
    db: DatabaseManager = Depends(get_db),
):
    try:
        target_date = date or date_type.today().isoformat()
        return db.get_daily_summary(user_id, target_date)
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid date format: {error}",
        ) from error
    except psycopg2.Error as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error while retrieving summary: {error}",
        ) from error
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error while retrieving summary: {error}",
        ) from error

# Used when the user selects one of the 3 recipes to consume.
@app.post("/api/consume")
def consume_recipe(
    payload: RecipeLog,
    user_id: int = Depends(get_current_user),
    db: DatabaseManager = Depends(get_db),
):
    try:
        db.log_consumed_recipe(
            user_id=user_id,
            recipe_name=payload.recipe_name,
            recipe_json=payload.recipe_json,
            calories=payload.calories,
            protein=payload.protein,
            carbs=payload.carbs,
            fat=payload.fat,
            date=payload.date,
        )
        return {"status": "success", "message": "Recipe consumption logged successfully."}
    except psycopg2.Error as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error while logging consumption: {error}",
        ) from error
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error while logging consumption: {error}",
        ) from error


@app.post("/api/consume/remove")
def remove_consumed_recipe_entry(
    payload: RemoveConsumedRecipeRequest,
    user_id: int = Depends(get_current_user),
    db: DatabaseManager = Depends(get_db),
):
    try:
        db.remove_consumed_recipe_at_index(user_id, payload.date, payload.index)
        return {"status": "success", "message": "Recipe removed from daily log."}
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error
    except psycopg2.Error as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error while removing consumption: {error}",
        ) from error
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error while removing consumption: {error}",
        ) from error


def _optional_macro_float_record(record: dict[str, Any], key: str) -> Optional[float]:
    value = record.get(key)
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)
    return None


def _totals_from_catalog_record(record: dict[str, Any], grams: float) -> Optional[tuple[float, float, float, float]]:
    """Returns (kcal, protein, carbs, fat) totals if kcal_per_100g is available."""
    kcal_100 = _optional_macro_float_record(record, "kcal_per_100g")
    if kcal_100 is None:
        return None
    factor = float(grams) / 100.0
    p = _optional_macro_float_record(record, "protein_g_per_100g")
    c = _optional_macro_float_record(record, "carbs_g_per_100g")
    f = _optional_macro_float_record(record, "fat_g_per_100g")
    return (
        kcal_100 * factor,
        (p or 0.0) * factor,
        (c or 0.0) * factor,
        (f or 0.0) * factor,
    )


@app.post("/api/consume/portioned-product", response_model=PortionedProductConsumeResponse)
def consume_portioned_product(
    payload: PortionedProductConsumeRequest,
    user_id: int = Depends(get_current_user),
    db: DatabaseManager = Depends(get_db),
):
    """
    Log a single purchased product portion for today using catalog macros when possible,
    otherwise NutritionistAgent per-100g LLM estimate scaled by grams.
    """
    try:
        mapping = get_product_macros_mapping()
        totals: Optional[tuple[float, float, float, float]] = None
        from_catalog = False

        barcode_key = (payload.barcode or "").strip()
        if barcode_key != "" and barcode_key in mapping:
            raw_rec = mapping[barcode_key]
            if isinstance(raw_rec, dict):
                totals = _totals_from_catalog_record(raw_rec, payload.grams)
                if totals is not None:
                    from_catalog = True

        if totals is None:
            nutritionist_agent = get_nutritionist_agent()
            totals = nutritionist_agent.estimate_portion_totals(
                payload.product_name, payload.grams
            )

        kcal, protein, carbs, fat = totals
        kcal = max(0.0, float(kcal))
        protein = max(0.0, float(protein))
        carbs = max(0.0, float(carbs))
        fat = max(0.0, float(fat))

        today_string = date_type.today().isoformat()
        recipe_title = f"Producto: {payload.product_name}"
        accuracy_note_key = "portion_from_catalog" if from_catalog else "portion_estimated"
        recipe_json: dict[str, Any] = {
            "title": recipe_title,
            "estimated_time_minutes": 1,
            "justification_key": "manual_product_portion",
            "ingredients": [],
            "preparation_steps": [],
            "macro_breakdown": {
                "total_kcal": kcal,
                "protein_g": protein,
                "carb_g": carbs,
                "fat_g": fat,
                "accuracy_note_key": accuracy_note_key,
            },
        }

        db.log_consumed_recipe(
            user_id=user_id,
            recipe_name=recipe_title,
            recipe_json=recipe_json,
            calories=kcal,
            protein=protein,
            carbs=carbs,
            fat=fat,
            date=today_string,
        )

        source_note = (
            "From local product nutrition catalog (per 100 g)."
            if from_catalog
            else "Estimated from product name (model)."
        )

        return PortionedProductConsumeResponse(
            status="success",
            message="Product portion logged successfully.",
            data={
                "product_name": payload.product_name,
                "grams": payload.grams,
                "estimated_kcal": kcal,
                "estimated_protein_g": protein,
                "estimated_carb_g": carbs,
                "estimated_fat_g": fat,
                "source_note": source_note,
            },
        )
    except psycopg2.Error as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error while logging product portion: {error}",
        ) from error
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error while logging product portion: {error}",
        ) from error


@app.post("/api/exercise")
def log_exercise(
    payload: ExerciseLog,
    user_id: int = Depends(get_current_user),
    db: DatabaseManager = Depends(get_db),
):
    try:
        profile = db.get_user_profile(user_id)
        if profile is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User profile is required to log exercise.",
            )

        if payload.manual_burned_calories is not None:
            burned_calories = round(float(payload.manual_burned_calories), 2)
            manual_entry = True
        else:
            weight_kg = profile.get("weight")
            if weight_kg is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="User profile must include current weight to calculate burned calories.",
                )

            burned_calories = calculate_exercise_burned_calories(
                category=payload.category,
                duration_minutes=payload.duration_minutes,
                weight_kg=float(weight_kg),
                cardio_type=payload.type,
                intensity=payload.intensity,
            )
            manual_entry = False

        log_id = db.create_exercise_log_and_add_daily(
            user_id=user_id,
            log_date=payload.date,
            category=payload.category,
            duration_minutes=payload.duration_minutes,
            burned_calories=burned_calories,
            cardio_type=payload.type,
            strength_intensity=payload.intensity,
            manual_entry=manual_entry,
        )

        return {
            "status": "success",
            "message": "Exercise logged successfully.",
            "data": {
                "id": log_id,
                "date": payload.date,
                "category": payload.category,
                "duration_minutes": payload.duration_minutes,
                "type": payload.type,
                "intensity": payload.intensity,
                "burned_calories": burned_calories,
                "manual_entry": manual_entry,
            },
        }
    except HTTPException:
        raise
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid exercise data: {error}",
        ) from error
    except psycopg2.Error as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error while logging exercise: {error}",
        ) from error
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error while logging exercise: {error}",
        ) from error


def _serialize_exercise_log_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    created = out.get("created_at")
    if hasattr(created, "isoformat"):
        out["created_at"] = created.isoformat(timespec="seconds")
    return out


@app.get("/api/exercise")
def list_exercises_for_date(
    user_id: int = Depends(get_current_user),
    db: DatabaseManager = Depends(get_db),
    date: Optional[str] = Query(
        default=None,
        description="YYYY-MM-DD; defaults to today when omitted",
    ),
):
    try:
        target = date if date is not None else date_type.today().isoformat()
        try:
            datetime.strptime(target, "%Y-%m-%d")
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid date format. Use YYYY-MM-DD.",
            ) from error
        logs = db.get_exercise_logs_for_date(user_id, target)
        return {"logs": [_serialize_exercise_log_row(dict(x)) for x in logs]}
    except psycopg2.Error as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error while listing exercise logs: {error}",
        ) from error
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error while listing exercise logs: {error}",
        ) from error


@app.delete("/api/exercise/{log_id}")
def delete_exercise_log_endpoint(
    log_id: int,
    user_id: int = Depends(get_current_user),
    db: DatabaseManager = Depends(get_db),
):
    try:
        deleted = db.delete_exercise_log(user_id, log_id)
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Exercise log not found.",
            )
        return {"status": "success", "message": "Exercise log deleted."}
    except HTTPException:
        raise
    except psycopg2.Error as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error while deleting exercise log: {error}",
        ) from error
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error while deleting exercise log: {error}",
        ) from error


@app.get("/api/history/weight")
def get_weight_history(
    user_id: int = Depends(get_current_user),
    db: DatabaseManager = Depends(get_db),
):
    try:
        history = db.get_weight_history(user_id)
        return {"records": history}
    except psycopg2.Error as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error while retrieving weight history: {error}",
        ) from error
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error while retrieving weight history: {error}",
        ) from error

# For selecting a range of days to see the calorie history.
@app.get("/api/history/calories")
def get_calorie_history(
    days: int = Query(default=30, ge=1, le=365),
    user_id: int = Depends(get_current_user),
    db: DatabaseManager = Depends(get_db),
):
    try:
        history = db.get_calorie_history(user_id=user_id, days=days)
        return {"records": history}
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid days parameter: {error}",
        ) from error
    except psycopg2.Error as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error while retrieving calorie history: {error}",
        ) from error
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error while retrieving calorie history: {error}",
        ) from error

# For selecting a range of dates to see the calorie history.
@app.get("/api/history/advanced")
def get_advanced_calorie_history(
    start_date: str = Query(
        description="Start date in YYYY-MM-DD format",
        pattern=r"^\d{4}-\d{2}-\d{2}$",
    ),
    end_date: str = Query(
        description="End date in YYYY-MM-DD format",
        pattern=r"^\d{4}-\d{2}-\d{2}$",
    ),
    user_id: int = Depends(get_current_user),
    db: DatabaseManager = Depends(get_db),
):
    try:
        records = db.get_advanced_calorie_history(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
        )
        return {"records": records}
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid date range: {error}",
        ) from error
    except psycopg2.Error as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error while retrieving advanced history: {error}",
        ) from error
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error while retrieving advanced history: {error}",
        ) from error


@app.get("/api/history/exercise-burn")
def get_exercise_burn_history(
    start_date: str = Query(
        description="Start date in YYYY-MM-DD format",
        pattern=r"^\d{4}-\d{2}-\d{2}$",
    ),
    end_date: str = Query(
        description="End date in YYYY-MM-DD format",
        pattern=r"^\d{4}-\d{2}-\d{2}$",
    ),
    user_id: int = Depends(get_current_user),
    db: DatabaseManager = Depends(get_db),
):
    try:
        records = db.get_exercise_burn_history_by_day(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
        )
        return {"records": records}
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid date range: {error}",
        ) from error
    except psycopg2.Error as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error while retrieving exercise burn history: {error}",
        ) from error
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error while retrieving exercise burn history: {error}",
        ) from error


@app.post("/api/generate-menu")
def generate_menu(
    payload: GenerateMenuRequest,
    user_id: int = Depends(get_current_user),
    db: DatabaseManager = Depends(get_db),
):
    raise HTTPException(
        status_code=status.HTTP_410_GONE,
        detail="This endpoint has been deprecated. Use POST /api/menu-jobs and poll its status/result.",
    )

# Same as generate-menu but with streaming, so the frontend can see the progress of the menu generation.
@app.post("/api/generate-menu/stream")
def generate_menu_stream(
    payload: GenerateMenuRequest,
    user_id: int = Depends(get_current_user),
    db: DatabaseManager = Depends(get_db),
):
    raise HTTPException(
        status_code=status.HTTP_410_GONE,
        detail="This streaming endpoint has been deprecated. Use POST /api/menu-jobs and poll its status/result.",
    )


@app.post("/api/cheat-meal")
def cheat_meal(
    payload: CheatMealRequest,
    user_id: int = Depends(get_current_user),
    db: DatabaseManager = Depends(get_db),
):
    try:
        nutritionist_agent = get_nutritionist_agent()
        report = nutritionist_agent.estimate_cheat_meal(payload.description)

        today_string = date_type.today().isoformat()
        db.log_consumed_recipe(
            user_id=user_id,
            # description is the description of the cheat meal given by the user.
            recipe_name=f"Cheat Meal: {payload.description}",
            recipe_json={
                "title": f"Cheat Meal: {payload.description}",
                "estimated_time_minutes": 1,
                "justification": "Cheat meal was estimated from user text using the Nutritionist agent.",
                "ingredients": [],
                "preparation_steps": [],
                "caloric_note": report.estimation_notes,
                "macro_breakdown": {
                    "total_kcal": float(report.estimated_kcal),
                    "protein_g": float(report.estimated_protein_g),
                    "carb_g": float(report.estimated_carb_g),
                    "fat_g": float(report.estimated_fat_g),
                    "accuracy_note": "Estimated (not from OpenFoodFacts).",
                },
            },
            calories=float(report.estimated_kcal),
            protein=float(report.estimated_protein_g),
            carbs=float(report.estimated_carb_g),
            fat=float(report.estimated_fat_g),
            date=today_string,
        )

        return {
            "status": "success",
            "message": "Cheat meal estimated and logged successfully.",
            "data": {
                "description": payload.description,
                "estimated_kcal": report.estimated_kcal,
                "estimated_protein_g": report.estimated_protein_g,
                "estimated_carb_g": report.estimated_carb_g,
                "estimated_fat_g": report.estimated_fat_g,
                "estimation_notes": report.estimation_notes,
            },
        }
    except psycopg2.Error as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error in cheat-meal: {error}",
        ) from error
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error in cheat-meal: {error}",
        ) from error

# response_model is used to specify the Pydantic model that will be returned by the endpoint.  
# UploadFile is a FastAPI dependency that handles the uploaded file.
# File is a FastAPI dependency that handles the uploaded file.
# ... is used to mark the parameter as required.
@app.post(
    "/api/vision/process",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=VisionEnqueueResponse,
)
def process_vision_image(image: UploadFile = File(...)):
    """Enqueue the vision pipeline in Celery and return a task_id (async)."""
    temp_file_path: Optional[Path] = None
    try:
        uploads_dir = Path(config.DATA_DIR) / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)

        suffix = Path(image.filename).suffix if image.filename else ".jpg"
        # uuid4 is used to generate a random UUID. UUID is a unique identifier for the uploaded file.
        temp_file_path = uploads_dir / f"upload_{uuid4().hex}{suffix}"
        with open(temp_file_path, "wb") as destination:
            # shutil.copyfileobj is used to copy the uploaded file to the temporary file path.
            shutil.copyfileobj(image.file, destination)
            destination.flush()
            os.fsync(destination.fileno())
        try:
            image.file.close()
        except Exception:
            pass

        async_result = process_vision_image_task.delay(str(temp_file_path))
        return VisionEnqueueResponse(status="accepted", task_id=str(async_result.id))
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Vision processing failed: {error}",
        ) from error


@app.get("/api/vision/status/{task_id}", response_model=VisionStatusResponse)
def vision_status(task_id: str):
    """
    Poll Celery task status for the async vision pipeline.
    Returns {state, result?, error?}.
    """
    import importlib

    AsyncResult = importlib.import_module("celery.result").AsyncResult
    res = AsyncResult(task_id, app=celery_app)
    state = str(res.state or "PENDING")
    payload: dict[str, Any] = {
        "status": "success",
        "task_id": str(task_id),
        "state": state,
        "result": None,
        "error": None,
    }
    if state == "SUCCESS":
        try:
            value = res.result
            payload["result"] = value if isinstance(value, dict) else {"value": value}
        except Exception as exc:
            payload["error"] = str(exc)
    elif state == "FAILURE":
        payload["error"] = str(res.result)
    return VisionStatusResponse(**payload)


@app.post("/api/vision/confirm")
def confirm_vision_products(
    payload: VisionConfirmRequest,
    user_id: int = Depends(get_current_user),
    db: DatabaseManager = Depends(get_db),
):
    """Stores the user-verified product list as the latest scan memory."""
    try:
        db.save_recent_scan(
            user_id=user_id,
            image_path="verified_by_user",
            detected_barcodes=payload.model_dump()["verified_products"],
        )
        return {"status": "success", "message": "Verified products saved successfully."}
    except psycopg2.Error as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error while saving verified products: {error}",
        ) from error
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error in vision-confirm: {error}",
        ) from error

@app.get("/api/products/autocomplete", response_model=ProductAutocompleteResponse)
def autocomplete_products(
    query: str = Query(min_length=1, description="Partial commercial product name"),
    limit: int = Query(default=10, ge=1, le=25),
    lang: str = Query(default="es", description="UI language: es or en (display name)"),
    user_id: int = Depends(get_current_user),
):
    mapping, _has_traduced_overlay = get_autocomplete_product_mapping()

    q = normalize_product_text(query)
    if q == "":
        return ProductAutocompleteResponse(suggestions=[])

    lang_norm = (lang or "es").strip().lower()
    if lang_norm.startswith("en"):
        lang_norm = "en"
    else:
        lang_norm = "es"

    top = autocomplete_top_matches(mapping, q, lang_norm, limit)
    suggestions: list[ProductAutocompleteSuggestion] = []
    for _fuzzy, barcode, record in top:
        display_name = display_product_name_for_lang(record, lang_norm)
        if not display_name:
            continue
        suggestions.append(
            ProductAutocompleteSuggestion(
                name=display_name,
                barcode=barcode,
                kcal_per_100g=_optional_macro_float_record(record, "kcal_per_100g"),
                protein_g_per_100g=_optional_macro_float_record(record, "protein_g_per_100g"),
                fat_g_per_100g=_optional_macro_float_record(record, "fat_g_per_100g"),
                carbs_g_per_100g=_optional_macro_float_record(record, "carbs_g_per_100g"),
            ),
        )
    return ProductAutocompleteResponse(suggestions=suggestions)


@app.post("/api/products/macros-by-barcodes", response_model=ProductMacrosBatchResponse)
def get_product_macros_batch(
    payload: ProductMacrosBatchRequest,
    user_id: int = Depends(get_current_user),
):
    """Returns macro fields per 100 g for known barcodes (local JSON catalog)."""
    mapping, _ = get_autocomplete_product_mapping()
    if not mapping:
        mapping = get_product_macros_mapping()

    lang_raw = payload.lang
    lang_norm = (lang_raw or "es").strip().lower()
    if lang_norm.startswith("en"):
        lang_norm = "en"
    else:
        lang_norm = "es"

    seen: set[str] = set()
    out: dict[str, ProductAutocompleteSuggestion] = {}

    for raw in payload.barcodes:
        if not isinstance(raw, str):
            continue
        code = raw.strip()
        if code == "" or code in seen:
            continue
        seen.add(code)
        if len(seen) > 50:
            break

        record = mapping.get(code)
        if not isinstance(record, dict):
            continue
        display_name = display_product_name_for_lang(record, lang_norm)
        if not display_name:
            continue

        out[code] = ProductAutocompleteSuggestion(
            name=display_name,
            barcode=code,
            kcal_per_100g=_optional_macro_float_record(record, "kcal_per_100g"),
            protein_g_per_100g=_optional_macro_float_record(record, "protein_g_per_100g"),
            fat_g_per_100g=_optional_macro_float_record(record, "fat_g_per_100g"),
            carbs_g_per_100g=_optional_macro_float_record(record, "carbs_g_per_100g"),
        )

    return ProductMacrosBatchResponse(macros=out)


@app.get("/api/vision/previous")
def get_previous_vision_scan(
    user_id: int = Depends(get_current_user),
    db: DatabaseManager = Depends(get_db),
):
    """Returns the most recently saved user-verified product list."""
    try:
        latest_scan = db.get_latest_scan(user_id)
        if latest_scan is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No previous scan found.",
            )
        return {
            "image_path": latest_scan.get("image_path"),
            "detected_barcodes": latest_scan.get("detected_barcodes", []),
        }
    except HTTPException:
        raise
    except psycopg2.Error as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error while retrieving previous scan: {error}",
        ) from error
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error in vision-previous: {error}",
        ) from error

