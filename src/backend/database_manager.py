from __future__ import annotations
import json
import os
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Optional

import psycopg2
import psycopg2.errors
import psycopg2.extras

from src.backend.allergy_ids import migrate_legacy_allergies

# Process-local: schema migrations run at most once per worker (see ensure_schema_once).
_schema_ensured = False
_schema_guard = threading.Lock()


class DatabaseManager:
    """SQLite DAO for users, profiles, nutrition tracking, and scans."""

    # Advisory lock keys for one-time DDL (avoids deadlocks vs concurrent DML across workers).
    _SCHEMA_ADVISORY_KEY1 = 1_802_473_311
    _SCHEMA_ADVISORY_KEY2 = 1

    def __init__(self, db_path: str | None = None) -> None:
        # db_path is ignored in PostgreSQL mode. It is kept to preserve the public API.
        self.db_path = db_path
        # psycopg2 is the PostgreSQL adapter for Python. We initialize the connection to the database to None because it is not yet established.
        self.connection: Optional[psycopg2.extensions.connection[Any]] = None

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------
    @staticmethod
    def _apply_schema_ddl(cursor: psycopg2.extras.RealDictCursor) -> None:
        """CREATE/ALTER statements only; caller must run inside a transaction."""
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL DEFAULT '',
                hashed_password TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id INTEGER PRIMARY KEY,
                location TEXT,
                age INTEGER,
                gender TEXT,
                language TEXT NOT NULL DEFAULT 'es',
                units TEXT NOT NULL DEFAULT 'metric',
                weight REAL,
                height REAL,
                activity_level TEXT,
                allergies TEXT,
                physical_goal TEXT,
                weight_goal_rate REAL,
                target_calories REAL,
                target_protein_g REAL,
                target_carb_g REAL,
                target_fat_g REAL,
                meals_per_day INTEGER NOT NULL DEFAULT 3,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS daily_consumption (
                user_id INTEGER NOT NULL,
                consumption_date TEXT NOT NULL,
                consumed_calories REAL NOT NULL DEFAULT 0,
                consumed_protein REAL NOT NULL DEFAULT 0,
                consumed_carbs REAL NOT NULL DEFAULT 0,
                consumed_fat REAL NOT NULL DEFAULT 0,
                burned_calories REAL NOT NULL DEFAULT 0,
                consumed_recipes TEXT NOT NULL DEFAULT '[]',
                consumed_recipes_json TEXT NOT NULL DEFAULT '[]',
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (user_id, consumption_date)
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS weight_history (
                user_id INTEGER NOT NULL,
                date TEXT NOT NULL,
                weight_kg REAL NOT NULL,
                PRIMARY KEY (user_id, date)
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS recent_scans (
                id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL,
                scan_date TEXT NOT NULL,
                image_path TEXT NOT NULL,
                detected_barcodes TEXT NOT NULL
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS exercise_logs (
                id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL,
                log_date TEXT NOT NULL,
                category TEXT NOT NULL,
                duration_minutes INTEGER NOT NULL,
                cardio_type TEXT,
                strength_intensity TEXT,
                burned_calories REAL NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_exercise_logs_user_date
            ON exercise_logs (user_id, log_date)
            """
        )

        # Safe column additions for backward compatibility.
        cursor.execute(
            "ALTER TABLE daily_consumption ADD COLUMN IF NOT EXISTS burned_calories REAL NOT NULL DEFAULT 0"
        )
        cursor.execute(
            "ALTER TABLE daily_consumption ADD COLUMN IF NOT EXISTS consumed_recipes_json TEXT NOT NULL DEFAULT '[]'"
        )
        cursor.execute(
            "ALTER TABLE recent_scans ADD COLUMN IF NOT EXISTS user_id INTEGER NOT NULL DEFAULT 1"
        )
        cursor.execute(
            "ALTER TABLE users ADD COLUMN IF NOT EXISTS name TEXT NOT NULL DEFAULT ''"
        )
        cursor.execute(
            "ALTER TABLE user_profiles ADD COLUMN IF NOT EXISTS language TEXT NOT NULL DEFAULT 'es'"
        )
        cursor.execute(
            "ALTER TABLE user_profiles ADD COLUMN IF NOT EXISTS units TEXT NOT NULL DEFAULT 'metric'"
        )
        cursor.execute(
            "ALTER TABLE users ADD COLUMN IF NOT EXISTS password_credential BOOLEAN NOT NULL DEFAULT TRUE"
        )
        cursor.execute(
            "ALTER TABLE exercise_logs ADD COLUMN IF NOT EXISTS manual_entry BOOLEAN NOT NULL DEFAULT FALSE"
        )

        cursor.execute(
            "ALTER TABLE user_profiles ADD COLUMN IF NOT EXISTS meals_per_day INTEGER NOT NULL DEFAULT 3"
        )

        # ------------------------------------------------------------------
        # Referential integrity (defensive cleanup + ON DELETE CASCADE FKs)
        # ------------------------------------------------------------------
        # Defensive cleanup: remove orphan rows before adding FK constraints.
        cursor.execute(
            """
            DELETE FROM user_profiles up
            WHERE NOT EXISTS (SELECT 1 FROM users u WHERE u.id = up.user_id)
            """
        )
        cursor.execute(
            """
            DELETE FROM daily_consumption dc
            WHERE NOT EXISTS (SELECT 1 FROM users u WHERE u.id = dc.user_id)
            """
        )
        cursor.execute(
            """
            DELETE FROM weight_history wh
            WHERE NOT EXISTS (SELECT 1 FROM users u WHERE u.id = wh.user_id)
            """
        )
        cursor.execute(
            """
            DELETE FROM recent_scans rs
            WHERE NOT EXISTS (SELECT 1 FROM users u WHERE u.id = rs.user_id)
            """
        )
        cursor.execute(
            """
            DELETE FROM exercise_logs el
            WHERE NOT EXISTS (SELECT 1 FROM users u WHERE u.id = el.user_id)
            """
        )

        # Add FK constraints idempotently (safe to re-run at startup).
        cursor.execute(
            """
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1
                    FROM pg_constraint
                    WHERE conname = 'fk_user_profiles_user_id'
                      AND conrelid = 'user_profiles'::regclass
                ) THEN
                    EXECUTE 'ALTER TABLE user_profiles '
                            'ADD CONSTRAINT fk_user_profiles_user_id '
                            'FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE';
                END IF;
            END $$;
            """
        )
        cursor.execute(
            """
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1
                    FROM pg_constraint
                    WHERE conname = 'fk_daily_consumption_user_id'
                      AND conrelid = 'daily_consumption'::regclass
                ) THEN
                    EXECUTE 'ALTER TABLE daily_consumption '
                            'ADD CONSTRAINT fk_daily_consumption_user_id '
                            'FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE';
                END IF;
            END $$;
            """
        )
        cursor.execute(
            """
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1
                    FROM pg_constraint
                    WHERE conname = 'fk_weight_history_user_id'
                      AND conrelid = 'weight_history'::regclass
                ) THEN
                    EXECUTE 'ALTER TABLE weight_history '
                            'ADD CONSTRAINT fk_weight_history_user_id '
                            'FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE';
                END IF;
            END $$;
            """
        )
        cursor.execute(
            """
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1
                    FROM pg_constraint
                    WHERE conname = 'fk_recent_scans_user_id'
                      AND conrelid = 'recent_scans'::regclass
                ) THEN
                    EXECUTE 'ALTER TABLE recent_scans '
                            'ADD CONSTRAINT fk_recent_scans_user_id '
                            'FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE';
                END IF;
            END $$;
            """
        )
        cursor.execute(
            """
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1
                    FROM pg_constraint
                    WHERE conname = 'fk_exercise_logs_user_id'
                      AND conrelid = 'exercise_logs'::regclass
                ) THEN
                    EXECUTE 'ALTER TABLE exercise_logs '
                            'ADD CONSTRAINT fk_exercise_logs_user_id '
                            'FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE';
                END IF;
            END $$;
            """
        )

    @classmethod
    def ensure_schema_once(cls, db_url: str) -> None:
        """
        Runs migrations once per process under a PostgreSQL transaction advisory lock so
        concurrent workers do not interleave DDL with application DML (deadlock risk).
        """
        global _schema_ensured
        if _schema_ensured:
            return
        with _schema_guard:
            if _schema_ensured:
                return
            conn = psycopg2.connect(db_url)
            try:
                conn.autocommit = False
                try:
                    with conn.cursor() as lock_cur:
                        lock_cur.execute(
                            "SELECT pg_advisory_xact_lock(%s, %s)",
                            (cls._SCHEMA_ADVISORY_KEY1, cls._SCHEMA_ADVISORY_KEY2),
                        )
                    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                        cls._apply_schema_ddl(cursor)
                    conn.commit()
                except Exception:
                    conn.rollback()
                    raise
            finally:
                conn.close()
            _schema_ensured = True

    def _cursor(self) -> psycopg2.extras.RealDictCursor:
        if self.connection is None:
            raise RuntimeError("Database connection is not initialized.")
        return self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    # ------------------------------------------------------------------
    # Users
    # ------------------------------------------------------------------
    def create_user(
        self,
        email: str,
        hashed_password: str,
        name: str,
        *,
        password_credential: bool = True,
    ) -> int:
        """Creates a user and returns its generated identifier."""
        if self.connection is None:
            raise RuntimeError("Database connection is not initialized.")

        try:
            with self.connection:
                with self._cursor() as cursor:
                    cursor.execute(
                        """
                        INSERT INTO users (email, name, hashed_password, password_credential)
                        VALUES (%s, %s, %s, %s)
                        RETURNING id
                        """,
                        (email, name, hashed_password, password_credential),
                    )
                    # fetchone() is a method that returns the first row of the result set.
                    row = cursor.fetchone()
                    if row is None or "id" not in row:
                        raise RuntimeError("Failed to create user.")
                    return int(row["id"])
        # IntegrityError is raised when a unique constraint is violated.
        except psycopg2.IntegrityError as error:
            raise ValueError("A user with that email already exists.") from error

    # Used for authentication, for example when the user logs in or in Google OAuth2.
    def get_user_by_email(self, email: str) -> dict | None:
        """Returns one user row by email or None."""
        with self._cursor() as cursor:
            cursor.execute(
                """
                SELECT id, email, name, hashed_password, created_at
                FROM users
                WHERE email = %s
                """,
                (email,),
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    # Authentication by ID for not dependent on email.
    def get_user_by_id(self, user_id: int) -> dict | None:
        """Returns one user row by identifier or None."""
        with self._cursor() as cursor:
            cursor.execute(
                """
                SELECT id, email, name, hashed_password, password_credential, created_at
                FROM users
                WHERE id = %s
                """,
                (int(user_id),),
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def set_user_name(self, user_id: int, name: str) -> None:
        """Updates a user display name."""
        if self.connection is None:
            raise RuntimeError("Database connection is not initialized.")
        with self.connection:
            with self._cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE users
                    SET name = %s
                    WHERE id = %s
                    """,
                    (name, int(user_id)),
                )

    def update_user_email(self, user_id: int, new_email: str) -> None:
        """Updates the user's email address."""
        if self.connection is None:
            raise RuntimeError("Database connection is not initialized.")
        with self.connection:
            with self._cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE users
                    SET email = %s
                    WHERE id = %s
                    """,
                    (new_email, int(user_id)),
                )

    def update_user_password(
        self,
        user_id: int,
        new_hashed_password: str,
        *,
        password_credential: bool | None = None,
    ) -> None:
        """Updates the user's password hash. Optionally sets password_credential (e.g. True after first password set)."""
        if self.connection is None:
            raise RuntimeError("Database connection is not initialized.")
        with self.connection:
            with self._cursor() as cursor:
                if password_credential is None:
                    cursor.execute(
                        """
                        UPDATE users
                        SET hashed_password = %s
                        WHERE id = %s
                        """,
                        (new_hashed_password, int(user_id)),
                    )
                else:
                    cursor.execute(
                        """
                        UPDATE users
                        SET hashed_password = %s, password_credential = %s
                        WHERE id = %s
                        """,
                        (new_hashed_password, password_credential, int(user_id)),
                    )

    # Used to update language and units.
    def update_user_preferences(self, user_id: int, language: str, units: str) -> None:
        """
        Updates user preferences stored in the profile row.
        This requires an existing profile row.
        """
        if self.connection is None:
            raise RuntimeError("Database connection is not initialized.")
        with self.connection:
            with self._cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE user_profiles
                    SET language = %s, units = %s, updated_at = CURRENT_TIMESTAMP
                    WHERE user_id = %s
                    """,
                    (language, units, int(user_id)),
                )
                if cursor.rowcount == 0:
                    raise ValueError(
                        "User profile not found. Create the profile before updating preferences."
                    )

    def delete_user_account(self, user_id: int) -> None:
        """Deletes the user account; associated rows are removed by ON DELETE CASCADE FKs."""
        if self.connection is None:
            raise RuntimeError("Database connection is not initialized.")
        with self.connection:
            with self._cursor() as cursor:
                cursor.execute(
                    "DELETE FROM users WHERE id = %s",
                    (int(user_id),),
                )

    # Used to get the calorie history when the user wants to view their calorie history.
    def get_advanced_calorie_history(
        self,
        user_id: int,
        start_date: str,
        end_date: str,
    ) -> list[dict[str, Any]]:
        """
        Returns daily aggregates for calories and all macros in the inclusive range.
        Missing days are filled with zeros for chart stability.
        """
        normalized_start = self._normalize_date(start_date)
        normalized_end = self._normalize_date(end_date)

        start_dt = datetime.strptime(normalized_start, "%Y-%m-%d").date()
        end_dt = datetime.strptime(normalized_end, "%Y-%m-%d").date()
        if end_dt < start_dt:
            raise ValueError("end_date must be on or after start_date.")

        with self._cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    consumption_date AS date,
                    consumed_calories,
                    consumed_protein,
                    consumed_carbs,
                    consumed_fat,
                    burned_calories
                FROM daily_consumption
                WHERE user_id = %s
                  AND consumption_date >= %s
                  AND consumption_date <= %s
                ORDER BY consumption_date ASC
                """,
                (int(user_id), normalized_start, normalized_end),
            )
            rows = cursor.fetchall()

        by_date: dict[str, dict[str, Any]] = {}
        # by_date is a dictionary that maps the date to the row (daily consumption row)
        for row in rows:
            by_date[str(row.get("date"))] = dict(row)

        result: list[dict[str, Any]] = []
        current_dt = start_dt
        while current_dt <= end_dt:
            date_text = current_dt.strftime("%Y-%m-%d")
            existing = by_date.get(date_text)
            if existing is None:
                result.append(
                    {
                        "date": date_text,
                        "consumed_calories": 0.0,
                        "consumed_protein": 0.0,
                        "consumed_carbs": 0.0,
                        "consumed_fat": 0.0,
                        "burned_calories": 0.0,
                    }
                )
            else:
                result.append(
                    {
                        "date": date_text,
                        "consumed_calories": float(existing.get("consumed_calories", 0.0)),
                        "consumed_protein": float(existing.get("consumed_protein", 0.0)),
                        "consumed_carbs": float(existing.get("consumed_carbs", 0.0)),
                        "consumed_fat": float(existing.get("consumed_fat", 0.0)),
                        "burned_calories": float(existing.get("burned_calories", 0.0)),
                    }
                )
            current_dt = current_dt + timedelta(days=1)

        return result

    # ------------------------------------------------------------------
    # User profile
    # ------------------------------------------------------------------

    # Useful when the user updates their profile.
    def save_user_profile(self, user_id: int, profile: dict) -> None:
        """Upserts user profile by user identifier."""
        # allergies_value is the allergies of the user. We want to insert it in the user_profiles table.
        allergies_value = profile.get("allergies", [])
        # If allergies_value is a list, we convert it to a JSON string.
        if isinstance(allergies_value, list):
            allergies_text = json.dumps(allergies_value, ensure_ascii=True)
        # If allergies_value is None, we set it to an empty JSON string.
        elif allergies_value is None:
            allergies_text = "[]"
        else:
            allergies_text = str(allergies_value)

        payload = {
            "location": profile.get("location"),
            "age": profile.get("age"),
            "gender": profile.get("gender"),
            "weight": profile.get("weight"),
            "height": profile.get("height"),
            "activity_level": profile.get("activity_level"),
            "allergies": allergies_text,
            "physical_goal": profile.get("physical_goal"),
            "weight_goal_rate": profile.get("weight_goal_rate"),
            "target_calories": profile.get("target_calories"),
            "target_protein_g": profile.get("target_protein_g"),
            "target_carb_g": profile.get("target_carb_g"),
            "target_fat_g": profile.get("target_fat_g"),
            "meals_per_day": profile.get("meals_per_day", 3),
        }

        if self.connection is None:
            raise RuntimeError("Database connection is not initialized.")

        with self.connection:
            with self._cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO user_profiles (
                        user_id,
                        location,
                        age,
                        gender,
                        weight,
                        height,
                        activity_level,
                        allergies,
                        physical_goal,
                        weight_goal_rate,
                        target_calories,
                        target_protein_g,
                        target_carb_g,
                        target_fat_g,
                        meals_per_day,
                        updated_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (user_id) DO UPDATE SET
                        location = EXCLUDED.location,
                        age = EXCLUDED.age,
                        gender = EXCLUDED.gender,
                        weight = EXCLUDED.weight,
                        height = EXCLUDED.height,
                        activity_level = EXCLUDED.activity_level,
                        allergies = EXCLUDED.allergies,
                        physical_goal = EXCLUDED.physical_goal,
                        weight_goal_rate = EXCLUDED.weight_goal_rate,
                        target_calories = EXCLUDED.target_calories,
                        target_protein_g = EXCLUDED.target_protein_g,
                        target_carb_g = EXCLUDED.target_carb_g,
                        target_fat_g = EXCLUDED.target_fat_g,
                        meals_per_day = EXCLUDED.meals_per_day,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (
                        int(user_id),
                        payload["location"],
                        payload["age"],
                        payload["gender"],
                        payload["weight"],
                        payload["height"],
                        payload["activity_level"],
                        payload["allergies"],
                        payload["physical_goal"],
                        payload["weight_goal_rate"],
                        payload["target_calories"],
                        payload["target_protein_g"],
                        payload["target_carb_g"],
                        payload["target_fat_g"],
                        payload["meals_per_day"],
                    ),
                )

                # Persist today's weight snapshot for trend charts (overwrite same-day value).
                if payload.get("weight") is not None:
                    today_text = datetime.now().strftime("%Y-%m-%d")
                    cursor.execute(
                        """
                        INSERT INTO weight_history (user_id, date, weight_kg)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (user_id, date) DO UPDATE SET
                            weight_kg = EXCLUDED.weight_kg
                        """,
                        (int(user_id), today_text, float(payload["weight"])),
                    )

    # Used to get the user profile when the user wants to view their profile.
    def get_user_profile(self, user_id: int) -> dict | None:
        """Returns user profile as dictionary or None if not found."""
        with self._cursor() as cursor:
            cursor.execute("SELECT * FROM user_profiles WHERE user_id = %s", (int(user_id),))
            row = cursor.fetchone()
            if row is None:
                return None

            profile = dict(row)
            profile["allergies"] = migrate_legacy_allergies(
                self._parse_json_list(profile.get("allergies"))
            )

            profile["meals_per_day"] = int(profile.get("meals_per_day") or 3)
            return profile

    # ------------------------------------------------------------------
    # Daily consumption
    # ------------------------------------------------------------------
    def log_consumed_recipe(
        self,
        user_id: int,
        recipe_name: str,
        recipe_json: dict[str, Any],
        calories: float,
        protein: float,
        carbs: float,
        fat: float,
        date: str,
    ) -> None:
        """
        Adds recipe macros to the specified day and appends recipe name
        to `consumed_recipes`.
        """
        normalized_date = self._normalize_date(date)
        self._ensure_daily_row(user_id, normalized_date)

        with self._cursor() as cursor:
            cursor.execute(
                """
                SELECT consumed_recipes, consumed_recipes_json
                FROM daily_consumption
                WHERE user_id = %s AND consumption_date = %s
                """,
                (int(user_id), normalized_date),
            )
            current_row = cursor.fetchone()

        existing_recipe_names = self._parse_json_list(
            current_row["consumed_recipes"] if current_row else "[]"
        )
        # Only the title
        existing_recipe_names.append(recipe_name)

        # We need the json format of the recipes to insert it in the daily_consumption table. the JSON includes the full recipe data.
        existing_recipe_json_list = self._parse_json_list(
            current_row["consumed_recipes_json"] if current_row else "[]"
        )
        if not isinstance(existing_recipe_json_list, list):
            existing_recipe_json_list = []

        existing_recipe_json_list.append(recipe_json)

        updated_recipes_text = json.dumps(existing_recipe_names, ensure_ascii=True)
        updated_recipes_json_text = json.dumps(existing_recipe_json_list, ensure_ascii=True)

        if self.connection is None:
            raise RuntimeError("Database connection is not initialized.")

        with self.connection:
            with self._cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE daily_consumption
                    SET
                        consumed_calories = consumed_calories + %s,
                        consumed_protein = consumed_protein + %s,
                        consumed_carbs = consumed_carbs + %s,
                        consumed_fat = consumed_fat + %s,
                        consumed_recipes = %s,
                        consumed_recipes_json = %s,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE user_id = %s AND consumption_date = %s
                    """,
                    (
                        float(calories),
                        float(protein),
                        float(carbs),
                        float(fat),
                        updated_recipes_text,
                        updated_recipes_json_text,
                        int(user_id),
                        normalized_date,
                    ),
                )

    @staticmethod
    def _totals_from_stored_recipe_json(recipe_json: dict[str, Any]) -> tuple[float, float, float, float]:
        """Returns (kcal, protein, carbs, fat) from a stored consumed recipe JSON blob."""
        mb = recipe_json.get("macro_breakdown")
        if isinstance(mb, dict):
            kcal = float(mb.get("total_kcal") or 0)
            p = float(mb.get("protein_g") or 0)
            c = float(mb.get("carb_g") or 0)
            f = float(mb.get("fat_g") or 0)
            return (max(0.0, kcal), max(0.0, p), max(0.0, c), max(0.0, f))
        return (0.0, 0.0, 0.0, 0.0)

    def remove_consumed_recipe_at_index(self, user_id: int, date: str, index: int) -> None:
        """
        Removes one consumed recipe by list index for the given day and subtracts its macros
        from the daily aggregates.
        """
        if index < 0:
            raise ValueError("Invalid recipe index.")

        normalized_date = self._normalize_date(date)
        self._ensure_daily_row(user_id, normalized_date)

        with self._cursor() as cursor:
            cursor.execute(
                """
                SELECT consumed_recipes, consumed_recipes_json
                FROM daily_consumption
                WHERE user_id = %s AND consumption_date = %s
                """,
                (int(user_id), normalized_date),
            )
            current_row = cursor.fetchone()

        existing_recipe_names = self._parse_json_list(
            current_row["consumed_recipes"] if current_row else "[]"
        )
        if not isinstance(existing_recipe_names, list):
            existing_recipe_names = []
        existing_recipe_json_list = self._parse_json_list(
            current_row["consumed_recipes_json"] if current_row else "[]"
        )
        if not isinstance(existing_recipe_json_list, list):
            existing_recipe_json_list = []

        if index >= len(existing_recipe_json_list):
            raise ValueError("Invalid recipe index.")

        removed = existing_recipe_json_list.pop(index)
        if not isinstance(removed, dict):
            removed = {}

        if index < len(existing_recipe_names):
            existing_recipe_names.pop(index)
        else:
            title = removed.get("title")
            if isinstance(title, str) and title in existing_recipe_names:
                existing_recipe_names.remove(title)

        kcal_r, protein_r, carbs_r, fat_r = self._totals_from_stored_recipe_json(removed)

        updated_recipes_text = json.dumps(existing_recipe_names, ensure_ascii=True)
        updated_recipes_json_text = json.dumps(existing_recipe_json_list, ensure_ascii=True)

        if self.connection is None:
            raise RuntimeError("Database connection is not initialized.")

        with self.connection:
            with self._cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE daily_consumption
                    SET
                        consumed_calories = GREATEST(0, consumed_calories - %s),
                        consumed_protein = GREATEST(0, consumed_protein - %s),
                        consumed_carbs = GREATEST(0, consumed_carbs - %s),
                        consumed_fat = GREATEST(0, consumed_fat - %s),
                        consumed_recipes = %s,
                        consumed_recipes_json = %s,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE user_id = %s AND consumption_date = %s
                    """,
                    (
                        float(kcal_r),
                        float(protein_r),
                        float(carbs_r),
                        float(fat_r),
                        updated_recipes_text,
                        updated_recipes_json_text,
                        int(user_id),
                        normalized_date,
                    ),
                )

    # Used for the dashboard to show the daily summary.
    def get_daily_summary(self, user_id: int, date: str) -> dict:
        """
        Returns target/consumed/remaining macro summary for the given date.
        Targets are read directly from user_profiles.
        """
        normalized_date = self._normalize_date(date)
        self._ensure_daily_row(user_id, normalized_date)

        profile = self.get_user_profile(user_id)

        with self._cursor() as cursor:
            cursor.execute(
                """
                SELECT *
                FROM daily_consumption
                WHERE user_id = %s AND consumption_date = %s
                """,
                (int(user_id), normalized_date),
            )
            daily_row = cursor.fetchone()
        daily = dict(daily_row) if daily_row else {}

        target_calories = float(profile.get("target_calories", 0.0)) if profile else 0.0
        target_protein = float(profile.get("target_protein_g", 0.0)) if profile else 0.0
        target_carbs = float(profile.get("target_carb_g", 0.0)) if profile else 0.0
        target_fat = float(profile.get("target_fat_g", 0.0)) if profile else 0.0

        consumed_calories = float(daily.get("consumed_calories", 0.0))
        consumed_protein = float(daily.get("consumed_protein", 0.0))
        consumed_carbs = float(daily.get("consumed_carbs", 0.0))
        consumed_fat = float(daily.get("consumed_fat", 0.0))
        burned_calories = float(daily.get("burned_calories", 0.0))
        burned_strength, burned_cardio = self._exercise_category_totals_for_date(
            user_id, normalized_date
        )

        consumed_recipes = self._parse_json_list(daily.get("consumed_recipes", "[]"))
        consumed_recipes_json = self._parse_json_list(daily.get("consumed_recipes_json", "[]"))
        if not isinstance(consumed_recipes_json, list):
            consumed_recipes_json = []

        # Backward compatibility: if consumed_recipes_json is empty, fall back to titles only.
        if len(consumed_recipes_json) == 0 and isinstance(consumed_recipes, list):
            consumed_recipe_payloads: list[Any] = [
                {"title": name} for name in consumed_recipes
            ]
        else:
            consumed_recipe_payloads = consumed_recipes_json

        return {
            "date": normalized_date,
            "targets": {
                "calories": round(target_calories, 2),
                "protein_g": round(target_protein, 2),
                "carbs_g": round(target_carbs, 2),
                "fat_g": round(target_fat, 2),
            },
            "consumed": {
                "calories": round(consumed_calories, 2),
                "protein_g": round(consumed_protein, 2),
                "carbs_g": round(consumed_carbs, 2),
                "fat_g": round(consumed_fat, 2),
                "burned_calories": round(burned_calories, 2),
                "burned_calories_strength": round(burned_strength, 2),
                "burned_calories_cardio": round(burned_cardio, 2),
                "recipes": consumed_recipe_payloads,
            },
            "remaining": {
                "calories": round(target_calories - consumed_calories + burned_calories, 2),
                "protein_g": round(target_protein - consumed_protein, 2),
                "carbs_g": round(target_carbs - consumed_carbs, 2),
                "fat_g": round(target_fat - consumed_fat, 2),
            },
            "profile_found": profile is not None,
        }

    def log_burned_calories(self, user_id: int, date: str, burned_calories: float) -> None:
        """Adds burned calories to the specified day."""
        normalized_date = self._normalize_date(date)
        self._ensure_daily_row(user_id, normalized_date)

        if self.connection is None:
            raise RuntimeError("Database connection is not initialized.")

        with self.connection:
            with self._cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE daily_consumption
                    SET
                        burned_calories = burned_calories + %s,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE user_id = %s AND consumption_date = %s
                    """,
                    (float(burned_calories), int(user_id), normalized_date),
                )

    def _exercise_category_totals_for_date(self, user_id: int, date: str) -> tuple[float, float]:
        """Returns (strength_kcal, cardio_kcal) summed from exercise_logs for that day."""
        normalized = self._normalize_date(date)
        with self._cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    COALESCE(
                        SUM(CASE WHEN category = 'strength' THEN burned_calories ELSE 0 END),
                        0
                    ) AS s,
                    COALESCE(
                        SUM(CASE WHEN category = 'cardio' THEN burned_calories ELSE 0 END),
                        0
                    ) AS c
                FROM exercise_logs
                WHERE user_id = %s AND log_date = %s
                """,
                (int(user_id), normalized),
            )
            row = cursor.fetchone()
            if row is None:
                return 0.0, 0.0
            return float(row.get("s", 0.0)), float(row.get("c", 0.0))

    def create_exercise_log_and_add_daily(
        self,
        user_id: int,
        log_date: str,
        category: str,
        duration_minutes: int,
        burned_calories: float,
        cardio_type: str | None,
        strength_intensity: str | None,
        *,
        manual_entry: bool = False,
    ) -> int:
        """
        Inserts one exercise_logs row and increments daily_consumption.burned_calories
        in a single transaction.
        """
        normalized_date = self._normalize_date(log_date)
        self._ensure_daily_row(user_id, normalized_date)

        if self.connection is None:
            raise RuntimeError("Database connection is not initialized.")

        with self.connection:
            with self._cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO exercise_logs (
                        user_id, log_date, category, duration_minutes,
                        cardio_type, strength_intensity, burned_calories, manual_entry
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        int(user_id),
                        normalized_date,
                        category,
                        int(duration_minutes),
                        cardio_type,
                        strength_intensity,
                        float(burned_calories),
                        manual_entry,
                    ),
                )
                row = cursor.fetchone()
                if row is None:
                    raise RuntimeError("INSERT exercise_logs did not return id.")
                log_id = int(row["id"])
                cursor.execute(
                    """
                    UPDATE daily_consumption
                    SET
                        burned_calories = burned_calories + %s,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE user_id = %s AND consumption_date = %s
                    """,
                    (float(burned_calories), int(user_id), normalized_date),
                )
        return log_id

    def get_exercise_logs_for_date(self, user_id: int, date: str) -> list[dict[str, Any]]:
        """Returns exercise session rows for the given calendar day, newest first."""
        normalized = self._normalize_date(date)
        with self._cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    id,
                    log_date,
                    category,
                    duration_minutes,
                    cardio_type,
                    strength_intensity,
                    burned_calories,
                    manual_entry,
                    created_at
                FROM exercise_logs
                WHERE user_id = %s AND log_date = %s
                ORDER BY id DESC
                """,
                (int(user_id), normalized),
            )
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    def delete_exercise_log(self, user_id: int, log_id: int) -> bool:
        """Deletes one log row and subtracts its kcal from daily_consumption. Returns False if missing."""
        if self.connection is None:
            raise RuntimeError("Database connection is not initialized.")

        with self.connection:
            with self._cursor() as cursor:
                cursor.execute(
                    """
                    SELECT log_date, burned_calories
                    FROM exercise_logs
                    WHERE id = %s AND user_id = %s
                    """,
                    (int(log_id), int(user_id)),
                )
                row = cursor.fetchone()
                if row is None:
                    return False
                normalized = str(row["log_date"])
                burned = float(row["burned_calories"])
                cursor.execute(
                    "DELETE FROM exercise_logs WHERE id = %s AND user_id = %s",
                    (int(log_id), int(user_id)),
                )
                cursor.execute(
                    """
                    UPDATE daily_consumption
                    SET
                        burned_calories = GREATEST(0, burned_calories - %s),
                        updated_at = CURRENT_TIMESTAMP
                    WHERE user_id = %s AND consumption_date = %s
                    """,
                    (burned, int(user_id), normalized),
                )
        return True

    def get_exercise_burn_history_by_day(
        self,
        user_id: int,
        start_date: str,
        end_date: str,
    ) -> list[dict[str, Any]]:
        """
        Per-day strength vs cardio burn in an inclusive date range.
        Missing days are filled with zeros (same pattern as get_advanced_calorie_history).
        """
        normalized_start = self._normalize_date(start_date)
        normalized_end = self._normalize_date(end_date)

        start_dt = datetime.strptime(normalized_start, "%Y-%m-%d").date()
        end_dt = datetime.strptime(normalized_end, "%Y-%m-%d").date()
        if end_dt < start_dt:
            raise ValueError("end_date must be on or after start_date.")

        with self._cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    log_date AS date,
                    COALESCE(
                        SUM(CASE WHEN category = 'strength' THEN burned_calories ELSE 0 END),
                        0
                    ) AS burned_strength,
                    COALESCE(
                        SUM(CASE WHEN category = 'cardio' THEN burned_calories ELSE 0 END),
                        0
                    ) AS burned_cardio
                FROM exercise_logs
                WHERE user_id = %s
                  AND log_date >= %s
                  AND log_date <= %s
                GROUP BY log_date
                ORDER BY log_date ASC
                """,
                (int(user_id), normalized_start, normalized_end),
            )
            rows = cursor.fetchall()

        by_date: dict[str, dict[str, float]] = {}
        for row in rows:
            d = str(row.get("date"))
            by_date[d] = {
                "burned_strength": float(row.get("burned_strength", 0.0)),
                "burned_cardio": float(row.get("burned_cardio", 0.0)),
            }

        result: list[dict[str, Any]] = []
        current_dt = start_dt
        while current_dt <= end_dt:
            date_text = current_dt.strftime("%Y-%m-%d")
            agg = by_date.get(date_text)
            if agg is None:
                result.append(
                    {
                        "date": date_text,
                        "burned_strength": 0.0,
                        "burned_cardio": 0.0,
                    }
                )
            else:
                result.append(
                    {
                        "date": date_text,
                        "burned_strength": agg["burned_strength"],
                        "burned_cardio": agg["burned_cardio"],
                    }
                )
            current_dt = current_dt + timedelta(days=1)

        return result

    # ------------------------------------------------------------------
    # Recent scans
    # ------------------------------------------------------------------
    def save_recent_scan(self, user_id: int, image_path: str, detected_barcodes: list) -> None:
        """Stores one scan session with image path and detected barcode list."""
        if self.connection is None:
            raise RuntimeError("Database connection is not initialized.")

        scan_date = datetime.now().isoformat(timespec="seconds")
        barcode_text = json.dumps(detected_barcodes or [], ensure_ascii=True)

        with self.connection:
            with self._cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO recent_scans (user_id, scan_date, image_path, detected_barcodes)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (int(user_id), scan_date, image_path, barcode_text),
                )

    def get_latest_scan(self, user_id: int) -> dict | None:
        """Returns the most recent scan row or None if empty."""
        with self._cursor() as cursor:
            cursor.execute(
                """
                SELECT id, user_id, scan_date, image_path, detected_barcodes
                FROM recent_scans
                WHERE user_id = %s
                ORDER BY id DESC
                LIMIT 1
                """,
                (int(user_id),),
            )
            row = cursor.fetchone()
            if row is None:
                return None

            latest = dict(row)
            latest["detected_barcodes"] = self._parse_json_list(
                latest.get("detected_barcodes")
            )
            return latest

    def get_weight_history(self, user_id: int) -> list[dict]:
        """Returns all weight records sorted by date ascending."""
        with self._cursor() as cursor:
            cursor.execute(
                """
                SELECT date, weight_kg
                FROM weight_history
                WHERE user_id = %s
                ORDER BY date ASC
                """,
                (int(user_id),),
            )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_calorie_history(self, user_id: int, days: int = 30) -> list[dict]:
        """Returns consumption and burned calories for the last `days` days."""
        if days < 0:
            raise ValueError("days must be >= 0")

        start_date = (datetime.now().date() - timedelta(days=int(days))).strftime("%Y-%m-%d")

        with self._cursor() as cursor:
            cursor.execute(
                """
                SELECT consumption_date AS date, consumed_calories, burned_calories
                FROM daily_consumption
                WHERE user_id = %s
                  AND consumption_date >= %s
                ORDER BY consumption_date ASC
                """,
                (int(user_id), start_date),
            )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def close(self) -> None:
        """Closes database connection."""
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def __enter__(self) -> "DatabaseManager":
        db_url = os.environ.get("DATABASE_URL")
        if not db_url:
            raise RuntimeError("DATABASE_URL environment variable is required.")

        # Migrations are not run here: per-request ALTER TABLE caused deadlocks vs concurrent DML.
        self.ensure_schema_once(db_url)
        self.connection = psycopg2.connect(db_url)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _ensure_daily_row(self, user_id: int, date_text: str) -> None:
        if self.connection is None:
            raise RuntimeError("Database connection is not initialized.")
        delay = 0.03
        for attempt in range(5):
            try:
                with self.connection:
                    with self._cursor() as cursor:
                        cursor.execute(
                            """
                            INSERT INTO daily_consumption (user_id, consumption_date)
                            VALUES (%s, %s)
                            ON CONFLICT (user_id, consumption_date) DO NOTHING
                            """,
                            (int(user_id), date_text),
                        )
                return
            except psycopg2.errors.DeadlockDetected:
                if attempt == 4:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 0.5)

    @staticmethod
    def _normalize_date(date_text: str) -> str:
        if not date_text:
            return datetime.now().strftime("%Y-%m-%d")
        try:
            parsed = datetime.strptime(date_text, "%Y-%m-%d")
            return parsed.strftime("%Y-%m-%d")
        except ValueError as error:
            raise ValueError("date must have format YYYY-MM-DD") from error

    @staticmethod
    def _parse_json_list(value: Any) -> list:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            raw = value.strip()
            if raw == "":
                return []
            try:
                parsed = json.loads(raw)
                return parsed if isinstance(parsed, list) else []
            except json.JSONDecodeError:
                return []
        return []

