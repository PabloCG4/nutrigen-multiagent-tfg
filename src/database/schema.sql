-- ------------------------------------------------------------------
-- Users Table
-- ------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL DEFAULT '',
    hashed_password TEXT NOT NULL,
    password_credential BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- ------------------------------------------------------------------
-- User Profiles Table
-- ------------------------------------------------------------------
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
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_user_profiles_user_id FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- ------------------------------------------------------------------
-- Daily Consumption Table
-- ------------------------------------------------------------------
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
    PRIMARY KEY (user_id, consumption_date),
    CONSTRAINT fk_daily_consumption_user_id FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- ------------------------------------------------------------------
-- Weight History Table
-- ------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS weight_history (
    user_id INTEGER NOT NULL,
    date TEXT NOT NULL,
    weight_kg REAL NOT NULL,
    PRIMARY KEY (user_id, date),
    CONSTRAINT fk_weight_history_user_id FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- ------------------------------------------------------------------
-- Recent Scans Table
-- ------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS recent_scans (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL DEFAULT 1,
    scan_date TEXT NOT NULL,
    image_path TEXT NOT NULL,
    detected_barcodes TEXT NOT NULL,
    CONSTRAINT fk_recent_scans_user_id FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- ------------------------------------------------------------------
-- Exercise Logs Table
-- ------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS exercise_logs (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    log_date TEXT NOT NULL,
    category TEXT NOT NULL,
    duration_minutes INTEGER NOT NULL,
    cardio_type TEXT,
    strength_intensity TEXT,
    burned_calories REAL NOT NULL,
    manual_entry BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_exercise_logs_user_id FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- ------------------------------------------------------------------
-- Indexes
-- ------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_exercise_logs_user_date ON exercise_logs (user_id, log_date);