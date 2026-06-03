from __future__ import annotations

import math
from typing import Optional, Tuple


def round_half_up(value: float) -> int:
    """Deterministic round-half-up (avoids banker's rounding)."""
    return math.floor(float(value) + 0.5)


def calculate_dynamic_targets(
    *,
    age: int,
    gender: str,
    weight: float,
    height: float,
    activity_level: str,
    physical_goal: str,
    weight_goal_rate: float,
) -> Tuple[int, int, int, int]:
    """
    Backend nutrition target calculation

    Returns:
      (target_calories, protein_g, carb_g, fat_g)
    """
    gender_value = gender.strip().lower()
    if gender_value in ("male", "m", "man", "hombre"):
        bmr = (10 * weight) + (6.25 * height) - (5 * age) + 5
    elif gender_value in ("female", "f", "woman", "mujer"):
        bmr = (10 * weight) + (6.25 * height) - (5 * age) - 161
    else:
        raise ValueError("gender must be either male or female.")

    activity_value = activity_level.strip().lower()
    # NEAT-only multipliers (lifestyle / job movement). Not full TDEE activity tiers.
    neat_multipliers: dict[str, float] = {
        "sedentary": 1.2,  # desk / minimal daily movement
        "lightly_active": 1.3,  # mostly standing / light daily movement
        "moderately_active": 1.4,  # walking / moderate on-feet work
        "very_active": 1.5,  # heavy manual labour / very high NEAT
        "extremely_active": 1.5,  # same ceiling as very_active; caps lifestyle load
    }
    if activity_value not in neat_multipliers:
        raise ValueError(
            "activity_level must be one of: sedentary, lightly_active, "
            "moderately_active, very_active, extremely_active."
        )

    neat_baseline_kcal = bmr * neat_multipliers[activity_value]
    # ~7700 kcal per kg of body-mass change; spread across the week as a daily delta.
    daily_weight_delta_kcal = (weight_goal_rate * 7700.0) / 7.0
    adjusted_target = neat_baseline_kcal + daily_weight_delta_kcal
    target_calories = max(1200, round_half_up(adjusted_target))

    goal_value = physical_goal.strip().lower()
    if goal_value in ("weight_loss", "loss", "lose"):
        protein_pct, carb_pct, fat_pct = 30, 40, 30
    elif goal_value in ("muscle_gain", "gain", "bulk"):
        protein_pct, carb_pct, fat_pct = 30, 45, 25
    else:
        protein_pct, carb_pct, fat_pct = 25, 50, 25

    protein_g = round_half_up((target_calories * protein_pct / 100.0) / 4.0)
    carb_g = round_half_up((target_calories * carb_pct / 100.0) / 4.0)
    fat_g = round_half_up((target_calories * fat_pct / 100.0) / 9.0)

    return target_calories, protein_g, carb_g, fat_g


def compute_meal_targets_from_daily_summary(
    *,
    meals_per_day: int,
    meals_logged_today: int,
    remaining: dict[str, float],
    min_meal_kcal: int = 300,
) -> dict[str, int]:
    """
    Calculates the nutritional budget for the current meal based on remaining daily targets.
    """
    meals_remaining = max(1, meals_per_day - meals_logged_today)

    meal_kcal = max(
        min_meal_kcal, 
        round_half_up(remaining.get("calories", 0.0) / meals_remaining)
    )
    meal_protein_g = max(
        0, 
        round_half_up(remaining.get("protein_g", 0.0) / meals_remaining)
    )
    meal_carb_g = max(
        0, 
        round_half_up(remaining.get("carbs_g", 0.0) / meals_remaining)
    )
    meal_fat_g = max(
        0, 
        round_half_up(remaining.get("fat_g", 0.0) / meals_remaining)
    )

    return {
        "meal_caloric_target": meal_kcal,
        "meal_protein_g": meal_protein_g,
        "meal_carb_g": meal_carb_g,
        "meal_fat_g": meal_fat_g,
        "meals_remaining": meals_remaining,
    }