export interface MacroSnapshot {
  calories: number
  protein_g: number
  carbs_g: number
  fat_g: number
}

export type IngredientSource = "available" | "to_buy" | "nutritionist_addition"

export interface RecipeIngredient {
  name: string
  quantity_g: number
  source: IngredientSource
}

export interface MacroBreakdown {
  total_kcal: number
  protein_g: number
  carb_g: number
  fat_g: number
  accuracy_note?: string
  accuracy_exact_count?: number
  accuracy_estimated_count?: number
  accuracy_note_key?: 'portion_from_catalog' | 'portion_estimated'
}

export interface ConsumedRecipe {
  title: string
  estimated_time_minutes?: number
  justification?: string
  justification_key?: 'manual_product_portion'
  ingredients?: RecipeIngredient[]
  preparation_steps?: string[]
  caloric_note?: string
  macro_breakdown?: MacroBreakdown
}

export interface ConsumedSnapshot extends MacroSnapshot {
  burned_calories: number
  /** kcal from logged strength sessions (sum of exercise_logs); remainder may be legacy burn without breakdown */
  burned_calories_strength: number
  /** kcal from logged cardio sessions */
  burned_calories_cardio: number
  recipes: ConsumedRecipe[]
}

export interface SummaryResponse {
  date: string
  targets: MacroSnapshot
  consumed: ConsumedSnapshot
  remaining: MacroSnapshot
  profile_found: boolean
}
