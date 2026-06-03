/** Canonical EU-14 allergen IDs (English snake_case); must match backend `ALLOWED_ALLERGY_IDS`. */
export const ALLERGY_PRESET_IDS = [
  'gluten',
  'crustaceans',
  'eggs',
  'fish',
  'peanuts',
  'soy',
  'milk',
  'tree_nuts',
  'celery',
  'mustard',
  'sesame',
  'sulfites',
  'lupin',
  'molluscs',
] as const

export type AllergyPresetId = (typeof ALLERGY_PRESET_IDS)[number]
