import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import i18next from 'i18next'
import { getApiBaseUrl } from '../config/apiBaseUrl'
import { apiFetch } from '../api/apiFetch'
import { processVisionImageWithPolling } from '../api/vision'
import { useAuth } from '../context/AuthContext'
import {
  decodeJwtSub,
  encodeStorageScopeId,
  getMenuJobKey,
  getMenuSessionKey,
  getRecentProductsKey,
  removeLegacyRecipesStorageKeys,
} from '../utils/recipesStorageScope'
import { filterAndSortRecentMatches } from '../utils/productTextNormalize'
import { Pin, Search } from 'lucide-react'
import { DeleteButton } from '../components/DeleteButton'

/**
 * Recipes page: build a verified product list (camera / gallery / autocomplete or last scan),
 * generate menu options via an async backend job, optionally refine macros, log one recipe to
 * the user’s day (Dashboard/summary), or estimate a cheat meal.
 *
 * System ties: AuthContext + Bearer token, apiFetch/getApiBaseUrl, i18n (recipesPage.*, errors.*).
 * APIs: /api/vision/*, /api/products/autocomplete, /api/products/macros-by-barcodes,
 * /api/menu-jobs, /api/generate-menu/macros, /api/consume, /api/consume/portioned-product,
 * /api/cheat-meal.
 */

// --- Domain types & DTOs (mirror backend JSON where applicable) ---

type DishType =
  | 'soup_stew'
  | 'salad_side'
  | 'main_course'
  | 'dessert'
  | 'snack_breakfast'
  | 'drink'

type DietaryStyle =
  | 'vegan'
  | 'vegetarian'
  | 'celiac'
  | 'mediterranean'
  | 'asian'
  | 'arabic'
  | 'latin'

type ProductFlow = 'scan' | 'previous' | 'cheat' | 'manual_add'

interface VisionAlternative {
  name: string
  barcode: string
}

interface VerifiedProduct {
  barcode: string
  name: string
}

interface ProductAutocompleteSuggestion {
  name: string
  barcode: string
  kcal_per_100g?: number | null
  protein_g_per_100g?: number | null
  fat_g_per_100g?: number | null
  carbs_g_per_100g?: number | null
}

// --- Small UI helpers (no React state) ---

/** Pretty-print macro numbers for autocomplete rows (drop trailing “.0”). */
function formatMacroDisplayValue(value: number): string {
  if (Number.isInteger(value)) {
    return String(value)
  }
  return value.toFixed(1).replace(/\.0$/, '')
}

/** Per-100g macros as compact pills (white on token-colored circles) for autocomplete rows. */
function AutocompleteMacroLine({ suggestion }: { suggestion: ProductAutocompleteSuggestion }): ReactNode {
  const { t } = useTranslation()
  const kcal = suggestion.kcal_per_100g
  const p = suggestion.protein_g_per_100g
  const c = suggestion.carbs_g_per_100g
  const f = suggestion.fat_g_per_100g

  const hasAny =
    (kcal != null && !Number.isNaN(kcal)) ||
    (p != null && !Number.isNaN(p)) ||
    (c != null && !Number.isNaN(c)) ||
    (f != null && !Number.isNaN(f))

  if (!hasAny) {
    return null
  }

  const pill =
    'inline-flex min-h-7 min-w-7 shrink-0 items-center justify-center rounded-full px-2 py-1 text-[10px] font-semibold leading-none text-white'

  const segments: ReactNode[] = []
  if (kcal != null && !Number.isNaN(kcal)) {
    segments.push(
      <span key="kcal" className={`${pill} bg-[var(--color-secondary)]`}>
        {formatMacroDisplayValue(kcal)} kcal
      </span>,
    )
  }
  if (p != null && !Number.isNaN(p)) {
    segments.push(
      <span key="p" className={`${pill} bg-[var(--color-protein)]`}>
        P {formatMacroDisplayValue(p)}g
      </span>,
    )
  }
  if (c != null && !Number.isNaN(c)) {
    segments.push(
      <span key="c" className={`${pill} bg-[var(--color-carbs)]`}>
        C {formatMacroDisplayValue(c)}g
      </span>,
    )
  }
  if (f != null && !Number.isNaN(f)) {
    segments.push(
      <span key="f" className={`${pill} bg-[var(--color-fat)]`}>
        F {formatMacroDisplayValue(f)}g
      </span>,
    )
  }

  return (
    <div
      className="mt-1 flex flex-wrap items-center gap-1.5"
      title={t('recipesPage.nutritionPer100g', 'Nutrition per 100 g')}
    >
      {segments}
    </div>
  )
}

const productSearchRowButtonClass =
  'flex w-full items-start justify-between gap-4 px-4 py-3 text-left text-sm transition hover:bg-[var(--color-surface-soft)]'

/** Shared autocomplete dropdown: full recent list (empty query) or recent matches + catalog (typed). */
function ProductSearchDropdown({
  variant,
  allRecentProducts,
  matchingRecents,
  catalogSuggestions,
  isSearching,
  onPick,
  listKeyPrefix,
  maxHeightClass,
}: {
  variant: 'recent-only' | 'typed-search'
  allRecentProducts: ProductAutocompleteSuggestion[]
  matchingRecents: ProductAutocompleteSuggestion[]
  catalogSuggestions: ProductAutocompleteSuggestion[]
  isSearching: boolean
  onPick: (s: ProductAutocompleteSuggestion) => void
  listKeyPrefix: string
  maxHeightClass: string
}): ReactNode {
  const { t } = useTranslation()

  const renderRow = (s: ProductAutocompleteSuggestion, keySuffix: string): ReactNode => (
    <li key={`${listKeyPrefix}-${keySuffix}-${s.barcode}`}>
      <button
        type="button"
        className={productSearchRowButtonClass}
        onMouseDown={(event) => event.preventDefault()}
        onClick={() => onPick(s)}
      >
        <div className="min-w-0 flex-1">
          <div className="font-semibold text-[var(--color-primary)]">{s.name}</div>
          <AutocompleteMacroLine suggestion={s} />
        </div>
        <span className="shrink-0 whitespace-nowrap text-xs font-semibold text-[var(--color-secondary)]">
          {s.barcode}
        </span>
      </button>
    </li>
  )

  if (variant === 'recent-only') {
    return (
      <div className="absolute z-20 mt-2 w-full overflow-hidden rounded-2xl border border-[var(--color-border)] bg-white shadow-lg">
        <div>
          <div className="border-b border-slate-100 px-4 py-2 text-[10px] font-bold uppercase tracking-wide text-[var(--color-primary)]">
            {t('recipesPage.recentProducts')}
          </div>
          <ul className={`${maxHeightClass} overflow-y-auto`}>
            {allRecentProducts.map((s) => renderRow(s, 'recent'))}
          </ul>
        </div>
      </div>
    )
  }

  const showSearchingOnly = isSearching && matchingRecents.length === 0 && catalogSuggestions.length === 0

  return (
    <div className="absolute z-20 mt-2 w-full overflow-hidden rounded-2xl border border-[var(--color-border)] bg-white shadow-lg">
      {showSearchingOnly ? (
        <div className="px-4 py-3 text-sm font-semibold text-[var(--color-secondary)]">
          {t('recipesPage.searching')}
        </div>
      ) : (
        <div className={`${maxHeightClass} overflow-y-auto`}>
          {matchingRecents.length > 0 ? (
            <div>
              <div className="border-b border-slate-100 px-4 py-2 text-[10px] font-bold uppercase tracking-wide text-[var(--color-primary)]">
                {t('recipesPage.recentProducts')}
              </div>
              <ul>{matchingRecents.map((s) => renderRow(s, 'match'))}</ul>
            </div>
          ) : null}
          {catalogSuggestions.length > 0 ? (
            <div className={matchingRecents.length > 0 ? 'border-t border-slate-100' : ''}>
              <div className="border-b border-slate-100 px-4 py-2 text-[10px] font-bold uppercase tracking-wide text-[var(--color-primary)]">
                {t('recipesPage.catalogSuggestions')}
              </div>
              <ul>{catalogSuggestions.map((s) => renderRow(s, 'cat'))}</ul>
            </div>
          ) : null}
        </div>
      )}
    </div>
  )
}

/** Macros row for a vision card when batch lookup returned data for this barcode. */
function VisionItemMacroLine({
  barcode,
  macrosByBarcode,
}: {
  barcode: string
  macrosByBarcode: Record<string, ProductAutocompleteSuggestion>
}): ReactNode {
  const key = barcode.trim()
  if (key === '') return null
  const snap = macrosByBarcode[key]
  if (!snap) return null
  return <AutocompleteMacroLine suggestion={snap} />
}

/** Product card in scan/previous flows: change, priority pin, delete. */
function VisionProductCard({
  item,
  isPriority,
  macrosByBarcode,
  onOpenAlternative,
  onTogglePriority,
  onDelete,
}: {
  item: RecipesFromVisionItem
  isPriority: boolean
  macrosByBarcode: Record<string, ProductAutocompleteSuggestion>
  onOpenAlternative: () => void
  onTogglePriority: () => void
  onDelete: () => void
}): ReactNode {
  const { t } = useTranslation()

  return (
    <div
      className={`group relative rounded-2xl border p-4 shadow-sm transition ${isPriority
        ? 'border-[var(--color-active-border)] bg-[var(--color-active-bg)]'
        : 'border-[var(--color-border)] bg-white hover:border-[var(--color-active-border)]'
        }`}
    >
      <div className="flex items-start justify-between gap-3">
        <button type="button" onClick={onOpenAlternative} className="min-w-0 flex-1 text-left">
          <div className="text-sm font-semibold text-[var(--color-primary)]">
            {item.selectedName || t('recipesPage.unknown')}
          </div>
          <VisionItemMacroLine barcode={item.selectedBarcode} macrosByBarcode={macrosByBarcode} />
          <div className="mt-2 text-xs font-semibold text-[var(--color-secondary)] underline decoration-dashed decoration-[var(--color-secondary)]">
            {t('recipesPage.clickToChange')}
          </div>
        </button>

        <div className="flex shrink-0 flex-col items-end gap-2">
          <button
            type="button"
            onClick={onTogglePriority}
            title={t('recipesPage.priorityIngredientAria')}
            aria-label={t('recipesPage.priorityIngredientAria')}
            aria-pressed={isPriority}
            className={`inline-flex items-center gap-1 rounded-lg border px-2 py-1.5 text-[10px] font-bold uppercase tracking-wide transition ${isPriority
              ? 'border-[var(--color-active-border)] bg-white text-[var(--color-secondary)]'
              : 'border-[var(--color-border)] bg-[var(--color-surface)] text-[var(--color-secondary)] hover:border-[var(--color-active-border)]'
              }`}
          >
            <Pin className={`h-3.5 w-3.5 ${isPriority ? 'fill-current' : ''}`} aria-hidden />
            <span className="hidden sm:inline">{t('recipesPage.priorityIngredient')}</span>
          </button>
          <DeleteButton onClick={onDelete} aria-label={t('recipesPage.delete')}>
            {t('recipesPage.delete')}
          </DeleteButton>
        </div>
      </div>
    </div>
  )
}

interface ProductAutocompleteResponse {
  suggestions: ProductAutocompleteSuggestion[]
}

interface ProductMacrosBatchResponse {
  macros: Record<string, ProductAutocompleteSuggestion>
}

interface MacroBreakdown {
  total_kcal: number
  protein_g: number
  carb_g: number
  fat_g: number
  accuracy_note?: string
  accuracy_exact_count?: number
  accuracy_estimated_count?: number
  accuracy_note_key?: 'portion_from_catalog' | 'portion_estimated'
}

type IngredientSource = 'available' | 'to_buy' | 'nutritionist_addition'

interface FinalIngredient {
  name: string
  quantity_g: number
  source: IngredientSource
}

interface FinalRecipe {
  title: string
  estimated_time_minutes?: number
  justification?: string
  justification_key?: 'manual_product_portion'
  caloric_note?: string
  macro_breakdown?: MacroBreakdown
  ingredients?: FinalIngredient[]
  preparation_steps?: string[]
}

interface CheatMealResponse {
  status: string
  message: string
  data: {
    description: string
    estimated_kcal: number
    estimated_protein_g: number
    estimated_carb_g: number
    estimated_fat_g: number
    estimation_notes: string
  }
}

interface RecipesFromVisionItem {
  id: string
  detectedName: string
  detectedBarcode: string
  alternatives: VisionAlternative[]
  selectedName: string
  selectedBarcode: string
}

type NutritionMetric = 'kcal' | 'protein' | 'carbs' | 'fat'
type AlternativeModalSource = 'scan' | 'previous'
type PersistedRecipesFlow = 'scan' | 'previous'

interface PersistedMenuSession {
  savedAt: number
  flow: PersistedRecipesFlow
  visionItems: RecipesFromVisionItem[]
  recipes: FinalRecipe[]
  priorityVisionItemId?: string | null
}

// localStorage: per-account keys via `recipesStorageScope` (JWT sub). TTL so stale scans/jobs do not linger across days.
const MENU_SESSION_TTL_MS = 2 * 60 * 60 * 1000
const RECENT_PRODUCTS_MAX = 10

type MenuJobState = 'queued' | 'running' | 'done' | 'error'

interface PersistedMenuJob {
  jobId: string
  savedAt: number
  progress: number
  state: MenuJobState
}

/** YYYY-MM-DD for /api/consume (same convention as Exercise/Dashboard date fields). */
function formatDateISO(date: Date): string {
  const year = date.getFullYear()
  const month = String(date.getMonth() + 1).padStart(2, '0')
  const day = String(date.getDate()).padStart(2, '0')
  return `${year}-${month}-${day}`
}

function getTodayISO(): string {
  return formatDateISO(new Date())
}

// Parse recent products from localStorage (scoped key).
function parseRecentProductsFromStorage(storageKey: string): ProductAutocompleteSuggestion[] {
  try {
    const raw = localStorage.getItem(storageKey)
    if (!raw) return []
    const parsed: unknown = JSON.parse(raw)
    if (!Array.isArray(parsed)) return []
    const out: ProductAutocompleteSuggestion[] = []
    for (const row of parsed) {
      if (typeof row !== 'object' || row === null) continue
      const r = row as Record<string, unknown>
      const name = typeof r.name === 'string' ? r.name.trim() : ''
      const barcode = typeof r.barcode === 'string' ? r.barcode.trim() : ''
      if (name === '' || barcode === '') continue
      out.push({
        name,
        barcode,
        kcal_per_100g: typeof r.kcal_per_100g === 'number' ? r.kcal_per_100g : null,
        protein_g_per_100g: typeof r.protein_g_per_100g === 'number' ? r.protein_g_per_100g : null,
        fat_g_per_100g: typeof r.fat_g_per_100g === 'number' ? r.fat_g_per_100g : null,
        carbs_g_per_100g: typeof r.carbs_g_per_100g === 'number' ? r.carbs_g_per_100g : null,
      })
    }
    return out.slice(0, RECENT_PRODUCTS_MAX)
  } catch {
    return []
  }
}

function persistRecentProducts(list: ProductAutocompleteSuggestion[], storageKey: string | null): void {
  if (!storageKey) return
  try {
    localStorage.setItem(storageKey, JSON.stringify(list.slice(0, RECENT_PRODUCTS_MAX)))
  } catch {
    // Ignore quota errors.
  }
}

// Detect expired/invalid JWT from API error text; triggers logout like other pages.
function isAuthFailureMessage(message: string): boolean {
  const trimmed = message.trim()
  if (trimmed === 'Could not validate credentials.') return true
  const lower = trimmed.toLowerCase()
  return lower.includes('could not validate credentials') || trimmed.includes('HTTP 401')
}

// Best-effort FastAPI `{ "detail": "..." }` (or similar) extraction for user-facing errors.
async function readErrorDetail(response: Response, fallback: string): Promise<string> {
  try {
    const payload: unknown = await response.json()
    if (
      typeof payload === 'object' &&
      payload !== null &&
      'detail' in payload &&
      typeof (payload as { detail: unknown }).detail === 'string'
    ) {
      return (payload as { detail: string }).detail
    }
  } catch {
    // Ignore parsing errors.
  }
  return fallback
}

// Labels for macro chips; reuses dashboard/history i18n keys for consistency.
function metricLabel(metric: NutritionMetric): string {
  if (metric === 'kcal') return i18next.t('dashboard.caloriesLabel')
  if (metric === 'protein') return i18next.t('history.protein')
  if (metric === 'carbs') return i18next.t('history.carbs')
  return i18next.t('history.fat')
}

// Read a single macro from a generated recipe’s optional macro_breakdown.
function metricValue(recipe: FinalRecipe, metric: NutritionMetric): number | null {
  const m = recipe.macro_breakdown
  if (!m) return null
  if (metric === 'kcal') return m.total_kcal
  if (metric === 'protein') return m.protein_g
  if (metric === 'carbs') return m.carb_g
  return m.fat_g
}

function metricUnit(metric: NutritionMetric): string {
  if (metric === 'kcal') return 'kcal'
  return 'g'
}

// Tailwind class bundles for the small macro summary tiles on each recipe card.
function metricColor(metric: NutritionMetric): string {
  if (metric === 'kcal') return 'bg-[var(--color-secondary)] text-white'
  if (metric === 'protein') return 'bg-[var(--color-protein)] text-white'
  if (metric === 'carbs') return 'bg-[var(--color-carbs)] text-white'
  return 'bg-[var(--color-fat)] text-white'
}

function macroAccuracyLine(m: MacroBreakdown | undefined, t: (key: string, opts?: Record<string, unknown>) => string): string {
  if (!m) return ''
  if (typeof m.accuracy_exact_count === 'number' && typeof m.accuracy_estimated_count === 'number') {
    return t('recipesPage.macroAccuracyNote', {
      exact: m.accuracy_exact_count,
      estimated: m.accuracy_estimated_count,
    })
  }
  if (m.accuracy_note_key === 'portion_from_catalog') return t('recipesPage.portionAccuracyFromCatalog')
  if (m.accuracy_note_key === 'portion_estimated') return t('recipesPage.portionAccuracyEstimated')
  return m.accuracy_note ?? ''
}

function ingredientSourceLabel(source: string, t: (key: string, opts?: Record<string, unknown>) => string): string {
  return t(`recipesPage.ingredientSource.${source}`, { defaultValue: source })
}

// =============================================================================
// Main page component
// =============================================================================

export function Recipes() {
  const navigate = useNavigate()
  const location = useLocation()
  const { isAuthenticated, token, logout } = useAuth()
  const { t, i18n } = useTranslation()
  const macrosComputeInFlightRef = useRef<boolean>(false)

  const storageScopeId = useMemo(() => {
    const sub = decodeJwtSub(token)
    return sub ? encodeStorageScopeId(sub) : null
  }, [token])

  const menuSessionKey = storageScopeId ? getMenuSessionKey(storageScopeId) : null
  const menuJobKey = storageScopeId ? getMenuJobKey(storageScopeId) : null
  const recentProductsKey = storageScopeId ? getRecentProductsKey(storageScopeId) : null

  /** Backend catalog display language for autocomplete / macros (es | en). */
  const apiLang = useMemo(() => {
    const l = (i18n.language ?? 'es').toLowerCase()
    return l.startsWith('en') ? 'en' : 'es'
  }, [i18n.language])

  /** Last known scoped job key; keep non-null after logout so we can remove persisted job on sign-out. */
  const lastMenuJobKeyRef = useRef<string | null>(null)
  useEffect(() => {
    if (menuJobKey) {
      lastMenuJobKeyRef.current = menuJobKey
    }
  }, [menuJobKey])

  const legacyRecipesKeysRemovedRef = useRef(false)
  useEffect(() => {
    if (!token || legacyRecipesKeysRemovedRef.current) return
    legacyRecipesKeysRemovedRef.current = true
    removeLegacyRecipesStorageKeys()
  }, [token])

  // --- State: high-level flow + vision pipeline + product list ---
  const [flow, setFlow] = useState<ProductFlow>('scan')

  const [isVisionProcessing, setIsVisionProcessing] = useState<boolean>(false)
  const [visionError, setVisionError] = useState<string | null>(null)
  const [visionItems, setVisionItems] = useState<RecipesFromVisionItem[]>([])
  const [priorityVisionItemId, setPriorityVisionItemId] = useState<string | null>(null)

  const [isLoadingPrevious, setIsLoadingPrevious] = useState<boolean>(false)
  const [previousError, setPreviousError] = useState<string | null>(null)

  // --- Modal: pick another barcode/name for one vision row (scan or previous flow) ---
  const [activeAlternativeIndex, setActiveAlternativeIndex] = useState<number | null>(null)
  const [activeAlternativeSource, setActiveAlternativeSource] = useState<AlternativeModalSource>('scan')
  const [alternativeSelectionIndex, setAlternativeSelectionIndex] = useState<number>(0)
  const [modalSearchQuery, setModalSearchQuery] = useState<string>('')
  const [modalSuggestions, setModalSuggestions] = useState<ProductAutocompleteSuggestion[]>([])
  const [isModalSearching, setIsModalSearching] = useState<boolean>(false)
  const [modalSearchError, setModalSearchError] = useState<string | null>(null)

  // --- Async menu job: confirm barcodes → POST job → poll until done ---
  const [isConfirming, setIsConfirming] = useState<boolean>(false)
  const [isGeneratingMenu, setIsGeneratingMenu] = useState<boolean>(false)
  const [generationError, setGenerationError] = useState<string | null>(null)
  const [menuGenerationProgress, setMenuGenerationProgress] = useState<number>(0)

  const [recipes, setRecipes] = useState<FinalRecipe[]>([])
  const [consumeBusyIndex, setConsumeBusyIndex] = useState<number | null>(null)
  const [openRecipeStepsIndex, setOpenRecipeStepsIndex] = useState<number | null>(null)

  // --- Main-page manual product search (same API as modal, different state) ---
  const [manualSearchQuery, setManualSearchQuery] = useState<string>('')
  const [manualSuggestions, setManualSuggestions] = useState<ProductAutocompleteSuggestion[]>([])
  const [isManualSearching, setIsManualSearching] = useState<boolean>(false)
  const [manualSearchError, setManualSearchError] = useState<string | null>(null)

  // --- Menu generation form (sent with job create) ---
  const [diners, setDiners] = useState<string>('2')
  const [timeAvailable, setTimeAvailable] = useState<string>('30')
  const [dishType, setDishType] = useState<DishType>('main_course')
  const [dietaryStyle, setDietaryStyle] = useState<DietaryStyle | 'none'>('none')
  const [specialRequests, setSpecialRequests] = useState<string>('')

  // --- Cheat meal branch (separate API; does not use vision items) ---
  const [isCheatSubmitting, setIsCheatSubmitting] = useState<boolean>(false)
  const [cheatDescription, setCheatDescription] = useState<string>('')
  const [cheatError, setCheatError] = useState<string | null>(null)
  const [cheatResult, setCheatResult] = useState<CheatMealResponse['data'] | null>(null)
  const [hasHydratedMenuSession, setHasHydratedMenuSession] = useState<boolean>(false)
  const jobPollIntervalRef = useRef<number | null>(null)
  const manualSearchRootRef = useRef<HTMLDivElement | null>(null)
  const modalAutocompleteRootRef = useRef<HTMLDivElement | null>(null)
  const manualAddSearchRootRef = useRef<HTMLDivElement | null>(null)
  const manualSearchSeqRef = useRef(0)
  const modalSearchSeqRef = useRef(0)
  const manualAddSearchSeqRef = useRef(0)
  const prevStorageScopeIdRef = useRef<string | null | undefined>(undefined)

  const [macrosByBarcode, setMacrosByBarcode] = useState<Record<string, ProductAutocompleteSuggestion>>({})
  const [recentProductPicks, setRecentProductPicks] = useState<ProductAutocompleteSuggestion[]>([])

  const [mainSearchInputFocused, setMainSearchInputFocused] = useState(false)
  const [modalSearchInputFocused, setModalSearchInputFocused] = useState(false)
  const [manualAddSearchInputFocused, setManualAddSearchInputFocused] = useState(false)

  const [manualAddQuery, setManualAddQuery] = useState('')
  const [manualAddSuggestions, setManualAddSuggestions] = useState<ProductAutocompleteSuggestion[]>([])
  const [isManualAddSearching, setIsManualAddSearching] = useState(false)
  const [manualAddSearchError, setManualAddSearchError] = useState<string | null>(null)

  const [portionPick, setPortionPick] = useState<ProductAutocompleteSuggestion | null>(null)
  const [portionGrams, setPortionGrams] = useState<string>('100')
  const [isPortionSubmitting, setIsPortionSubmitting] = useState(false)
  const [portionError, setPortionError] = useState<string | null>(null)
  const [portionSuccessMessage, setPortionSuccessMessage] = useState<string | null>(null)

  // Push a recent product to the local storage.
  const pushRecentProduct = useCallback((suggestion: ProductAutocompleteSuggestion): void => {
    const barcode = suggestion.barcode.trim()
    const name = suggestion.name.trim()
    if (barcode === '' || name === '') return
    setRecentProductPicks((prev) => {
      const next = [
        {
          ...suggestion,
          name,
          barcode,
        },
        ...prev.filter((p) => p.barcode !== barcode),
      ].slice(0, RECENT_PRODUCTS_MAX)
      persistRecentProducts(next, recentProductsKey)
      return next
    })
  }, [recentProductsKey])

  // Generate a list of barcodes from the vision items.
  const visionBarcodeList = useMemo((): string[] => {
    const s = new Set<string>()
    for (const item of visionItems) {
      const a = item.selectedBarcode.trim()
      const b = item.detectedBarcode.trim()
      if (a !== '') s.add(a)
      if (b !== '') s.add(b)
      for (const alt of item.alternatives ?? []) {
        const c = alt.barcode.trim()
        if (c !== '') s.add(c)
      }
    }
    return [...s]
  }, [visionItems])

  // Generate a key for the vision barcode list.
  const visionBarcodeListKey = useMemo(() => [...visionBarcodeList].sort().join('|'), [visionBarcodeList])

  // Guard route: unauthenticated users never see this page (Navbar also hides most actions).
  useEffect(() => {
    if (!isAuthenticated || !token) {
      navigate('/login', { replace: true })
    }
  }, [isAuthenticated, token, navigate])

  // When the logged-in account changes, clear in-memory recipe session state before hydrating from that account's storage.
  useEffect(() => {
    if (!isAuthenticated || !token || !storageScopeId) {
      return
    }
    const prev = prevStorageScopeIdRef.current
    // Strict validation: only reset if both previous and current scopes are valid strings and do not match
    if (typeof prev === 'string' && typeof storageScopeId === 'string' && prev !== storageScopeId) {
      if (jobPollIntervalRef.current !== null) {
        window.clearInterval(jobPollIntervalRef.current)
        jobPollIntervalRef.current = null
      }
      setHasHydratedMenuSession(false)
      setVisionItems([])
      setRecipes([])
      setIsGeneratingMenu(false)
      setMenuGenerationProgress(0)
      setGenerationError(null)
      setMacrosByBarcode({})
    }
    prevStorageScopeIdRef.current = storageScopeId
  }, [isAuthenticated, token, storageScopeId])

  useEffect(() => {
    if (!recentProductsKey) {
      setRecentProductPicks([])
      return
    }
    setRecentProductPicks(parseRecentProductsFromStorage(recentProductsKey))
  }, [recentProductsKey])

  // Fetch the macros for the vision barcodes.
  useEffect(() => {
    if (!token) return
    if (visionBarcodeList.length === 0) {
      setMacrosByBarcode({})
      return
    }
    // Cancel the request if the component is unmounted.
    let cancelled = false
    void (async (): Promise<void> => {
      try {
        const response = await apiFetch(`${getApiBaseUrl()}/api/products/macros-by-barcodes`, {
          method: 'POST',
          headers: {
            Authorization: `Bearer ${token}`,
            'Content-Type': 'application/json',
          },
          // Send the list of barcodes to the API.
          body: JSON.stringify({ barcodes: visionBarcodeList, lang: apiLang }),
        })
        if (!response.ok) return
        const payload: unknown = await response.json()
        if (
          typeof payload !== 'object' ||
          payload === null ||
          !('macros' in payload) ||
          typeof (payload as ProductMacrosBatchResponse).macros !== 'object'
        ) {
          return
        }
        const macros = (payload as ProductMacrosBatchResponse).macros
        // If the user navigated away before the server responded,
        // 'cancelled' will be true. If so, abort and DO NOT update React state.
        // This prevents the error "Can't perform a React state update on an unmounted component."
        if (cancelled) return
        // Combine the previous macros with the new ones.
        setMacrosByBarcode((prev) => ({ ...prev, ...macros }))
      } catch {
        // Display-only enrichment; ignore failures.
      }
    })()
    // Cleanup function: If the component is destroyed or the list of barcodes changes, 
    // React executes this and marks 'cancelled' as true.
    return () => {
      cancelled = true
    }
  }, [token, visionBarcodeListKey, apiLang])

  // Debounced GET /api/products/autocomplete for the inline search (scan + previous tabs only).
  useEffect(() => {
    if (!token) return
    if (flow !== 'scan' && flow !== 'previous') return

    const query = manualSearchQuery.trim()
    // If the query is less than 2 characters, clear the suggestions and stop the search.
    if (query.length < 2) {
      manualSearchSeqRef.current += 1
      setManualSuggestions([])
      setManualSearchError(null)
      setIsManualSearching(false)
      return
    }

    manualSearchSeqRef.current += 1
    const seq = manualSearchSeqRef.current
    const queryCaptured = query

    setIsManualSearching(true)
    setManualSearchError(null)
    // Create a new AbortController to cancel the request if the user navigates away.
    const controller = new AbortController()
    // Debounce pattern: Wait 300 milliseconds before executing.
    // If the user types another letter before 300ms, this timer will be cancelled (see Cleanup final).
    const timeout = window.setTimeout(() => {
      const run = async (): Promise<void> => {
        try {
          const encoded = encodeURIComponent(queryCaptured)
          const response = await apiFetch(
            `${getApiBaseUrl()}/api/products/autocomplete?query=${encoded}&limit=10&lang=${encodeURIComponent(apiLang)}`,
            {
              method: 'GET',
              headers: { Authorization: `Bearer ${token}` },
              signal: controller.signal,
            },
          )

          if (!response.ok) {
            const detail = await readErrorDetail(
              response,
              `Failed to load suggestions (HTTP ${response.status}).`,
            )
            if (isAuthFailureMessage(detail)) {
              logout()
              return
            }
            throw new Error(detail)
          }

          const payload: unknown = await response.json()
          if (
            typeof payload !== 'object' ||
            payload === null ||
            !('suggestions' in payload) ||
            !Array.isArray((payload as { suggestions: unknown[] }).suggestions)
          ) {
            if (seq === manualSearchSeqRef.current) {
              setManualSuggestions([])
            }
            return
          }

          const parsed = payload as ProductAutocompleteResponse
          // 4. Ticket control (Race Condition Check)
          // Before drawing the results, we check: ¿My local ticket (#5)
          // is still the same as the global ticket (manualSearchSeqRef.current)?
          // If the user continued typing, the global ticket will now be #6.
          // Since #5 !== #6, we discard these old results by returning.
          if (seq !== manualSearchSeqRef.current) return
          setManualSuggestions(parsed.suggestions.slice(0, 10))
        } catch (error) {
          if (error instanceof Error && error.name === 'AbortError') return
          if (seq !== manualSearchSeqRef.current) return
          const detail =
            error instanceof Error ? error.message : 'Unexpected error while searching products.'
          setManualSearchError(detail)
          setManualSuggestions([])
        } finally {
          // Only set the searching state to false if the local ticket is still the same as the global ticket, is the most recent one
          if (seq === manualSearchSeqRef.current) {
            setIsManualSearching(false)
          }
        }
      }

      void run()
    }, 300)

    return () => {
      window.clearTimeout(timeout)
      controller.abort()
    }
  }, [manualSearchQuery, flow, token, logout, apiLang])

  // Automatically scroll to the bottom if requested by routing state once recipes are populated
  useEffect(() => {
    if (location.state?.scrollToBottom && recipes.length > 0) {
      window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' })
    }
  }, [location.state, recipes])

  // Debounced autocomplete for “add manually” flow (log portion to today).
  useEffect(() => {
    if (!token) return
    if (flow !== 'manual_add') return

    const query = manualAddQuery.trim()
    if (query.length < 2) {
      manualAddSearchSeqRef.current += 1
      setManualAddSuggestions([])
      setManualAddSearchError(null)
      setIsManualAddSearching(false)
      return
    }

    manualAddSearchSeqRef.current += 1
    const seq = manualAddSearchSeqRef.current
    const queryCaptured = query

    setIsManualAddSearching(true)
    setManualAddSearchError(null)

    const controller = new AbortController()
    const timeout = window.setTimeout(() => {
      const run = async (): Promise<void> => {
        try {
          const encoded = encodeURIComponent(queryCaptured)
          const response = await apiFetch(
            `${getApiBaseUrl()}/api/products/autocomplete?query=${encoded}&limit=10&lang=${encodeURIComponent(apiLang)}`,
            {
              method: 'GET',
              headers: { Authorization: `Bearer ${token}` },
              signal: controller.signal,
            },
          )

          if (!response.ok) {
            const detail = await readErrorDetail(
              response,
              `Failed to load suggestions (HTTP ${response.status}).`,
            )
            if (isAuthFailureMessage(detail)) {
              logout()
              return
            }
            throw new Error(detail)
          }

          const payload: unknown = await response.json()
          if (
            typeof payload !== 'object' ||
            payload === null ||
            !('suggestions' in payload) ||
            !Array.isArray((payload as { suggestions: unknown[] }).suggestions)
          ) {
            if (seq === manualAddSearchSeqRef.current) {
              setManualAddSuggestions([])
            }
            return
          }

          const parsed = payload as ProductAutocompleteResponse
          if (seq !== manualAddSearchSeqRef.current) return
          setManualAddSuggestions(parsed.suggestions.slice(0, 10))
        } catch (error) {
          if (error instanceof Error && error.name === 'AbortError') return
          if (seq !== manualAddSearchSeqRef.current) return
          const detail =
            error instanceof Error ? error.message : 'Unexpected error while searching products.'
          setManualAddSearchError(detail)
          setManualAddSuggestions([])
        } finally {
          if (seq === manualAddSearchSeqRef.current) {
            setIsManualAddSearching(false)
          }
        }
      }

      void run()
    }, 300)

    return () => {
      window.clearTimeout(timeout)
      controller.abort()
    }
  }, [manualAddQuery, flow, token, logout, apiLang])

  // Dismiss main-page suggestion / recent dropdown on outside click.
  useEffect(() => {
    if (flow !== 'scan' && flow !== 'previous') return
    const onPointerDown = (event: MouseEvent): void => {
      const root = manualSearchRootRef.current
      if (root && root.contains(event.target as Node)) return
      setManualSuggestions([])
      setMainSearchInputFocused(false)
    }
    document.addEventListener('mousedown', onPointerDown)
    return () => document.removeEventListener('mousedown', onPointerDown)
  }, [flow, manualSuggestions.length, mainSearchInputFocused])

  // Subset of vision rows the backend accepts for /api/vision/confirm (must have a barcode).
  const verifiedProducts: VerifiedProduct[] = useMemo(() => {
    return visionItems
      .filter((item) => item.selectedBarcode.trim() !== '')
      .map((item) => ({ barcode: item.selectedBarcode, name: item.selectedName }))
  }, [visionItems])

  const canGenerateMenu = verifiedProducts.length > 0 && !isConfirming && !isGeneratingMenu

  /** Full local reset + drop persisted menu session (after consume or user abandon). */
  const resetToInitialState = (): void => {
    if (menuSessionKey) {
      localStorage.removeItem(menuSessionKey)
    }
    // Defer the physical job destruction until the user completes or resets the workspace
    clearMenuJob()
    setFlow('scan')
    setIsVisionProcessing(false)
    setVisionError(null)
    setIsLoadingPrevious(false)
    setPreviousError(null)
    setVisionItems([])
    setPriorityVisionItemId(null)

    setActiveAlternativeIndex(null)
    setAlternativeSelectionIndex(0)

    setIsConfirming(false)
    setIsGeneratingMenu(false)
    setGenerationError(null)
    setMenuGenerationProgress(0)
    setConsumeBusyIndex(null)

    setRecipes([])
    setOpenRecipeStepsIndex(null)

    setManualSearchQuery('')
    setManualSuggestions([])
    setIsManualSearching(false)
    setManualSearchError(null)

    setDiners('2')
    setTimeAvailable('30')
    setDishType('main_course')
    setDietaryStyle('none')
    setSpecialRequests('')

    setIsCheatSubmitting(false)
    setCheatDescription('')
    setCheatError(null)
    setCheatResult(null)

    setManualAddQuery('')
    setManualAddSuggestions([])
    setIsManualAddSearching(false)
    setManualAddSearchError(null)
    setPortionPick(null)
    setPortionGrams('100')
    setPortionError(null)
    setPortionSuccessMessage(null)
    setMainSearchInputFocused(false)
    setModalSearchInputFocused(false)
    setManualAddSearchInputFocused(false)
  }

  /** Toggle exclusive priority pin for one vision row (used in RAG queries). */
  const togglePriorityVisionItem = (id: string): void => {
    setPriorityVisionItemId((previous) => (previous === id ? null : id))
  }

  /** Remove one row from the working ingredient list. */
  const handleDeleteVisionItem = (id: string): void => {
    setVisionItems((previous) => previous.filter((item) => item.id !== id))
    setPriorityVisionItemId((previous) => (previous === id ? null : previous))
  }

  /** Add a catalog product from autocomplete as a new “verified” row (by barcode). */
  const addManualProduct = (suggestion: ProductAutocompleteSuggestion): void => {
    const barcode = suggestion.barcode.trim()
    const name = suggestion.name.trim()
    if (barcode === '' || name === '') return

    setVisionItems((previous) => {
      const exists = previous.some((item) => item.selectedBarcode === barcode)
      if (exists) return previous
      return [
        ...previous,
        {
          id: `manual-${Date.now()}-${barcode}`,
          detectedName: name,
          detectedBarcode: barcode,
          alternatives: [],
          selectedName: name,
          selectedBarcode: barcode,
        },
      ]
    })

    setManualSearchQuery('')
    setManualSuggestions([])
    setManualSearchError(null)
    setMainSearchInputFocused(false)
    pushRecentProduct(suggestion)
  }

  const openAlternativeModal = (index: number): void => {
    openAlternativeModalWithSource(index, 'scan')
  }

  /** Opens replace modal; preselects list index matching current barcode when possible. */
  const openAlternativeModalWithSource = (index: number, source: AlternativeModalSource): void => {
    setActiveAlternativeIndex(index)
    setActiveAlternativeSource(source)
    setModalSearchQuery('')
    setModalSuggestions([])
    setIsModalSearching(false)
    setModalSearchError(null)
    setModalSearchInputFocused(false)
    const item = visionItems[index]
    if (!item) {
      setAlternativeSelectionIndex(0)
      return
    }

    if (item.selectedBarcode === item.detectedBarcode) {
      setAlternativeSelectionIndex(0)
      return
    }

    const matchIndex = (item.alternatives ?? []).findIndex(
      (alt) => alt.barcode === item.selectedBarcode,
    )
    if (matchIndex >= 0) {
      setAlternativeSelectionIndex(matchIndex + 1)
      return
    }

    setAlternativeSelectionIndex(0)
  }

  /** Reset modal-local search state and index. */
  const closeAlternativeModal = useCallback((): void => {
    // Setting everything to null or empty is a good way to reset the state of the modal, closing the window.
    setActiveAlternativeIndex(null)
    setModalSearchQuery('')
    setModalSuggestions([])
    setIsModalSearching(false)
    setModalSearchError(null)
    setModalSearchInputFocused(false)
  }, [])

  /** Modal autocomplete: same endpoint as main search, updates the active row then closes. Select an alternative from the search bar and apply it to the active item. */
  const applyManualSelectionToActiveItem = (suggestion: ProductAutocompleteSuggestion): void => {
    if (activeAlternativeIndex === null) return
    const barcode = suggestion.barcode.trim()
    const name = suggestion.name.trim()
    if (barcode === '' || name === '') return
    setVisionItems((previous) => {
      const copy = previous.slice()
      const current = copy[activeAlternativeIndex]
      if (!current) return previous
      // Override the selected name and barcode with the new one.
      copy[activeAlternativeIndex] = {
        ...current,
        selectedName: name,
        selectedBarcode: barcode,
      }
      return copy
    })
    pushRecentProduct(suggestion)
    closeAlternativeModal()
  }

  /** Apply choice from detected + alternatives list inside the modal. Select an alternative from the list of the vision items and apply it to the active item. */
  const applyAlternativeSelection = (): void => {
    if (activeAlternativeIndex === null) return
    setVisionItems((previous) => {
      const copy = previous.slice()
      const current = copy[activeAlternativeIndex]
      if (!current) return previous

      const detected = { name: current.detectedName, barcode: current.detectedBarcode }
      const alternatives = current.alternatives ?? []

      const chosen =
        alternativeSelectionIndex === 0
          ? detected
          : alternatives[alternativeSelectionIndex - 1] ?? detected

      copy[activeAlternativeIndex] = {
        ...current,
        selectedName: chosen.name,
        selectedBarcode: chosen.barcode,
      }
      return copy
    })
    closeAlternativeModal()
  }

  // Modal-only debounced autocomplete while replace dialog is open. Same as the main search, but for the modal.
  useEffect(() => {
    if (!token) return
    if (activeAlternativeIndex === null) {
      modalSearchSeqRef.current += 1
      setModalSuggestions([])
      setModalSearchError(null)
      setIsModalSearching(false)
      return
    }

    const query = modalSearchQuery.trim()
    if (query.length < 2) {
      modalSearchSeqRef.current += 1
      setModalSuggestions([])
      setModalSearchError(null)
      setIsModalSearching(false)
      return
    }

    modalSearchSeqRef.current += 1
    const seq = modalSearchSeqRef.current
    const queryCaptured = query

    setIsModalSearching(true)
    setModalSearchError(null)

    const controller = new AbortController()
    const timeout = window.setTimeout(() => {
      const run = async (): Promise<void> => {
        try {
          const encoded = encodeURIComponent(queryCaptured)
          const response = await apiFetch(
            `${getApiBaseUrl()}/api/products/autocomplete?query=${encoded}&limit=10&lang=${encodeURIComponent(apiLang)}`,
            {
              method: 'GET',
              headers: { Authorization: `Bearer ${token}` },
              signal: controller.signal,
            },
          )

          if (!response.ok) {
            const detail = await readErrorDetail(
              response,
              `Failed to load suggestions (HTTP ${response.status}).`,
            )
            if (isAuthFailureMessage(detail)) {
              logout()
              return
            }
            throw new Error(detail)
          }

          const payload: unknown = await response.json()
          if (
            typeof payload !== 'object' ||
            payload === null ||
            !('suggestions' in payload) ||
            !Array.isArray((payload as { suggestions: unknown[] }).suggestions)
          ) {
            if (seq === modalSearchSeqRef.current) {
              setModalSuggestions([])
            }
            return
          }

          const parsed = payload as ProductAutocompleteResponse
          if (seq !== modalSearchSeqRef.current) return
          setModalSuggestions(parsed.suggestions.slice(0, 10))
        } catch (error) {
          if (error instanceof Error && error.name === 'AbortError') return
          if (seq !== modalSearchSeqRef.current) return
          const detail =
            error instanceof Error ? error.message : 'Unexpected error while searching products.'
          setModalSearchError(detail)
          setModalSuggestions([])
        } finally {
          if (seq === modalSearchSeqRef.current) {
            setIsModalSearching(false)
          }
        }
      }

      void run()
    }, 300)

    return () => {
      window.clearTimeout(timeout)
      controller.abort()
    }
  }, [modalSearchQuery, activeAlternativeIndex, token, logout, apiLang])

  useEffect(() => {
    if (activeAlternativeIndex === null) return
    const onPointerDown = (event: MouseEvent): void => {
      const root = modalAutocompleteRootRef.current
      if (root && root.contains(event.target as Node)) return
      setModalSuggestions([])
      setModalSearchInputFocused(false)
    }
    document.addEventListener('mousedown', onPointerDown)
    return () => document.removeEventListener('mousedown', onPointerDown)
  }, [modalSuggestions.length, activeAlternativeIndex, modalSearchInputFocused])

  useEffect(() => {
    if (flow !== 'manual_add') return
    // If the user clicks outside the manual add search bar, clear the suggestions and close the input.
    const onPointerDown = (event: MouseEvent): void => {
      const root = manualAddSearchRootRef.current
      // The key: Did the click occur INSIDE our search bar?
      // If 'root' contains the element that was clicked (event.target), ignore the click.
      if (root && root.contains(event.target as Node)) return
      setManualAddSuggestions([])
      setManualAddSearchInputFocused(false)
    }
    document.addEventListener('mousedown', onPointerDown)
    return () => document.removeEventListener('mousedown', onPointerDown)
  }, [flow, manualAddSuggestions.length, manualAddSearchInputFocused])

  // If the user presses the Escape key, close the alternative modal.
  useEffect(() => {
    if (activeAlternativeIndex === null) return
    const onKeyDown = (event: KeyboardEvent): void => {
      if (event.key === 'Escape') closeAlternativeModal()
    }
    document.addEventListener('keydown', onKeyDown)
    return () => document.removeEventListener('keydown', onKeyDown)
  }, [activeAlternativeIndex, closeAlternativeModal])

  /** POST image to /api/vision/process; populates visionItems from model-detected products. */
  const handleProcessFile = async (file: File): Promise<void> => {
    if (!token) return
    setVisionError(null)
    setIsVisionProcessing(true)
    setRecipes([])
    try {
      const products = await processVisionImageWithPolling(token, file, {
        pollIntervalMs: 2000,
        timeoutMs: 4 * 60 * 1000,
      })

      const items: RecipesFromVisionItem[] = products.map((p, index) => ({
        id: `${index}-${p.detected_barcode}`,
        detectedName: p.detected_name,
        detectedBarcode: p.detected_barcode,
        alternatives: p.alternatives,
        selectedName: p.detected_name,
        selectedBarcode: p.detected_barcode,
      }))

      setVisionItems(items)
    } catch (error) {
      const detail = error instanceof Error ? error.message : 'Unexpected error while processing image.'
      setVisionError(detail)
    } finally {
      setIsVisionProcessing(false)
    }
  }

  /** Loads last server-stored scan (/api/vision/previous) as visionItems (no new photo). */
  const handleLoadPrevious = async (): Promise<void> => {
    if (!token) return
    setPreviousError(null)
    setIsLoadingPrevious(true)
    setRecipes([])
    try {
      const response = await apiFetch(`${getApiBaseUrl()}/api/vision/previous`, {
        method: 'GET',
        headers: { Authorization: `Bearer ${token}` },
      })

      if (!response.ok) {
        const detail = await readErrorDetail(response, `Failed to load previous scan (HTTP ${response.status}).`)
        if (isAuthFailureMessage(detail)) {
          logout()
          return
        }
        throw new Error(detail)
      }

      const payload: unknown = await response.json()
      if (
        typeof payload !== 'object' ||
        payload === null ||
        !('detected_barcodes' in payload) ||
        !Array.isArray((payload as { detected_barcodes: unknown[] }).detected_barcodes)
      ) {
        throw new Error(t('errors.invalidPreviousScanResponse'))
      }

      const detected = (payload as { detected_barcodes: VerifiedProduct[] }).detected_barcodes
      const items: RecipesFromVisionItem[] = detected.map((p, index) => ({
        id: `${index}-${p.barcode}`,
        detectedName: p.name,
        detectedBarcode: p.barcode,
        alternatives: [],
        selectedName: p.name,
        selectedBarcode: p.barcode,
      }))
      setVisionItems(items)
    } catch (error) {
      const detail = error instanceof Error ? error.message : 'Unexpected error while loading previous scan.'
      setPreviousError(detail)
    } finally {
      setIsLoadingPrevious(false)
    }
  }

  /**
   * 1) POST /api/vision/confirm with barcodes
   * 2) POST /api/menu-jobs to enqueue generation
   * 3) Poll job + fetch result; optional follow-up POST /api/generate-menu/macros if pending.
   */
  const confirmVisionAndGenerateMenu = async (): Promise<void> => {
    if (!token) return
    if (!canGenerateMenu) return
    setGenerationError(null)
    setIsConfirming(true)
    setIsGeneratingMenu(false)
    try {
      const dinersValue = Number.parseInt(diners, 10)
      const timeValue = Number.parseInt(timeAvailable, 10)
      if (!Number.isFinite(dinersValue) || dinersValue < 1) {
        throw new Error(t('errors.dinersMin'))
      }
      if (!Number.isFinite(timeValue) || timeValue < 1) {
        throw new Error(t('errors.cookingTimeMin'))
      }

      const dietaryStyleValue = dietaryStyle === 'none' ? null : dietaryStyle
      const specialRequestsValue = specialRequests.trim() === '' ? null : specialRequests.trim()

      const priorityItem = priorityVisionItemId
        ? visionItems.find((row) => row.id === priorityVisionItemId)
        : null
      const priorityBarcodeRaw = priorityItem?.selectedBarcode.trim() ?? ''
      const priorityBarcode =
        priorityBarcodeRaw !== '' &&
          verifiedProducts.some((product) => product.barcode === priorityBarcodeRaw)
          ? priorityBarcodeRaw
          : null

      // Phase 1: Confirm ingredients
      // Tell the server: "Hey, of everything you sent me before, 
      // the user has confirmed that they have this exactly (verifiedProducts)".
      const confirmPayload = { verified_products: verifiedProducts }
      const confirmResponse = await apiFetch(`${getApiBaseUrl()}/api/vision/confirm`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify(confirmPayload),
      })
      if (!confirmResponse.ok) {
        const detail = await readErrorDetail(confirmResponse, `Failed to confirm products (HTTP ${confirmResponse.status}).`)
        if (isAuthFailureMessage(detail)) {
          logout()
          return
        }
        throw new Error(detail)
      }

      // Phase 2: Enqueue the recipe generation
      setIsConfirming(false) // We are done with the confirmation phase, we can now start the generation phase.
      setIsGeneratingMenu(true) // We are now generating the menu.
      setMenuGenerationProgress(0)
      const requestPayload = {
        barcodes: verifiedProducts.map((p) => p.barcode),
        priority_barcode: priorityBarcode,
        diners: dinersValue,
        time_available: timeValue,
        dish_type: dishType,
        dietary_style: dietaryStyleValue,
        special_requests: specialRequestsValue,
      }

      // Send the request to the server to enqueue the recipe generation.
      const createJobResponse = await apiFetch(`${getApiBaseUrl()}/api/menu-jobs`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify(requestPayload),
      })

      if (!createJobResponse.ok) {
        const detail = await readErrorDetail(createJobResponse, `Menu generation failed (HTTP ${createJobResponse.status}).`)
        if (isAuthFailureMessage(detail)) {
          logout()
          return
        }
        throw new Error(detail)
      }
      const payload: unknown = await createJobResponse.json()
      const jobId =
        typeof payload === 'object' &&
          payload !== null &&
          'job_id' in payload &&
          typeof (payload as { job_id: unknown }).job_id === 'string'
          ? (payload as { job_id: string }).job_id
          : ''
      if (!jobId) {
        throw new Error(t('errors.invalidJobResponse'))
      }

      // Save the job ID to the local storage to keep track of the job.
      persistMenuJob({ jobId, savedAt: Date.now(), progress: 0, state: 'queued' })
      // Activate a timer that will ask the server every few seconds: 
      // "Is the job 456 done? Is the job 456 done?"
      startJobPolling(jobId)
      void pollJobOnce(jobId)
    } catch (error) {
      const detail = error instanceof Error ? error.message : 'Unexpected error while generating menu.'
      setGenerationError(detail)
    } finally {
      setIsConfirming(false)
      // Do not force-stop generating here: job may still be running in background.
    }
  }

  /** Logs recipe to /api/consume for today → updates Dashboard & History aggregates. */
  const consumeRecipe = async (recipe: FinalRecipe, index: number): Promise<void> => {
    if (!token) return
    setConsumeBusyIndex(index)
    try {
      const kcal = recipe.macro_breakdown?.total_kcal ?? 0
      const protein = recipe.macro_breakdown?.protein_g ?? 0
      const carbs = recipe.macro_breakdown?.carb_g ?? 0
      const fat = recipe.macro_breakdown?.fat_g ?? 0

      const payload = {
        recipe_name: recipe.title,
        recipe_json: recipe,
        calories: kcal,
        protein,
        carbs,
        fat,
        date: getTodayISO(),
      }

      const response = await apiFetch(`${getApiBaseUrl()}/api/consume`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify(payload),
      })

      if (!response.ok) {
        const detail = await readErrorDetail(response, `Failed to log recipe (HTTP ${response.status}).`)
        if (isAuthFailureMessage(detail)) {
          logout()
          return
        }
        throw new Error(detail)
      }

      resetToInitialState()
      navigate('/', { replace: true })
    } catch (error) {
      const detail = error instanceof Error ? error.message : 'Unexpected error while logging recipe.'
      setGenerationError(detail)
    } finally {
      setConsumeBusyIndex(null)
    }
  }

  /** Free-text cheat meal → /api/cheat-meal; shows estimated macros only (no consume here). */
  const submitCheatMeal = async (): Promise<void> => {
    if (!token) return
    const description = cheatDescription.trim()
    if (description === '') {
      setCheatError(t('recipesPage.cheatDescriptionRequired'))
      return
    }
    setCheatError(null)
    setCheatResult(null)
    setIsCheatSubmitting(true)
    try {
      const response = await apiFetch(`${getApiBaseUrl()}/api/cheat-meal`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ description }),
      })

      if (!response.ok) {
        const detail = await readErrorDetail(response, `Cheat meal estimation failed (HTTP ${response.status}).`)
        if (isAuthFailureMessage(detail)) {
          logout()
          return
        }
        throw new Error(detail)
      }

      const payload: unknown = await response.json()
      if (
        typeof payload !== 'object' ||
        payload === null ||
        !('data' in payload) ||
        typeof (payload as { data: unknown }).data !== 'object' ||
        (payload as { data: unknown }).data === null
      ) {
        throw new Error(t('errors.invalidCheatMealResponse'))
      }

      const parsed = payload as CheatMealResponse
      setCheatResult(parsed.data)
      setCheatDescription('')
    } catch (error) {
      const detail = error instanceof Error ? error.message : 'Unexpected error while estimating cheat meal.'
      setCheatError(detail)
    } finally {
      setIsCheatSubmitting(false)
    }
  }

  const openPortionModalForProduct = (suggestion: ProductAutocompleteSuggestion): void => {
    pushRecentProduct(suggestion)
    setPortionPick(suggestion)
    setPortionGrams('100')
    setPortionError(null)
    setPortionSuccessMessage(null)
    setManualAddQuery('')
    setManualAddSuggestions([])
    setIsManualAddSearching(false)
    setManualAddSearchError(null)
    setManualAddSearchInputFocused(false)
  }

  const closePortionModal = (): void => {
    setPortionPick(null)
    setPortionGrams('100')
    setPortionError(null)
    setPortionSuccessMessage(null)
    setIsPortionSubmitting(false)
  }

  const submitPortionedProduct = async (): Promise<void> => {
    if (!token || !portionPick) return
    const grams = Number.parseFloat(portionGrams.replace(',', '.'))
    if (!Number.isFinite(grams) || grams <= 0 || grams > 10000) {
      setPortionError(t('recipesPage.portionGramsInvalid'))
      return
    }
    setIsPortionSubmitting(true)
    setPortionError(null)
    setPortionSuccessMessage(null)
    try {
      const response = await apiFetch(`${getApiBaseUrl()}/api/consume/portioned-product`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          product_name: portionPick.name.trim(),
          barcode: portionPick.barcode.trim() === '' ? null : portionPick.barcode.trim(),
          grams,
        }),
      })

      if (!response.ok) {
        const detail = await readErrorDetail(
          response,
          `Failed to log product portion (HTTP ${response.status}).`,
        )
        if (isAuthFailureMessage(detail)) {
          logout()
          return
        }
        throw new Error(detail)
      }

      const payload: unknown = await response.json()
      const data =
        typeof payload === 'object' &&
          payload !== null &&
          'data' in payload &&
          typeof (payload as { data: unknown }).data === 'object' &&
          (payload as { data: unknown }).data !== null
          ? ((payload as { data: Record<string, unknown> }).data as {
            estimated_kcal?: unknown
            estimated_protein_g?: unknown
            estimated_carb_g?: unknown
            estimated_fat_g?: unknown
            source_note?: unknown
          })
          : null

      if (data && typeof data.estimated_kcal === 'number') {
        const fmt = (v: unknown): string =>
          typeof v === 'number' && Number.isFinite(v) ? v.toFixed(1) : '—'
        setPortionSuccessMessage(
          t('recipesPage.portionLoggedSummary', {
            kcal: String(Math.round(data.estimated_kcal)),
            p: fmt(data.estimated_protein_g),
            c: fmt(data.estimated_carb_g),
            f: fmt(data.estimated_fat_g),
          }),
        )
      } else {
        setPortionSuccessMessage(t('recipesPage.portionLoggedGeneric'))
      }
    } catch (error) {
      const detail = error instanceof Error ? error.message : 'Unexpected error while logging portion.'
      setPortionError(detail)
    } finally {
      setIsPortionSubmitting(false)
    }
  }

  // Display labels for <select>; API still receives the English enum `value` strings.
  const dishTypeOptions: Array<{ value: DishType; label: string }> = useMemo(
    () => [
      { value: 'soup_stew', label: t('recipesPage.dishTypeSoupStew') },
      { value: 'salad_side', label: t('recipesPage.dishTypeSaladSide') },
      { value: 'main_course', label: t('recipesPage.dishTypeMainCourse') },
      { value: 'dessert', label: t('recipesPage.dishTypeDessert') },
      { value: 'snack_breakfast', label: t('recipesPage.dishTypeSnackBreakfast') },
      { value: 'drink', label: t('recipesPage.dishTypeDrink') },
    ],
    [t, i18n.language],
  )

  const dietaryStyleOptions: Array<{ value: DietaryStyle; label: string }> = useMemo(
    () => [
      { value: 'vegan', label: t('recipesPage.dietVegan') },
      { value: 'vegetarian', label: t('recipesPage.dietVegetarian') },
      { value: 'celiac', label: t('recipesPage.dietCeliac') },
      { value: 'mediterranean', label: t('recipesPage.dietMediterranean') },
      { value: 'asian', label: t('recipesPage.dietAsian') },
      { value: 'arabic', label: t('recipesPage.dietArabic') },
      { value: 'latin', label: t('recipesPage.dietLatin') },
    ],
    [t, i18n.language],
  )

  // Restore in-progress session from localStorage once per login (TTL: MENU_SESSION_TTL_MS).
  useEffect(() => {
    if (!isAuthenticated || !token) {
      return
    }

    if (!menuSessionKey) {
      setHasHydratedMenuSession(true)
      return
    }

    // 1. Search for the menu key in the browser's local storage (localStorage).
    const raw = localStorage.getItem(menuSessionKey)
    // 2. If the menu key is not found, set the hasHydratedMenuSession state to true and finish the function.
    if (!raw) {
      setHasHydratedMenuSession(true)
      return
    }

    // 3. TTL (Time To Live) system:
    // Check if more than 2 hours (MENU_SESSION_TTL_MS) have passed since it was saved.
    // If so, consider it "expired", delete it and start from scratch.
    try {
      const parsed: unknown = JSON.parse(raw)
      if (typeof parsed !== 'object' || parsed === null) {
        localStorage.removeItem(menuSessionKey)
        setHasHydratedMenuSession(true)
        return
      }

      const payload = parsed as Partial<PersistedMenuSession>
      const savedAt = typeof payload.savedAt === 'number' ? payload.savedAt : 0
      const isExpired = Date.now() - savedAt > MENU_SESSION_TTL_MS
      if (isExpired) {
        localStorage.removeItem(menuSessionKey)
        setHasHydratedMenuSession(true)
        return
      }

      const flowValue = payload.flow
      const canHydrateFlow = flowValue === 'scan' || flowValue === 'previous'
      const canHydrateItems = Array.isArray(payload.visionItems)
      const canHydrateRecipes = Array.isArray(payload.recipes)
      if (!canHydrateFlow || !canHydrateItems || !canHydrateRecipes) {
        localStorage.removeItem(menuSessionKey)
        setHasHydratedMenuSession(true)
        return
      }

      // 4. Success! Restore the ingredients and recipes we had on the screen.
      setFlow(flowValue)
      setVisionItems(payload.visionItems as RecipesFromVisionItem[])
      setRecipes(payload.recipes as FinalRecipe[])
      const restoredPriority =
        typeof payload.priorityVisionItemId === 'string' ? payload.priorityVisionItemId : null
      const itemIds = new Set(
        (payload.visionItems as RecipesFromVisionItem[]).map((item) => item.id),
      )
      setPriorityVisionItemId(
        restoredPriority && itemIds.has(restoredPriority) ? restoredPriority : null,
      )
      setGenerationError(null)
    } catch {
      localStorage.removeItem(menuSessionKey)
    } finally {
      setHasHydratedMenuSession(true) // Flag to say "I have loaded the old stuff, you can start saving the new stuff".
    }
  }, [isAuthenticated, token, menuSessionKey])

  // --- Menu job persistence (survive refresh while job runs), manages the clock that asks the server if the job is done ---
  const clearMenuJob = (): void => {
    if (menuJobKey) {
      localStorage.removeItem(menuJobKey)
    }
  }

  const persistMenuJob = (value: PersistedMenuJob): void => {
    if (menuJobKey) {
      localStorage.setItem(menuJobKey, JSON.stringify(value))
    }
  }

  const stopJobPolling = (): void => {
    if (jobPollIntervalRef.current !== null) {
      window.clearInterval(jobPollIntervalRef.current)
      jobPollIntervalRef.current = null
    }
  }

  const startJobPolling = (jobId: string): void => {
    stopJobPolling()
    jobPollIntervalRef.current = window.setInterval(() => {
      void pollJobOnce(jobId)
    }, 1500)
  }

  /** Single poll of /api/menu-jobs/:id; on done, fetches /result and may enrich macros. */
  const pollJobOnce = async (jobId: string): Promise<void> => {
    if (!token) return
    try {
      // 1. Ask the server if the job is done.
      const response = await apiFetch(`${getApiBaseUrl()}/api/menu-jobs/${encodeURIComponent(jobId)}`, {
        method: 'GET',
        headers: { Authorization: `Bearer ${token}` },
      })
      if (!response.ok) {
        const detail = await readErrorDetail(response, `Failed to fetch job status (HTTP ${response.status}).`)
        if (isAuthFailureMessage(detail)) {
          logout()
          return
        }
        throw new Error(detail)
      }

      const payload: unknown = await response.json()
      const data = (payload as { data?: unknown }).data as
        | { state?: unknown; progress?: unknown; message?: unknown }
        | undefined

      // Extract the state ('queued', 'running', 'done', 'error') and the progress (0 to 100).
      const stateValue = typeof data?.state === 'string' ? (data.state as MenuJobState) : 'running'
      const progressValue = typeof data?.progress === 'number' ? data.progress : 0

      // 2. Update the progress bar in the interface.
      setIsGeneratingMenu(stateValue === 'queued' || stateValue === 'running')
      setMenuGenerationProgress(progressValue)

      // 3. Save the progress to disk in case the user reloads the page.
      persistMenuJob({
        jobId,
        savedAt: Date.now(),
        progress: progressValue,
        state: stateValue,
      })

      if (stateValue === 'error') {
        stopJobPolling()
        const msg = typeof data?.message === 'string' ? data.message : 'Menu generation failed.'
        setGenerationError(msg)
        clearMenuJob()
        return
      }

      if (stateValue === 'done') {
        stopJobPolling()
        const resultResponse = await apiFetch(
          `${getApiBaseUrl()}/api/menu-jobs/${encodeURIComponent(jobId)}/result`,
          { method: 'GET', headers: { Authorization: `Bearer ${token}` } },
        )
        if (!resultResponse.ok) {
          const detail = await readErrorDetail(
            resultResponse,
            `Failed to fetch job result (HTTP ${resultResponse.status}).`,
          )
          if (isAuthFailureMessage(detail)) {
            logout()
            return
          }
          throw new Error(detail)
        }
        const resultPayload: unknown = await resultResponse.json()
        const resultData = (resultPayload as { data?: unknown }).data as
          | { recipes?: unknown; macros_pending?: unknown }
          | undefined
        if (Array.isArray(resultData?.recipes)) {
          const nextRecipes = resultData.recipes as FinalRecipe[]
          setRecipes(nextRecipes)
          setMenuGenerationProgress(100)

          // 5. SECOND PHASE (Enrichment of Macros):
          // The AI returned the recipe text very quickly, but the mathematical calculations
          // may not be ready. If the server says "macros_pending: true"...
          const macrosPending = resultData?.macros_pending === true
          if (macrosPending && !macrosComputeInFlightRef.current) {
            macrosComputeInFlightRef.current = true
            try {
              // Ask the server to compute the macros.
              const macrosResponse = await apiFetch(`${getApiBaseUrl()}/api/generate-menu/macros`, {
                method: 'POST',
                headers: { Authorization: `Bearer ${token}` },
              })
              if (!macrosResponse.ok) {
                const detail = await readErrorDetail(
                  macrosResponse,
                  `Failed to compute macros (HTTP ${macrosResponse.status}).`,
                )
                if (isAuthFailureMessage(detail)) {
                  logout()
                  return
                }
                throw new Error(detail)
              }
              // If the server returns a "map" (dictionary) with the macros per recipe...
              const macrosPayload: unknown = await macrosResponse.json()
              const macrosData = (macrosPayload as { data?: unknown }).data as
                | { macro_map?: unknown }
                | undefined
              const macroMap = (macrosData?.macro_map ?? null) as Record<string, unknown> | null
              if (macroMap && typeof macroMap === 'object') {
                // Iterate over the recipes that were already on the screen, search for their macros
                // in the map, and inject them inside.
                const merged = nextRecipes.map((recipe) => {
                  const mb = macroMap[recipe.title]
                  if (!mb || typeof mb !== 'object') {
                    return recipe
                  }
                  return { ...recipe, macro_breakdown: mb as MacroBreakdown }
                })
                setRecipes(merged)
              }
            } finally {
              macrosComputeInFlightRef.current = false
            }
          }
        }
        // Save the job state as completed instead of destroying it prematurely
        persistMenuJob({
          jobId,
          savedAt: Date.now(),
          progress: 100,
          state: 'done',
        })
      }
    } catch (error) {
      const detail = error instanceof Error ? error.message : 'Unexpected error while polling menu job.'
      setGenerationError(detail)
      stopJobPolling()
      clearMenuJob()
    }
  }

  // Persist visionItems + recipes + flow for scan/previous (not cheat) so refresh keeps WIP.
  useEffect(() => {
    // 1. If we haven't finished loading the old session, do nothing. 
    // (If we saved now, we would overwrite the old session with an empty state).
    if (!hasHydratedMenuSession) {
      return
    }
    if (!isAuthenticated || !token) {
      return
    }
    if (!menuSessionKey) {
      return
    }

    // 2. Cheat meals does not need to be saved (it's a one-time thing).
    if (flow === 'cheat' || flow === 'manual_add') {
      localStorage.removeItem(menuSessionKey)
      return
    }

    // 3. Is there really something to save?
    const hasSessionState = visionItems.length > 0 || recipes.length > 0
    if (!hasSessionState) {
      localStorage.removeItem(menuSessionKey)
      return
    }

    // 4. Save the session state to the browser's local storage.
    const flowValue: PersistedRecipesFlow = flow === 'previous' ? 'previous' : 'scan'
    const payload: PersistedMenuSession = {
      savedAt: Date.now(),
      flow: flowValue,
      visionItems,
      recipes,
      priorityVisionItemId,
    }
    localStorage.setItem(menuSessionKey, JSON.stringify(payload))
  }, [
    hasHydratedMenuSession,
    isAuthenticated,
    token,
    menuSessionKey,
    flow,
    visionItems,
    recipes,
    priorityVisionItemId,
  ])

  // On mount, resume polling if a menu job id was left in localStorage (same TTL as session).
  useEffect(() => {
    // 1. If you are not logged in, ensure that the polling clock is turned off and delete the job from the disk.
    if (!isAuthenticated || !token) {
      stopJobPolling()
      const k = lastMenuJobKeyRef.current
      if (k) {
        localStorage.removeItem(k)
      }
      return
    }

    if (!menuJobKey) {
      stopJobPolling()
      return
    }

    // 2. Check if there is any job (Job) running on the disk.
    const raw = localStorage.getItem(menuJobKey)
    if (!raw) {
      return
    }
    try {
      const parsed: unknown = JSON.parse(raw)
      if (typeof parsed !== 'object' || parsed === null) {
        clearMenuJob()
        return
      }
      const payload = parsed as Partial<PersistedMenuJob>
      const jobId = typeof payload.jobId === 'string' ? payload.jobId : ''
      const savedAt = typeof payload.savedAt === 'number' ? payload.savedAt : 0
      if (!jobId) {
        clearMenuJob()
        return
      }
      // 3. Check if the job is expired (TTL system).
      if (Date.now() - savedAt > MENU_SESSION_TTL_MS) {
        clearMenuJob()
        return
      }
      // 4. JOB FOUND AND VALID!
      setIsGeneratingMenu(true)
      // Restore the percentage where it left off (or 0 if there was none).
      setMenuGenerationProgress(typeof payload.progress === 'number' ? payload.progress : 0)
      startJobPolling(jobId)
      void pollJobOnce(jobId)
    } catch {
      clearMenuJob()
    }
    return () => {
      stopJobPolling()
    }
  }, [isAuthenticated, token, menuJobKey])

  /* Recent searchs algorithms (must run before auth early return — hooks rules). */
  const mainMatchingRecents = useMemo(
    () => filterAndSortRecentMatches(recentProductPicks, manualSearchQuery.trim()),
    [recentProductPicks, manualSearchQuery],
  )
  const mainCatalogDeduped = useMemo(() => {
    const seen = new Set(mainMatchingRecents.map((s) => s.barcode.trim()))
    return manualSuggestions.filter((s) => !seen.has(s.barcode.trim()))
  }, [mainMatchingRecents, manualSuggestions])

  const modalMatchingRecents = useMemo(
    () => filterAndSortRecentMatches(recentProductPicks, modalSearchQuery.trim()),
    [recentProductPicks, modalSearchQuery],
  )
  const modalCatalogDeduped = useMemo(() => {
    const seen = new Set(modalMatchingRecents.map((s) => s.barcode.trim()))
    return modalSuggestions.filter((s) => !seen.has(s.barcode.trim()))
  }, [modalMatchingRecents, modalSuggestions])

  const manualAddMatchingRecents = useMemo(
    () => filterAndSortRecentMatches(recentProductPicks, manualAddQuery.trim()),
    [recentProductPicks, manualAddQuery],
  )
  const manualAddCatalogDeduped = useMemo(() => {
    const seen = new Set(manualAddMatchingRecents.map((s) => s.barcode.trim()))
    return manualAddSuggestions.filter((s) => !seen.has(s.barcode.trim()))
  }, [manualAddMatchingRecents, manualAddSuggestions])

  const showMainRecentPanel =
    mainSearchInputFocused &&
    manualSearchQuery.trim().length < 2 &&
    manualSuggestions.length === 0 &&
    recentProductPicks.length > 0

  const showMainTypedPanel =
    mainSearchInputFocused &&
    manualSearchQuery.trim().length >= 2 &&
    (mainMatchingRecents.length > 0 || isManualSearching || manualSuggestions.length > 0)

  const showMainSearchDropdown = showMainRecentPanel || showMainTypedPanel

  const showModalRecentPanel =
    activeAlternativeIndex !== null &&
    modalSearchInputFocused &&
    modalSearchQuery.trim().length < 2 &&
    modalSuggestions.length === 0 &&
    recentProductPicks.length > 0

  const showModalTypedPanel =
    activeAlternativeIndex !== null &&
    modalSearchInputFocused &&
    modalSearchQuery.trim().length >= 2 &&
    (modalMatchingRecents.length > 0 || isModalSearching || modalSuggestions.length > 0)

  const showModalSearchDropdown = showModalRecentPanel || showModalTypedPanel

  const showManualAddRecentPanel =
    flow === 'manual_add' &&
    manualAddSearchInputFocused &&
    manualAddQuery.trim().length < 2 &&
    manualAddSuggestions.length === 0 &&
    recentProductPicks.length > 0

  const showManualAddTypedPanel =
    flow === 'manual_add' &&
    manualAddSearchInputFocused &&
    manualAddQuery.trim().length >= 2 &&
    (manualAddMatchingRecents.length > 0 || isManualAddSearching || manualAddSuggestions.length > 0)

  const showManualAddSearchDropdown = showManualAddRecentPanel || showManualAddTypedPanel

  // Auth gate: effect already redirected; avoid flashing UI.
  if (!isAuthenticated || !token) {
    return null
  }

  return (
    <section className="space-y-6">
      {/* Page title + i18n subtitle */}
      <header className="flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <h1 className="text-xl font-bold text-[var(--color-primary)] sm:text-2xl">{t('recipesPage.title')}</h1>
          <p className="mt-1 text-sm font-semibold text-[var(--color-secondary)]">
            {t('recipesPage.subtitle')}
          </p>
        </div>
      </header>

      {/* Top-level mode: scan | previous | cheat | manual add — dropdown on small screens, cards from sm+ */}
      <div className="sm:hidden">
        <label htmlFor="recipes-flow-select" className="mb-1.5 block text-sm font-semibold text-[var(--color-primary)]">
          {t('recipesPage.modeSelectLabel')}
        </label>
        <select
          id="recipes-flow-select"
          value={flow}
          onChange={(event) => {
            setGenerationError(null)
            setFlow(event.target.value as ProductFlow)
          }}
          className="w-full rounded-xl border border-[var(--color-border)] bg-white px-3 py-2.5 text-sm font-semibold text-[var(--color-primary)] outline-none transition focus:border-[var(--color-border)] focus:ring-2 focus:ring-[var(--color-focus-ring)]"
        >
          <option value="scan">{t('recipesPage.scanTitle')}</option>
          <option value="previous">{t('recipesPage.previousTitle')}</option>
          <option value="cheat">{t('recipesPage.cheatTitle')}</option>
          <option value="manual_add">{t('recipesPage.manualAddTitle')}</option>
        </select>
      </div>

      <div className="hidden gap-4 sm:grid sm:grid-cols-2 xl:grid-cols-4">
        <button
          type="button"
          onClick={() => {
            setGenerationError(null)
            setFlow('scan')
          }}
          className={`rounded-2xl border p-5 text-left transition ${flow === 'scan'
            ? 'border-[var(--color-active-border)] bg-[var(--color-active-bg)]'
            : 'border-[var(--color-border)] bg-white hover:bg-[var(--color-surface-soft)]'
            }`}
        >
          <div className="mt-2 text-lg font-bold text-[var(--color-primary)]">{t('recipesPage.scanTitle')}</div>
          <div className="mt-2 text-sm font-semibold text-[var(--color-secondary)]">
            {i18n.language.startsWith('es')
              ? 'Sube o captura una foto de tu nevera o despensa en vista frontal, o de tu encimera en vista cenital'
              : t('recipesPage.scanBody')}
          </div>
        </button>

        <button
          type="button"
          onClick={() => {
            setGenerationError(null)
            setFlow('previous')
          }}
          className={`rounded-2xl border p-5 text-left transition ${flow === 'previous'
            ? 'border-[var(--color-active-border)] bg-[var(--color-active-bg)]'
            : 'border-[var(--color-border)] bg-white hover:bg-[var(--color-surface-soft)]'
            }`}
        >
          <div className="mt-2 text-lg font-bold text-[var(--color-primary)]">{t('recipesPage.previousTitle')}</div>
          <div className="mt-2 text-sm font-semibold text-[var(--color-secondary)]">{t('recipesPage.previousBody')}</div>
        </button>

        <button
          type="button"
          onClick={() => {
            setGenerationError(null)
            setFlow('cheat')
          }}
          className={`rounded-2xl border p-5 text-left transition ${flow === 'cheat'
            ? 'border-[var(--color-active-border)] bg-[var(--color-active-bg)]'
            : 'border-[var(--color-border)] bg-white hover:bg-[var(--color-surface-soft)]'
            }`}
        >
          <div className="mt-2 text-lg font-bold text-[var(--color-primary)]">{t('recipesPage.cheatTitle')}</div>
          <div className="mt-2 text-sm font-semibold text-[var(--color-secondary)]">{t('recipesPage.cheatBody')}</div>
        </button>

        <button
          type="button"
          onClick={() => {
            setGenerationError(null)
            setFlow('manual_add')
          }}
          className={`rounded-2xl border p-5 text-left transition ${flow === 'manual_add'
            ? 'border-[var(--color-active-border)] bg-[var(--color-active-bg)]'
            : 'border-[var(--color-border)] bg-white hover:bg-[var(--color-surface-soft)]'
            }`}
        >
          <div className="mt-2 text-lg font-bold text-[var(--color-primary)]">{t('recipesPage.manualAddTitle')}</div>
          <div className="mt-2 text-sm font-semibold text-[var(--color-secondary)]">{t('recipesPage.manualAddBody')}</div>
        </button>
      </div>

      {/* ---------- Scan flow: photo pipeline + manual search + detected cards ---------- */}
      {flow === 'scan' ? (
        <div className="space-y-4 rounded-2xl border border-[var(--color-border)] bg-[var(--color-surface)] p-4 shadow-sm sm:p-6">
          <h2 className="text-sm font-bold uppercase tracking-wide text-[var(--color-primary)]">
            {t('recipesPage.visionTitle')}
          </h2>

          <div className="grid gap-4 sm:grid-cols-2">
            <label className="cursor-pointer rounded-2xl border border-[var(--color-border)] bg-white p-4 text-center shadow-sm transition hover:bg-[var(--color-surface-soft)]">
              <input
                type="file"
                accept="image/*"
                className="hidden"
                disabled={isVisionProcessing}
                onChange={(event) => {
                  const file = event.target.files?.[0];
                  if (!file) return;
                  void handleProcessFile(file);
                  event.currentTarget.value = '';
                }}
              />
              <div className="text-sm font-semibold text-[var(--color-secondary)]">{t('recipesPage.uploadImage')}</div>
              <div className="mt-2 text-base font-bold text-[var(--color-primary)]">{t('recipesPage.fromDevice')}</div>
            </label>

            <label className="cursor-pointer rounded-2xl border border-[var(--color-border)] bg-white p-4 text-center shadow-sm transition hover:bg-[var(--color-surface-soft)]">
              <input
                type="file"
                accept="image/*"
                capture="environment"
                className="hidden"
                disabled={isVisionProcessing}
                onChange={(event) => {
                  const file = event.target.files?.[0];
                  if (!file) return;
                  void handleProcessFile(file);
                  event.currentTarget.value = '';
                }}
              />
              <div className="text-sm font-semibold text-[var(--color-secondary)]">{t('recipesPage.cameraCapture')}</div>
              <div className="mt-2 text-base font-bold text-[var(--color-primary)]">{t('recipesPage.usePhone')}</div>
            </label>
          </div>

          <div className="relative" ref={manualSearchRootRef}>
            <label className="block text-sm font-bold uppercase tracking-wide text-[var(--color-primary)]">
              {t('recipesPage.manualSearchLabel')}
            </label>
            <div className="mt-2 flex items-center gap-2 rounded-2xl border border-[var(--color-border)] bg-white px-4 py-3 shadow-sm">
              <Search className="h-4 w-4 shrink-0 text-[var(--color-secondary)]" aria-hidden />
              <input
                value={manualSearchQuery}
                onChange={(event) => setManualSearchQuery(event.target.value)}
                onFocus={() => setMainSearchInputFocused(true)}
                placeholder={t('recipesPage.manualSearchPlaceholder')}
                className="w-full bg-transparent text-sm font-semibold text-[var(--color-primary)] placeholder:text-[var(--color-border)] outline-none"
                disabled={isVisionProcessing || isConfirming || isGeneratingMenu}
              />
              <div className="text-xs font-semibold text-[var(--color-secondary)]">
                {isManualSearching ? t('recipesPage.searching') : null}
              </div>
            </div>

            {manualSearchError ? (
              <div className="mt-2 rounded-2xl border border-[var(--color-active-border)] bg-[var(--color-active-bg)] p-3 text-sm font-semibold text-[var(--color-secondary)]">
                {manualSearchError}
              </div>
            ) : null}

            {showMainSearchDropdown ? (
              showMainRecentPanel ? (
                <ProductSearchDropdown
                  variant="recent-only"
                  allRecentProducts={recentProductPicks}
                  matchingRecents={[]}
                  catalogSuggestions={[]}
                  isSearching={false}
                  onPick={addManualProduct}
                  listKeyPrefix="main-scan"
                  maxHeightClass="max-h-72"
                />
              ) : (
                <ProductSearchDropdown
                  variant="typed-search"
                  allRecentProducts={[]}
                  matchingRecents={mainMatchingRecents}
                  catalogSuggestions={mainCatalogDeduped}
                  isSearching={isManualSearching}
                  onPick={addManualProduct}
                  listKeyPrefix="main-scan"
                  maxHeightClass="max-h-72"
                />
              )
            ) : null}
          </div>

          {/* If the vision is processing, show a loading indicator. */}
          {isVisionProcessing ? (
            <div className="rounded-2xl border border-dashed border-slate-300 bg-white p-6 text-center">
              <div className="text-sm font-semibold text-[var(--color-secondary)]">{t('recipesPage.processingImage')}</div>
              <div className="mt-3 text-xs font-semibold text-[var(--color-secondary)]">
                {t('recipesPage.processingImageBody')}
              </div>
            </div>
          ) : null}

          {/* If there is an error, show it. */}
          {visionError ? (
            <div className="rounded-2xl border border-[var(--color-active-border)] bg-[var(--color-active-bg)] p-4 text-sm font-semibold text-[var(--color-secondary)]">
              {visionError}
            </div>
          ) : null}

          {/* If there are vision items, show them. */}
          {visionItems.length > 0 ? (
            <div className="space-y-4">
              <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
                <h3 className="text-sm font-bold uppercase tracking-wide text-[var(--color-primary)]">
                  {t('recipesPage.detectedProductsTitle')}
                </h3>
                <div className="text-sm font-bold text-[var(--color-secondary)]">
                  {t('recipesPage.verifiedItems')}{' '}
                  <span className="font-semibold text-[var(--color-primary)]">{verifiedProducts.length}</span>
                </div>
              </div>
              <p className="text-xs font-semibold text-[var(--color-secondary)]">
                {t('recipesPage.priorityIngredientHint')}
              </p>

              <div className="grid gap-3 md:grid-cols-2">
                {visionItems.map((item, index) => (
                  <VisionProductCard
                    key={item.id}
                    item={item}
                    isPriority={item.id === priorityVisionItemId}
                    macrosByBarcode={macrosByBarcode}
                    onOpenAlternative={() => openAlternativeModal(index)}
                    onTogglePriority={() => togglePriorityVisionItem(item.id)}
                    onDelete={() => handleDeleteVisionItem(item.id)}
                  />
                ))}
              </div>
            </div>
          ) : null} {/* If there are no vision items, show nothing. */}
        </div>
      ) : null} {/* If the flow is not new ingredients, show nothing. */}

      {/* ---------- Previous flow: fetch saved barcodes + same manual search / cards ---------- */}
      {flow === 'previous' ? (
        <div className="space-y-4 rounded-2xl border border-[var(--color-border)] bg-[var(--color-surface)] p-4 shadow-sm sm:p-6">
          <h2 className="text-sm font-bold uppercase tracking-wide text-[var(--color-primary)]">
            {t('recipesPage.previousScanTitle')}
          </h2>

          <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
            <div className="text-sm font-semibold text-[var(--color-secondary)]">
              {t('recipesPage.previousScanBody')}
            </div>
            <button
              type="button"
              disabled={isLoadingPrevious}
              onClick={() => void handleLoadPrevious()}
              className="rounded-xl bg-gradient-to-r from-emerald-600 to-sky-600 px-4 py-3 text-sm font-semibold uppercase tracking-wide text-white shadow-sm transition hover:from-emerald-700 hover:to-sky-700 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {isLoadingPrevious ? t('recipesPage.loading') : t('recipesPage.loadPreviousIngredients')}
            </button>
          </div>

          <div className="relative" ref={manualSearchRootRef}>
            <label className="block text-sm font-bold uppercase tracking-wide text-[var(--color-primary)]">
              {t('recipesPage.manualSearchLabel')}
            </label>
            <div className="mt-2 flex items-center gap-2 rounded-2xl border border-[var(--color-border)] bg-white px-4 py-3 shadow-sm">
              <Search className="h-4 w-4 shrink-0 text-[var(--color-secondary)]" aria-hidden />
              <input
                value={manualSearchQuery}
                onChange={(event) => setManualSearchQuery(event.target.value)}
                onFocus={() => setMainSearchInputFocused(true)}
                placeholder={t('recipesPage.manualSearchPlaceholder')}
                className="w-full bg-transparent text-sm font-semibold text-[var(--color-primary)] placeholder:text-[var(--color-border)] outline-none"
                disabled={isLoadingPrevious || isConfirming || isGeneratingMenu}
              />
              <div className="text-xs font-semibold text-[var(--color-secondary)]">
                {isManualSearching ? t('recipesPage.searching') : null}
              </div>
            </div>

            {manualSearchError ? (
              <div className="mt-2 rounded-2xl border border-[var(--color-active-border)] bg-[var(--color-active-bg)] p-3 text-sm font-semibold text-[var(--color-secondary)]">
                {manualSearchError}
              </div>
            ) : null}

            {showMainSearchDropdown ? (
              showMainRecentPanel ? (
                <ProductSearchDropdown
                  variant="recent-only"
                  allRecentProducts={recentProductPicks}
                  matchingRecents={[]}
                  catalogSuggestions={[]}
                  isSearching={false}
                  onPick={addManualProduct}
                  listKeyPrefix="main-prev"
                  maxHeightClass="max-h-72"
                />
              ) : (
                <ProductSearchDropdown
                  variant="typed-search"
                  allRecentProducts={[]}
                  matchingRecents={mainMatchingRecents}
                  catalogSuggestions={mainCatalogDeduped}
                  isSearching={isManualSearching}
                  onPick={addManualProduct}
                  listKeyPrefix="main-prev"
                  maxHeightClass="max-h-72"
                />
              )
            ) : null}
          </div>

          {previousError ? (
            <div className="rounded-2xl border border-[var(--color-active-border)] bg-[var(--color-active-bg)] p-4 text-sm font-semibold text-[var(--color-secondary)]">
              {previousError}
            </div>
          ) : null}

          {visionItems.length > 0 ? (
            <div className="space-y-4">
              <h3 className="text-sm font-bold uppercase tracking-wide text-[var(--color-primary)]">
                {t('recipesPage.verifiedItems')}
              </h3>
              <p className="text-xs font-semibold text-[var(--color-secondary)]">
                {t('recipesPage.priorityIngredientHint')}
              </p>
              <div className="grid gap-3 md:grid-cols-2">
                {visionItems.map((item, index) => (
                  <VisionProductCard
                    key={item.id}
                    item={item}
                    isPriority={item.id === priorityVisionItemId}
                    macrosByBarcode={macrosByBarcode}
                    onOpenAlternative={() => openAlternativeModalWithSource(index, 'previous')}
                    onTogglePriority={() => togglePriorityVisionItem(item.id)}
                    onDelete={() => handleDeleteVisionItem(item.id)}
                  />
                ))}
              </div>
            </div>
          ) : null}
        </div>
      ) : null}

      {/* ---------- Cheat meal: isolated form; skips vision + menu job ---------- */}
      {flow === 'cheat' ? (
        <div className="space-y-4 rounded-2xl border border-[var(--color-border)] bg-[var(--color-surface)] p-4 shadow-sm sm:p-6">
          <h2 className="text-sm font-bold uppercase tracking-wide text-[var(--color-primary)]">
            {t('recipesPage.cheatMealTitle')}
          </h2>
          <div className="space-y-2">
            <label htmlFor="cheat-description" className="text-sm font-semibold text-[var(--color-primary)]">
              {t('recipesPage.cheatMealQuestion')}
            </label>
            <textarea
              id="cheat-description"
              value={cheatDescription}
              onChange={(event) => setCheatDescription(event.target.value)}
              placeholder={t('recipesPage.cheatMealPlaceholder')}
              className="min-h-[110px] w-full rounded-2xl border border-slate-300 bg-white p-4 text-sm text-[var(--color-primary)] placeholder:text-[var(--color-border)] outline-none transition focus:border-[var(--color-secondary)] focus:ring-2 focus:ring-[var(--color-focus-ring)]"
            />
          </div>

          {cheatError ? (
            <div className="rounded-2xl border border-[var(--color-active-border)] bg-[var(--color-active-bg)] p-4 text-sm font-semibold text-[var(--color-secondary)]">
              {cheatError}
            </div>
          ) : null}

          <button
            type="button"
            onClick={() => void submitCheatMeal()}
            disabled={isCheatSubmitting}
            className="rounded-xl bg-gradient-to-r from-emerald-600 to-sky-600 px-4 py-3 text-sm font-semibold uppercase tracking-wide text-white shadow-sm transition hover:from-emerald-700 hover:to-sky-700 disabled:cursor-not-allowed disabled:opacity-60"
          >
            {isCheatSubmitting ? t('recipesPage.estimating') : t('recipesPage.estimateAndLog')}
          </button>

          {cheatResult ? (
            <div className="rounded-2xl border border-[var(--color-border)] bg-white p-5 shadow-sm">
              <div className="text-sm font-bold text-[var(--color-primary)]">{t('recipesPage.cheatMealLogged')}</div>
              <div className="mt-2 text-xs font-semibold text-[var(--color-secondary)]">
                {cheatResult.description}
              </div>

              <div className="mt-4 grid gap-3 sm:grid-cols-4">
                {(
                  [
                    { metric: 'kcal' as const, value: cheatResult.estimated_kcal },
                    { metric: 'protein' as const, value: cheatResult.estimated_protein_g },
                    { metric: 'carbs' as const, value: cheatResult.estimated_carb_g },
                    { metric: 'fat' as const, value: cheatResult.estimated_fat_g },
                  ] as const
                ).map((m) => (
                  <div key={m.metric} className={`rounded-xl px-3 py-2 text-sm font-semibold ${metricColor(m.metric)}`}>
                    {metricLabel(m.metric)}: {m.value}
                    {metricUnit(m.metric)}
                  </div>
                ))}
              </div>

              <p className="mt-4 text-sm font-semibold text-[var(--color-secondary)]">{cheatResult.estimation_notes}</p>
            </div>
          ) : null}
        </div>
      ) : null}

      {/* ---------- Menu job controls: visible for scan/previous only ---------- */}
      {flow !== 'cheat' && flow !== 'manual_add' ? (
        <div className="space-y-4 rounded-2xl border border-[var(--color-border)] bg-white p-4 shadow-sm sm:p-6">
          <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <h2 className="text-sm font-bold uppercase tracking-wide text-[var(--color-primary)]">
                {t('recipesPage.menuGenerationTitle')}
              </h2>
              <p className="mt-1 text-sm font-semibold text-[var(--color-secondary)]">
                {t('recipesPage.menuGenerationBody')}
              </p>
            </div>
            <div className="text-sm font-bold text-[var(--color-secondary)]">
              {t('recipesPage.verifiedItems')}{' '}
              <span className="font-semibold text-[var(--color-primary)]">{verifiedProducts.length}</span>
            </div>
          </div>

          {generationError ? (
            <div className="rounded-2xl border border-[var(--color-active-border)] bg-[var(--color-active-bg)] p-4 text-sm font-semibold text-[var(--color-secondary)]">
              {generationError}
            </div>
          ) : null}

          <div className="grid gap-4 lg:grid-cols-2">
            <div className="space-y-2">
              <label className="block text-sm font-medium text-[var(--color-primary)]" htmlFor="diners">
                {t('recipesPage.diners')}
              </label>
              <input
                id="diners"
                type="number"
                min={1}
                value={diners}
                onChange={(event) => setDiners(event.target.value)}
                className="w-full rounded-xl border border-slate-300 bg-white px-4 py-3 text-sm outline-none transition focus:border-[var(--color-secondary)] focus:ring-2 focus:ring-[var(--color-focus-ring)]"
              />
            </div>

            <div className="space-y-2">
              <label className="block text-sm font-medium text-[var(--color-primary)]" htmlFor="time-available">
                {t('recipesPage.cookingTimeMinutes')}
              </label>
              <input
                id="time-available"
                type="number"
                min={0}
                step={5}
                value={timeAvailable}
                onChange={(event) => setTimeAvailable(event.target.value)}
                className="w-full rounded-xl border border-slate-300 bg-white px-4 py-3 text-sm outline-none transition focus:border-[var(--color-secondary)] focus:ring-2 focus:ring-[var(--color-focus-ring)]"
              />
            </div>

            <div className="space-y-2">
              <label className="block text-sm font-medium text-[var(--color-primary)]" htmlFor="dish-type">
                {t('recipesPage.dishType')}
              </label>
              <select
                id="dish-type"
                value={dishType}
                onChange={(event) => setDishType(event.target.value as DishType)}
                className="w-full rounded-xl border border-slate-300 bg-white px-4 py-3 text-sm outline-none transition focus:border-[var(--color-secondary)] focus:ring-2 focus:ring-[var(--color-focus-ring)]"
              >
                {dishTypeOptions.map((opt) => (
                  <option key={opt.value} value={opt.value}>
                    {opt.label}
                  </option>
                ))}
              </select>
            </div>

            <div className="space-y-2">
              <label className="block text-sm font-medium text-[var(--color-primary)]" htmlFor="dietary-style">
                {t('recipesPage.dietaryStyleOptional')}
              </label>
              <select
                id="dietary-style"
                value={dietaryStyle}
                onChange={(event) => setDietaryStyle(event.target.value as DietaryStyle | 'none')}
                className="w-full rounded-xl border border-slate-300 bg-white px-4 py-3 text-sm outline-none transition focus:border-[var(--color-secondary)] focus:ring-2 focus:ring-[var(--color-focus-ring)]"
              >
                <option value="none">{t('common.none')}</option>
                {dietaryStyleOptions.map((opt) => (
                  <option key={opt.value} value={opt.value}>
                    {opt.label}
                  </option>
                ))}
              </select>
            </div>

            <div className="space-y-2 lg:col-span-2">
              <label className="block text-sm font-medium text-[var(--color-primary)]" htmlFor="special-requests">
                {t('recipesPage.specialRequestsOptional')}
              </label>
              <textarea
                id="special-requests"
                value={specialRequests}
                onChange={(event) => setSpecialRequests(event.target.value)}
                placeholder={t('recipesPage.specialRequestsPlaceholder')}
                className="min-h-[90px] w-full rounded-xl border border-slate-300 bg-white p-4 text-sm text-[var(--color-primary)] placeholder:text-[var(--color-secondary)] outline-none transition focus:border-[var(--color-secondary)] focus:ring-2 focus:ring-[var(--color-focus-ring)]"
              />
            </div>
          </div>

          <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-end">
            <button
              type="button"
              disabled={!canGenerateMenu}
              onClick={() => void confirmVisionAndGenerateMenu()}
              className="inline-flex items-center justify-center rounded-lg bg-gradient-to-r from-emerald-600 to-sky-600 px-5 py-3 text-sm font-semibold uppercase tracking-wide text-white shadow-sm transition hover:from-emerald-700 hover:to-sky-700 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {/* NESTED TERNARY OPERATOR (If/Else multiple) */}
              {isConfirming
                ? t('recipesPage.savingVerified')
                : isGeneratingMenu
                  ? t('recipesPage.generatingMenu')
                  : t('recipesPage.generateMenu')}
            </button>
            {/* If the menu is generating, show the progress bar. */}
            {isGeneratingMenu ? (
              <div className="w-full sm:w-72">
                <div className="h-2 w-full overflow-hidden rounded-full bg-slate-100">
                  <div
                    className="h-full rounded-full bg-gradient-to-r from-emerald-600 to-sky-600 transition-[width] duration-300"
                    style={{ width: `${menuGenerationProgress}%` }}
                  />
                </div>
                <div className="mt-2 text-right text-xs font-semibold text-[var(--color-secondary)]">
                  {menuGenerationProgress}%
                </div>
                {/* Background navigation notice */}
                <p className="mt-1 text-xs font-medium text-[var(--color-secondary)] sm:text-right">
                  {t('recipesPage.backgroundProcessingNotice')}
                </p>
              </div>
            ) : null}
          </div>
        </div>
      ) : null}

      {/* ---------- Manual add section: visible for manual_add only ---------- */}
      {flow === 'manual_add' ? (
        <div className="space-y-4 rounded-2xl border border-[var(--color-border)] bg-[var(--color-surface)] p-4 shadow-sm sm:p-6">
          <h2 className="text-sm font-bold uppercase tracking-wide text-[var(--color-primary)]">
            {t('recipesPage.manualAddSectionTitle')}
          </h2>
          <p className="text-sm font-semibold text-[var(--color-secondary)]">{t('recipesPage.manualAddSectionBody')}</p>

          <div className="relative" ref={manualAddSearchRootRef}>
            <label className="block text-sm font-bold uppercase tracking-wide text-[var(--color-primary)]">
              {t('recipesPage.manualSearchLabel')}
            </label>
            <div className="mt-2 flex items-center gap-2 rounded-2xl border border-[var(--color-border)] bg-white px-4 py-3 shadow-sm">
              <Search className="h-4 w-4 shrink-0 text-[var(--color-secondary)]" aria-hidden />
              <input
                value={manualAddQuery}
                onChange={(event) => setManualAddQuery(event.target.value)}
                onFocus={() => setManualAddSearchInputFocused(true)}
                placeholder={t('recipesPage.manualSearchPlaceholder')}
                className="w-full bg-transparent text-sm font-semibold text-[var(--color-primary)] placeholder:text-[var(--color-border)] outline-none"
              />
              <div className="text-xs font-semibold text-[var(--color-secondary)]">
                {isManualAddSearching ? t('recipesPage.searching') : null}
              </div>
            </div>

            {manualAddSearchError ? (
              <div className="mt-2 rounded-2xl border border-[var(--color-active-border)] bg-[var(--color-active-bg)] p-3 text-sm font-semibold text-[var(--color-secondary)]">
                {manualAddSearchError}
              </div>
            ) : null}

            {showManualAddSearchDropdown ? (
              showManualAddRecentPanel ? (
                <ProductSearchDropdown
                  variant="recent-only"
                  allRecentProducts={recentProductPicks}
                  matchingRecents={[]}
                  catalogSuggestions={[]}
                  isSearching={false}
                  onPick={openPortionModalForProduct}
                  listKeyPrefix="manual-add"
                  maxHeightClass="max-h-72"
                />
              ) : (
                <ProductSearchDropdown
                  variant="typed-search"
                  allRecentProducts={[]}
                  matchingRecents={manualAddMatchingRecents}
                  catalogSuggestions={manualAddCatalogDeduped}
                  isSearching={isManualAddSearching}
                  onPick={openPortionModalForProduct}
                  listKeyPrefix="manual-add"
                  maxHeightClass="max-h-72"
                />
              )
            ) : null}
          </div>
        </div>
      ) : null}

      {/* ---------- Job result: recipe cards; “Consume” posts to /api/consume ---------- */}
      {recipes.length > 0 ? (
        <div className="space-y-4 rounded-2xl border border-[var(--color-border)] bg-white p-4 shadow-sm sm:p-6">
          <h2 className="text-sm font-bold uppercase tracking-wide text-[var(--color-primary)]">
            {t('recipesPage.generatedRecipesTitle')}
          </h2>

          <div className="grid gap-4 lg:grid-cols-3">
            {recipes.map((recipe, index) => (
              <div key={`${recipe.title}-${index}`} className="rounded-2xl border border-[var(--color-border)] bg-[var(--color-surface)] p-4 shadow-sm">
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <div className="text-base font-bold text-[var(--color-primary)]">{recipe.title}</div>
                    <div className="mt-1 text-xs font-semibold text-[var(--color-secondary)]">
                      {t('recipesPage.perServingEstimate')}
                    </div>
                  </div>
                </div>

                <div className="mt-4 grid gap-2 sm:grid-cols-2">
                  {(
                    [
                      { metric: 'kcal' as const, value: metricValue(recipe, 'kcal') },
                      { metric: 'protein' as const, value: metricValue(recipe, 'protein') },
                      { metric: 'carbs' as const, value: metricValue(recipe, 'carbs') },
                      { metric: 'fat' as const, value: metricValue(recipe, 'fat') },
                    ] as const
                  ).map((m) => (
                    <div key={m.metric} className={`rounded-xl px-3 py-2 text-sm font-semibold ${metricColor(m.metric)}`}>
                      {metricLabel(m.metric)}: {m.value == null ? '—' : m.value}
                      {m.value == null ? '' : metricUnit(m.metric)}
                    </div>
                  ))}
                </div>

                <div className="mt-4 text-xs font-semibold text-[var(--color-secondary)]">
                  {macroAccuracyLine(recipe.macro_breakdown, t)}
                </div>

                <div className="mt-4">
                  <button
                    type="button"
                    className="w-full rounded-xl border border-slate-200 bg-white px-4 py-2.5 text-sm font-semibold text-[var(--color-secondary)] transition hover:bg-[var(--color-surface-soft)]"
                    onClick={() => {
                      setOpenRecipeStepsIndex((current) => (current === index ? null : index))
                    }}
                  >
                    {openRecipeStepsIndex === index ? t('recipesPage.hidePreparation') : t('recipesPage.viewPreparation')}
                  </button>
                </div>

                {openRecipeStepsIndex === index ? (
                  <div className="mt-3 rounded-2xl border border-[var(--color-border)] bg-white p-3">
                    <div className="text-sm font-bold text-[var(--color-primary)]">{t('recipesPage.necessaryIngredients')}</div>
                    {recipe.ingredients && recipe.ingredients.length > 0 ? (
                      <ul className="mt-2 space-y-2">
                        {recipe.ingredients.map((ing, idx) => (
                          <li
                            key={`${ing.name}-${idx}`}
                            className="flex items-start justify-between gap-3 rounded-xl bg-[var(--color-surface-soft)] px-3 py-2"
                          >
                            <div className="min-w-0">
                              <p className="truncate font-semibold text-[var(--color-primary)]">{ing.name}</p>
                              <p className="mt-0.5 text-xs font-semibold text-[var(--color-secondary)]">{ing.quantity_g} g</p>
                            </div>
                            <span className="whitespace-nowrap rounded-lg bg-white px-2 py-1 text-[11px] font-semibold text-[var(--color-secondary)]">
                              {ingredientSourceLabel(ing.source, t)}
                            </span>
                          </li>
                        ))}
                      </ul>
                    ) : (
                      <p className="mt-2 text-sm font-semibold text-[var(--color-secondary)]">{t('recipesPage.noIngredientBreakdown')}</p>
                    )}

                    <div className="mt-4 text-sm font-bold text-[var(--color-primary)]">{t('recipesPage.preparationSteps')}</div>
                    {recipe.preparation_steps && recipe.preparation_steps.length > 0 ? (
                      <ol className="mt-2 space-y-2">
                        {recipe.preparation_steps.map((step, stepIndex) => (
                          <li
                            key={`${recipe.title}-step-${stepIndex}`}
                            className="rounded-xl border border-[var(--color-border)] bg-white px-3 py-2 text-sm font-semibold text-[var(--color-secondary)]"
                          >
                            <span className="mr-2 font-semibold text-[var(--color-primary)]">{stepIndex + 1}.</span>
                            {step}
                          </li>
                        ))}
                      </ol>
                    ) : (
                      <p className="mt-2 text-sm font-semibold text-[var(--color-secondary)]">{t('recipesPage.noStepBreakdown')}</p>
                    )}
                  </div>
                ) : null}

                <button
                  type="button"
                  disabled={consumeBusyIndex === index}
                  onClick={() => void consumeRecipe(recipe, index)}
                  className="mt-5 w-full rounded-xl bg-gradient-to-r from-emerald-600 to-sky-600 px-4 py-3 text-sm font-semibold uppercase tracking-wide text-white shadow-sm transition hover:from-emerald-700 hover:to-sky-700 disabled:cursor-not-allowed disabled:opacity-60"
                >
                  {consumeBusyIndex === index ? t('recipesPage.logging') : t('recipesPage.consumeLogRecipe')}
                </button>
              </div>
            ))}
          </div>
        </div>
      ) : null}

      {/* ---------- Modal: replace product (autocomplete and/or vision alternatives) ---------- */}
      {activeAlternativeIndex !== null ? (
        <div
          className="fixed inset-0 z-[80] flex items-center justify-center bg-slate-950/70 p-4"
          onClick={closeAlternativeModal}
          role="presentation"
        >
          <div
            className="flex max-h-[92dvh] w-full max-w-lg flex-col overflow-hidden rounded-2xl bg-white shadow-xl"
            onClick={(event) => event.stopPropagation()}
            role="dialog"
            aria-modal="true"
          >
            {(() => {
              const item = visionItems[activeAlternativeIndex]
              if (!item) return null

              const alternatives = item.alternatives ?? []
              const availableChoices: Array<{ name: string; barcode: string }> = [
                { name: item.detectedName || t('recipesPage.detected'), barcode: item.detectedBarcode || '' },
                ...alternatives,
              ]
              const showVisionChoices = activeAlternativeSource === 'scan'

              return (
                <>
                  <div className="shrink-0 px-6 pt-6">
                    <h2 className="text-lg font-bold text-[var(--color-primary)]">{t('recipesPage.selectReplacement')}</h2>
                    <p className="mt-1 text-sm font-semibold text-[var(--color-secondary)]">
                      {t('recipesPage.current')}:{' '}
                      <span className="font-semibold text-[var(--color-primary)]">{item.selectedName}</span>
                    </p>
                  </div>

                  <div className="min-h-0 flex-1 overflow-y-auto overscroll-y-contain px-6 py-4">
                    <div className="space-y-3">
                      <div className="relative" ref={modalAutocompleteRootRef}>
                        <label className="block text-xs font-bold uppercase tracking-wide text-[var(--color-primary)]">
                          {t('recipesPage.manualSearchLabel')}
                        </label>
                        <div className="mt-2 flex items-center gap-2 rounded-2xl border border-[var(--color-border)] bg-white px-4 py-3 shadow-sm">
                          <Search className="h-4 w-4 shrink-0 text-[var(--color-secondary)]" aria-hidden />
                          <input
                            value={modalSearchQuery}
                            onChange={(event) => setModalSearchQuery(event.target.value)}
                            onFocus={() => setModalSearchInputFocused(true)}
                            placeholder={t('recipesPage.searchPlaceholderShort')}
                            className="w-full bg-transparent text-sm font-semibold text-[var(--color-primary)] placeholder:text-[var(--color-secondary)] outline-none"
                          />
                          <div className="text-xs font-semibold text-[var(--color-secondary)]">
                            {isModalSearching ? t('recipesPage.searching') : null}
                          </div>
                        </div>

                        {modalSearchError ? (
                          <div className="mt-2 rounded-2xl border border-[var(--color-active-border)] bg-[var(--color-active-bg)] p-3 text-sm font-semibold text-[var(--color-secondary)]">
                            {modalSearchError}
                          </div>
                        ) : null}

                        {showModalSearchDropdown ? (
                          showModalRecentPanel ? (
                            <ProductSearchDropdown
                              variant="recent-only"
                              allRecentProducts={recentProductPicks}
                              matchingRecents={[]}
                              catalogSuggestions={[]}
                              isSearching={false}
                              onPick={applyManualSelectionToActiveItem}
                              listKeyPrefix="modal-alt"
                              maxHeightClass="max-h-60"
                            />
                          ) : (
                            <ProductSearchDropdown
                              variant="typed-search"
                              allRecentProducts={[]}
                              matchingRecents={modalMatchingRecents}
                              catalogSuggestions={modalCatalogDeduped}
                              isSearching={isModalSearching}
                              onPick={applyManualSelectionToActiveItem}
                              listKeyPrefix="modal-alt"
                              maxHeightClass="max-h-60"
                            />
                          )
                        ) : null}
                      </div>

                      {showVisionChoices ? (
                        <>
                          <div className="pt-1 text-xs font-bold uppercase tracking-wide text-[var(--color-primary)]">
                            {t('recipesPage.visionAlternatives')}
                          </div>
                          {availableChoices.map((choice, idx) => (
                            <button
                              key={`${choice.barcode}-${idx}`}
                              type="button"
                              onClick={() => setAlternativeSelectionIndex(idx)}
                              className={`w-full rounded-xl border px-4 py-3 text-left transition ${alternativeSelectionIndex === idx
                                ? 'border-[var(--color-active-border)] bg-[var(--color-active-bg)]'
                                : 'border-slate-200 bg-white hover:border-slate-300 hover:bg-[var(--color-surface-soft)]'
                                }`}
                            >
                              <div className="text-sm font-semibold text-[var(--color-primary)]">
                                {choice.name || t('recipesPage.unknown')}
                              </div>
                              <VisionItemMacroLine barcode={choice.barcode} macrosByBarcode={macrosByBarcode} />
                            </button>
                          ))}
                        </>
                      ) : null}
                    </div>
                  </div>

                  <div className="shrink-0 border-t border-slate-100 px-6 py-4">
                    <div className="flex flex-col-reverse gap-2 sm:flex-row sm:justify-end sm:gap-3">
                      <button
                        type="button"
                        onClick={closeAlternativeModal}
                        className="rounded-xl border border-slate-300 px-4 py-3 text-sm font-semibold text-[var(--color-secondary)] transition hover:bg-[var(--color-surface-soft)]"
                      >
                        {t('recipesPage.cancel')}
                      </button>
                      {showVisionChoices ? (
                        <button
                          type="button"
                          onClick={() => applyAlternativeSelection()}
                          className="rounded-xl bg-gradient-to-r from-emerald-600 to-sky-600 px-4 py-3 text-sm font-semibold uppercase tracking-wide text-white shadow-sm transition hover:from-emerald-700 hover:to-sky-700"
                        >
                          {t('recipesPage.apply')}
                        </button>
                      ) : null}
                    </div>
                  </div>
                </>
              )
            })()}
          </div>
        </div>
      ) : null}

      {portionPick !== null ? (
        <div
          className="fixed inset-0 z-[90] flex items-center justify-center bg-slate-950/70 p-4"
          onClick={closePortionModal}
          role="presentation"
        >
          <div
            className="w-full max-w-md rounded-2xl bg-white p-6 shadow-xl"
            onClick={(event) => event.stopPropagation()}
            role="dialog"
            aria-modal="true"
          >
            <h2 className="text-lg font-bold text-[var(--color-primary)]">{t('recipesPage.portionModalTitle')}</h2>
            <p className="mt-2 text-sm font-semibold text-[var(--color-primary)]">{portionPick.name}</p>
            <div className="mt-2">
              <AutocompleteMacroLine suggestion={portionPick} />
            </div>

            {portionSuccessMessage ? (
              <>
                <div className="mt-4 rounded-xl border border-emerald-200 bg-emerald-50 p-4 text-sm font-semibold text-emerald-900">
                  {portionSuccessMessage}
                </div>
                <button
                  type="button"
                  onClick={closePortionModal}
                  className="mt-6 w-full rounded-xl bg-gradient-to-r from-emerald-600 to-sky-600 px-4 py-3 text-sm font-semibold uppercase tracking-wide text-white shadow-sm transition hover:from-emerald-700 hover:to-sky-700"
                >
                  {t('recipesPage.portionDone')}
                </button>
              </>
            ) : (
              <>
                <label
                  className="mt-4 block text-sm font-semibold text-[var(--color-primary)]"
                  htmlFor="portion-grams"
                >
                  {t('recipesPage.portionGramsLabel')}
                </label>
                <input
                  id="portion-grams"
                  type="number"
                  min={1}
                  max={10000}
                  step={10}
                  value={portionGrams}
                  onChange={(event) => setPortionGrams(event.target.value)}
                  className="mt-2 w-full rounded-xl border border-slate-300 bg-white px-4 py-3 text-sm outline-none transition focus:border-[var(--color-secondary)] focus:ring-2 focus:ring-[var(--color-focus-ring)]"
                />
                {portionError ? (
                  <div className="mt-3 rounded-xl border border-[var(--color-active-border)] bg-[var(--color-active-bg)] p-3 text-sm font-semibold text-[var(--color-secondary)]">
                    {portionError}
                  </div>
                ) : null}
                <div className="mt-6 flex flex-col-reverse gap-2 sm:flex-row sm:justify-end sm:gap-3">
                  <button
                    type="button"
                    onClick={closePortionModal}
                    className="rounded-xl border border-slate-300 px-4 py-3 text-sm font-semibold text-[var(--color-secondary)] transition hover:bg-[var(--color-surface-soft)]"
                  >
                    {t('recipesPage.cancel')}
                  </button>
                  <button
                    type="button"
                    disabled={isPortionSubmitting}
                    onClick={() => void submitPortionedProduct()}
                    className="rounded-xl bg-gradient-to-r from-emerald-600 to-sky-600 px-4 py-3 text-sm font-semibold uppercase tracking-wide text-white shadow-sm transition hover:from-emerald-700 hover:to-sky-700 disabled:cursor-not-allowed disabled:opacity-60"
                  >
                    {isPortionSubmitting ? t('recipesPage.logging') : t('recipesPage.confirmPortionLog')}
                  </button>
                </div>
              </>
            )}
          </div>
        </div>
      ) : null}
    </section>
  )
}

