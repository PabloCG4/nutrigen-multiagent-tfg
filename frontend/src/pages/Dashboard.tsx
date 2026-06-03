import { ChevronDown } from 'lucide-react'
import type { TFunction } from 'i18next'
import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { Pie, PieChart, ResponsiveContainer, Tooltip, Cell } from 'recharts'
import { useTranslation } from 'react-i18next'
import { removeConsumedRecipe } from '../api/consume'
import { fetchSummary } from '../api/summary'
import { DeleteButton } from '../components/DeleteButton'
import { useAuth } from '../context/AuthContext'
import type { ConsumedRecipe, SummaryResponse } from '../types/summary'
import { useMenuJobNotification } from '../hooks/useMenuJobNotification'

interface MacroChartDatum {
  /** Stable id for tooltip logic; `name` is translated for display. */
  nameKey: 'protein' | 'carbs' | 'fat' | 'empty'
  name: string
  value: number
  color: string
}

type LandingFeatureId = 'smart' | 'nutrition' | 'progress'

const LANDING_FEATURE_CARDS: ReadonlyArray<{
  id: LandingFeatureId
  titleKey: 'dashboard.featureSmartMenusTitle' | 'dashboard.featureNutritionTitle' | 'dashboard.featureProgressTitle'
  bodyKey: 'dashboard.featureSmartMenusBody' | 'dashboard.featureNutritionBody' | 'dashboard.featureProgressBody'
  expandKey:
  | 'dashboard.featureSmartMenusExpand'
  | 'dashboard.featureNutritionExpand'
  | 'dashboard.featureProgressExpand'
}> = [
    {
      id: 'smart',
      titleKey: 'dashboard.featureSmartMenusTitle',
      bodyKey: 'dashboard.featureSmartMenusBody',
      expandKey: 'dashboard.featureSmartMenusExpand',
    },
    {
      id: 'nutrition',
      titleKey: 'dashboard.featureNutritionTitle',
      bodyKey: 'dashboard.featureNutritionBody',
      expandKey: 'dashboard.featureNutritionExpand',
    },
    {
      id: 'progress',
      titleKey: 'dashboard.featureProgressTitle',
      bodyKey: 'dashboard.featureProgressBody',
      expandKey: 'dashboard.featureProgressExpand',
    },
  ]

/** Strip: protein → carbs → fat by share of kcal from macros (matches pie). */
function buildFoodCaloriesBackground(m: {
  proteinKcal: number
  carbsKcal: number
  fatKcal: number
  totalConsumedKcal: number
} | null): string {
  if (!m || m.totalConsumedKcal <= 0) {
    return 'var(--color-surface-soft)'
  }
  const { proteinKcal, carbsKcal, totalConsumedKcal } = m
  const pEnd = (proteinKcal / totalConsumedKcal) * 100
  const cEnd = pEnd + (carbsKcal / totalConsumedKcal) * 100
  return `linear-gradient(to right, var(--color-protein) 0%, var(--color-protein) ${pEnd}%, var(--color-carbs) ${pEnd}%, var(--color-carbs) ${cEnd}%, var(--color-fat) ${cEnd}%, var(--color-fat) 100%)`
}

/** Strip: protein → carbs → fat → exercise (strength red / cardio purple / legacy magenta), proportional to kcal. */
function buildRemainingCaloriesBackground(
  m: {
    proteinKcal: number
    carbsKcal: number
    fatKcal: number
    totalConsumedKcal: number
  } | null,
  exerciseKcal: number,
  strengthKcal: number,
  cardioKcal: number,
): string {
  const ex = Math.max(0, exerciseKcal)
  const s = Math.max(0, strengthKcal)
  const c = Math.max(0, cardioKcal)
  const pk = m?.proteinKcal ?? 0
  const ck = m?.carbsKcal ?? 0
  const fk = m?.fatKcal ?? 0
  const total = pk + ck + fk + ex
  if (total <= 0) {
    return 'linear-gradient(to right, var(--color-protein) 0% 25%, var(--color-carbs) 25% 50%, var(--color-fat) 50% 75%, var(--color-exercise) 75% 100%)'
  }
  const pEnd = (pk / total) * 100
  const cEnd = pEnd + (ck / total) * 100
  const fEnd = cEnd + (fk / total) * 100
  if (ex <= 0) {
    return `linear-gradient(to right, var(--color-protein) 0%, var(--color-protein) ${pEnd}%, var(--color-carbs) ${pEnd}%, var(--color-carbs) ${cEnd}%, var(--color-fat) ${cEnd}%, var(--color-fat) 100%)`
  }
  const W = 100 - fEnd
  const s1 = fEnd + (ex > 0 ? (s / ex) * W : 0)
  const s2 = s1 + (ex > 0 ? (c / ex) * W : 0)
  return `linear-gradient(to right, var(--color-protein) 0%, var(--color-protein) ${pEnd}%, var(--color-carbs) ${pEnd}%, var(--color-carbs) ${cEnd}%, var(--color-fat) ${cEnd}%, var(--color-fat) ${fEnd}%, var(--color-exercise-strength) ${fEnd}%, var(--color-exercise-strength) ${s1}%, var(--color-exercise-cardio) ${s1}%, var(--color-exercise-cardio) ${s2}%, var(--color-exercise) ${s2}%, var(--color-exercise) 100%)`
}

/** Full-width gradient for the exercise summary row (strength / cardio / legacy). */
function buildExerciseRowBackground(exerciseKcal: number, strengthKcal: number, cardioKcal: number): string {
  const ex = Math.max(0, exerciseKcal)
  const s = Math.max(0, strengthKcal)
  const c = Math.max(0, cardioKcal)
  if (ex <= 0) {
    return 'var(--color-surface-soft)'
  }
  const s0 = (s / ex) * 100
  const s1 = s0 + (c / ex) * 100
  return `linear-gradient(to right, var(--color-exercise-strength) 0%, var(--color-exercise-strength) ${s0}%, var(--color-exercise-cardio) ${s0}%, var(--color-exercise-cardio) ${s1}%, var(--color-exercise) ${s1}%, var(--color-exercise) 100%)`
}

function isCheatMealTitle(title: string): boolean {
  return /cheat\s*meal/i.test(title.trim())
}

// Detects backend 401 / invalid JWT messages so we can clear stale sessions login the user out 
// Used to handle authentication failures gracefully when the frontend thinks there is a session but the API does not accept it.
function isAuthenticationFailureMessage(message: string): boolean {
  const trimmed = message.trim()
  if (trimmed === 'Could not validate credentials.') {
    return true
  }
  const lower = trimmed.toLowerCase()
  if (lower.includes('could not validate credentials')) {
    return true
  }
  if (trimmed.includes('HTTP 401')) {
    return true
  }
  return false
}

const LEGACY_MANUAL_JUSTIFICATION_EN = 'Manual product portion logged from Recipes.'

function formatRecipeNotesText(recipe: ConsumedRecipe, t: TFunction): string | null {
  if (recipe.justification_key === 'manual_product_portion') {
    return t('recipesPage.justificationManualProductPortion')
  }
  if (recipe.justification === LEGACY_MANUAL_JUSTIFICATION_EN) {
    return t('recipesPage.justificationManualProductPortion')
  }
  if (recipe.justification) return recipe.justification
  return null
}

function ingredientSourceLabel(source: string, t: TFunction): string {
  return t(`recipesPage.ingredientSource.${source}`, { defaultValue: source })
}

export function Dashboard() {
  const { isAuthenticated, token, logout } = useAuth()
  const { renderMenuNotificationToast } = useMenuJobNotification()
  const { t, i18n } = useTranslation()
  // useState uses the default value at first, then updates it when the setter function is called.
  const [summary, setSummary] = useState<SummaryResponse | null>(null)
  // useState is a hook that allows us to store a value in the component's state and re-render the component when the state changes. 
  const [isLoading, setIsLoading] = useState<boolean>(true)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [activeRecipe, setActiveRecipe] = useState<ConsumedRecipe | null>(null)
  const [activeRecipeIndex, setActiveRecipeIndex] = useState<number | null>(null)
  const [removeRecipeBusy, setRemoveRecipeBusy] = useState(false)
  const [removeRecipeError, setRemoveRecipeError] = useState<string | null>(null)
  const [openLandingFeatureId, setOpenLandingFeatureId] = useState<LandingFeatureId | null>(null)

  useEffect(() => {
    // If the user doesnt have permissions to access the dashboard, reset the variables
    if (!isAuthenticated || !token) {
      setSummary(null)
      setErrorMessage(null)
      setIsLoading(false)
      return
    }

    // AbortController is used to cancel the request if the user navigates away from the page.
    const controller = new AbortController()

    const loadSummary = async (): Promise<void> => {
      setIsLoading(true)
      setErrorMessage(null)
      try {
        // Controller.signal is used to cancel the request if the user navigates away from the page.
        const payload = await fetchSummary(token, controller.signal)
        setSummary(payload)
      } catch (error) {
        if (error instanceof DOMException && error.name === 'AbortError') {
          return
        }
        const detail =
          error instanceof Error ? error.message : 'Unexpected error while loading dashboard.'
        // If the error is an authentication failure, log the user out
        if (isAuthenticationFailureMessage(detail)) {
          logout()
          return
        }
        setErrorMessage(detail)
      } finally {
        setIsLoading(false)
      }
    }

    void loadSummary()
    return () => controller.abort()
  }, [isAuthenticated, token, logout]) // It depends on logout because we need to log the user out if the authentication fails.

  // macroComputation is used to calculate the macro distribution of the user's consumed food.
  const macroComputation = useMemo(() => {
    if (!summary) return null

    const proteinG = Math.max(summary.consumed.protein_g, 0)
    const carbsG = Math.max(summary.consumed.carbs_g, 0)
    const fatG = Math.max(summary.consumed.fat_g, 0)

    const proteinKcal = proteinG * 4
    const carbsKcal = carbsG * 4
    const fatKcal = fatG * 9
    const totalConsumedKcal = proteinKcal + carbsKcal + fatKcal

    const proteinPct = totalConsumedKcal > 0 ? (proteinKcal / totalConsumedKcal) * 100 : 0
    const carbsPct = totalConsumedKcal > 0 ? (carbsKcal / totalConsumedKcal) * 100 : 0
    const fatPct = totalConsumedKcal > 0 ? (fatKcal / totalConsumedKcal) * 100 : 0

    return {
      proteinG,
      carbsG,
      fatG,
      proteinKcal,
      carbsKcal,
      fatKcal,
      totalConsumedKcal,
      proteinPct,
      carbsPct,
      fatPct,
    }
  }, [summary])

  // MacroChartDatum is a type from the recharts library that is used to create the macro distribution chart.
  const macroChartData = useMemo<MacroChartDatum[]>(() => {
    if (!macroComputation) return []

    if (macroComputation.totalConsumedKcal === 0) {
      return [
        {
          nameKey: 'empty',
          name: t('dashboard.macroChartEmpty'),
          value: 1,
          color: '#E5E7EB',
        },
      ]
    }

    return [
      {
        nameKey: 'protein',
        name: t('history.protein'),
        value: macroComputation.proteinKcal,
        color: 'var(--color-protein)',
      },
      {
        nameKey: 'carbs',
        name: t('history.carbs'),
        value: macroComputation.carbsKcal,
        color: 'var(--color-carbs)',
      },
      {
        nameKey: 'fat',
        name: t('history.fat'),
        value: macroComputation.fatKcal,
        color: 'var(--color-fat)',
      },
    ]
  }, [macroComputation, t, i18n.language])

  // If the user is not authenticated, show the login page
  if (!isAuthenticated) {
    return (
      <section className="space-y-10">
        {/* Article is a semantic HTML element that is used to group related content. */}
        <article className="overflow-hidden rounded-3xl border border-[var(--color-border)] bg-gradient-to-br from-emerald-700 via-teal-600 to-sky-600 p-8 text-white shadow-sm sm:p-12">
          <p className="text-sm font-semibold uppercase tracking-[0.24em] text-white/85">{t('dashboard.heroBadge')}</p>
          <h1 className="mt-3 max-w-3xl text-3xl font-semibold leading-tight sm:text-4xl">
            {t('dashboard.heroTitle')}
          </h1>
          <p className="mt-4 max-w-2xl text-sm text-white/85 sm:text-base">{t('dashboard.heroSubtitle')}</p>
          <div className="mt-8">
            <Link
              to="/login"
              className="inline-flex items-center rounded-xl bg-white px-5 py-3 text-sm font-semibold text-[var(--color-primary)] shadow-sm transition hover:bg-[var(--color-surface-soft)]"
            >
              {t('dashboard.heroCta')}
            </Link>
          </div>
        </article>

        <div className="grid gap-4 sm:grid-cols-3">
          {LANDING_FEATURE_CARDS.map((card) => {
            const isOpen = openLandingFeatureId === card.id
            return (
              <button
                key={card.id}
                type="button"
                onClick={() =>
                  setOpenLandingFeatureId((current) => (current === card.id ? null : card.id))
                }
                className="rounded-2xl border border-[var(--color-border)] bg-[var(--color-surface)] p-5 text-left shadow-sm transition-colors hover:bg-[var(--color-surface-soft)] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--color-focus-ring)]"
                aria-expanded={isOpen}
              >
                <div className="flex items-start justify-between gap-2">
                  <h2 className="text-sm font-semibold uppercase tracking-wide text-[var(--color-primary)]">
                    {t(card.titleKey)}
                  </h2>
                  <ChevronDown
                    className={`mt-0.5 h-5 w-5 shrink-0 text-[var(--color-secondary)] transition-transform ${isOpen ? 'rotate-180' : ''}`}
                    aria-hidden
                  />
                </div>
                <p className="mt-2 text-sm text-[var(--color-secondary)]">{t(card.bodyKey)}</p>
                {isOpen ? (
                  <p className="mt-3 border-t border-[var(--color-border)] pt-3 text-sm leading-relaxed text-[var(--color-secondary)]">
                    {t(card.expandKey)}
                  </p>
                ) : null}
              </button>
            )
          })}
        </div>
      </section>
    )
  }

  if (isLoading) {
    return (
      <section className="rounded-2xl border border-[var(--color-border)] bg-[var(--color-surface)] p-6 shadow-sm">
        <h1 className="text-xl font-bold text-[var(--color-primary)]">{t('dashboard.loadingTitle')}</h1>
        <p className="mt-3 text-sm text-[var(--color-secondary)]">{t('dashboard.loadingBody')}</p>
      </section>
    )
  }

  if (errorMessage) {
    return (
      <section className="rounded-2xl border border-[var(--color-active-border)] bg-[var(--color-active-bg)] p-6">
        <h1 className="text-xl font-bold text-[var(--color-primary)]">{t('dashboard.loadingTitle')}</h1>
        <p className="mt-3 text-sm font-semibold text-[var(--color-secondary)]">{errorMessage}</p>
      </section>
    )
  }

  // If the summary is not found, show the empty state
  if (!summary) {
    return (
      <section className="rounded-2xl border border-[var(--color-border)] bg-[var(--color-surface)] p-6 shadow-sm">
        <h1 className="text-xl font-semibold">{t('dashboard.emptyTitle')}</h1>
        <p className="mt-3 text-sm text-[var(--color-secondary)]">{t('dashboard.emptyBody')}</p>
      </section>
    )
  }

  // If the profile is not found, show the profile missing state
  if (!summary.profile_found) {
    return (
      <section className="rounded-2xl border border-[var(--color-active-border)] bg-[var(--color-active-bg)] p-6">
        <h1 className="text-xl font-bold text-[var(--color-primary)]">{t('dashboard.profileMissingTitle')}</h1>
        <p className="mt-3 text-sm font-bold text-[var(--color-secondary)]">
          {t('dashboard.profileMissingBody')}
        </p>
      </section>
    )
  }

  const closeRecipeModal = () => {
    setActiveRecipe(null)
    setActiveRecipeIndex(null)
    setRemoveRecipeError(null)
  }

  const handleRemoveConsumedRecipe = async () => {
    if (!token || !summary || activeRecipeIndex === null) return
    setRemoveRecipeBusy(true)
    setRemoveRecipeError(null)
    try {
      await removeConsumedRecipe(token, { date: summary.date, index: activeRecipeIndex })
      const controller = new AbortController()
      const payload = await fetchSummary(token, controller.signal)
      setSummary(payload)
      closeRecipeModal()
    } catch (error) {
      const detail =
        error instanceof Error ? error.message : 'Failed to remove recipe from daily log.'
      if (isAuthenticationFailureMessage(detail)) {
        logout()
        return
      }
      setRemoveRecipeError(detail)
    } finally {
      setRemoveRecipeBusy(false)
    }
  }

  // this return is the main content of the dashboard, if everything is ok (user logged in, profile found, summary found)
  return (
    <section className="space-y-6">
      <h1 className="text-xl font-bold text-[var(--color-primary)] sm:text-2xl">{t('dashboard.title')}</h1>

      {/* This is the main content of the dashboard, it is a grid with two columns, the first column is the macro distribution chart, the second column is the macro goals and the daily calories */}
      <div className="grid gap-6 lg:grid-cols-[1fr_1.2fr]">
        <article className="rounded-2xl border border-[var(--color-border)] bg-[var(--color-surface)] p-4 shadow-sm sm:p-6">
          <h2 className="mb-4 text-sm font-bold uppercase tracking-wide text-[var(--color-primary)]">{t('dashboard.macroDistributionTitle')}</h2>
          <div className="relative h-64 w-full sm:h-72 [&_.recharts-pie-sector]:outline-none [&_.recharts-pie-sector:focus]:outline-none [&_.recharts-pie-sector:focus-visible]:outline-none">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={macroChartData}
                  dataKey="value"
                  nameKey="name"
                  innerRadius={65}
                  outerRadius={95}
                  paddingAngle={2}
                  activeShape={false}
                  stroke="#fff"
                  strokeWidth={2}
                >
                  {macroChartData.map((entry) => (
                    <Cell key={entry.nameKey} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip
                  formatter={(value, _name, item) => {
                    const payload = (item as { payload?: MacroChartDatum } | undefined)?.payload
                    const numericValue =
                      typeof value === 'number' ? value : Number.parseFloat(String(value))
                    if (!Number.isFinite(numericValue)) {
                      return [`0 kcal`, payload?.name ?? '']
                    }

                    if (!macroComputation) {
                      return [`${numericValue.toFixed(0)} kcal`, payload?.name ?? '']
                    }

                    if (!payload) {
                      return [`${numericValue.toFixed(0)} kcal`, '']
                    }

                    if (payload.nameKey === 'empty') {
                      return ['0 kcal', payload.name]
                    }
                    if (payload.nameKey === 'protein') {
                      return [
                        `${macroComputation.proteinG.toFixed(1)} g (${numericValue.toFixed(0)} kcal)`,
                        payload.name,
                      ]
                    }
                    if (payload.nameKey === 'carbs') {
                      return [
                        `${macroComputation.carbsG.toFixed(1)} g (${numericValue.toFixed(0)} kcal)`,
                        payload.name,
                      ]
                    }
                    if (payload.nameKey === 'fat') {
                      return [
                        `${macroComputation.fatG.toFixed(1)} g (${numericValue.toFixed(0)} kcal)`,
                        payload.name,
                      ]
                    }

                    return [`${numericValue.toFixed(0)} kcal`, payload.name]
                  }}
                />
              </PieChart>
            </ResponsiveContainer>
            {macroComputation ? (
              <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
                <div className="text-center text-xs font-bold leading-snug text-[var(--color-app-text)]">
                  <p className="m-0">
                    {t('history.protein')} {macroComputation.proteinPct.toFixed(0)}%
                  </p>
                  <p className="m-0 mt-0.5">
                    {t('history.carbs')} {macroComputation.carbsPct.toFixed(0)}%
                  </p>
                  <p className="m-0 mt-0.5">
                    {t('history.fat')} {macroComputation.fatPct.toFixed(0)}%
                  </p>
                </div>
              </div>
            ) : null}
          </div>
          {/* If the macro computation is found, show the macro goals */}
          {macroComputation ? (
            <div className="mt-6">
              <h3 className="text-sm font-bold uppercase tracking-wide text-[var(--color-primary)]">
                {t('dashboard.macroGoalsTitle')}
              </h3>

              {(() => {
                const targetProteinG = Math.max(summary.targets.protein_g, 0)
                const targetCarbsG = Math.max(summary.targets.carbs_g, 0)
                const targetFatG = Math.max(summary.targets.fat_g, 0)

                const proteinProgress =
                  targetProteinG > 0 ? Math.min(100, (macroComputation.proteinG / targetProteinG) * 100) : 0
                const carbsProgress =
                  targetCarbsG > 0 ? Math.min(100, (macroComputation.carbsG / targetCarbsG) * 100) : 0
                const fatProgress =
                  targetFatG > 0 ? Math.min(100, (macroComputation.fatG / targetFatG) * 100) : 0

                return (
                  <div className="mt-4 space-y-4">
                    <div>
                      <div className="flex items-center justify-between text-xs">
                        <span className="font-semibold text-[var(--color-protein)]">{t('history.protein')}</span>
                        <span className="font-semibold" style={{ color: 'var(--color-protein)' }}>
                          {macroComputation.proteinG.toFixed(1)} g / {targetProteinG.toFixed(0)} g
                        </span>
                      </div>
                      <div className="mt-2 h-3.5 overflow-hidden rounded-full bg-[var(--color-surface-soft)]">
                        <div
                          className="h-full rounded-full"
                          style={{ width: `${proteinProgress}%`, backgroundColor: 'var(--color-protein)' }}
                        />
                      </div>
                    </div>

                    <div>
                      <div className="flex items-center justify-between text-xs">
                        <span className="font-semibold text-[var(--color-carbs)]">{t('history.carbs')}</span>
                        <span className="font-semibold" style={{ color: 'var(--color-carbs)' }}>
                          {macroComputation.carbsG.toFixed(1)} g / {targetCarbsG.toFixed(0)} g
                        </span>
                      </div>
                      <div className="mt-2 h-3.5 overflow-hidden rounded-full bg-[var(--color-surface-soft)]">
                        <div
                          className="h-full rounded-full"
                          style={{ width: `${carbsProgress}%`, backgroundColor: 'var(--color-carbs)' }}
                        />
                      </div>
                    </div>

                    <div>
                      <div className="flex items-center justify-between text-xs">
                        <span className="font-semibold text-[var(--color-fat)]">{t('history.fat')}</span>
                        <span className="font-semibold" style={{ color: 'var(--color-fat)' }}>
                          {macroComputation.fatG.toFixed(1)} g / {targetFatG.toFixed(0)} g
                        </span>
                      </div>
                      <div className="mt-2 h-3.5 overflow-hidden rounded-full bg-[var(--color-surface-soft)]">
                        <div
                          className="h-full rounded-full"
                          style={{ width: `${fatProgress}%`, backgroundColor: 'var(--color-fat)' }}
                        />
                      </div>
                    </div>
                  </div>
                )
              })()}
            </div>
          ) : null}
        </article>

        {/* This is the daily calories section, it shows the daily calories target, the calories consumed, the calories burned and the calories remaining */}
        <article className="rounded-2xl border border-[var(--color-border)] bg-[var(--color-surface)] p-4 shadow-sm sm:p-6">
          <h2 className="mb-4 text-sm font-bold uppercase tracking-wide text-[var(--color-primary)]">
            {t('dashboard.dailyCaloriesTitle')}
          </h2>
          <dl className="space-y-3 text-sm sm:text-base">
            <div
              className="flex items-center justify-between rounded-lg px-3 py-2 text-black bg-[var(--color-surface-soft)]"
            >
              <dt className="font-semibold">{t('dashboard.caloriesTarget')}</dt>
              <dd className="font-semibold">{summary.targets.calories.toFixed(0)} kcal</dd>
            </div>
            <div
              className="flex items-center justify-between rounded-lg px-3 py-2 text-black"
              style={{ background: buildFoodCaloriesBackground(macroComputation) }}
            >
              <dt className="font-semibold">{t('dashboard.caloriesFood')}</dt>
              <dd className="font-semibold">{summary.consumed.calories.toFixed(0)} kcal</dd>
            </div>
            <div
              className={`flex flex-col gap-1 rounded-lg px-3 py-2 text-black sm:flex-row sm:items-center sm:justify-between ${summary.consumed.burned_calories <= 0 ? 'bg-[var(--color-surface-soft)]' : ''
                }`}
              style={
                summary.consumed.burned_calories > 0
                  ? {
                    background: buildExerciseRowBackground(
                      summary.consumed.burned_calories,
                      summary.consumed.burned_calories_strength ?? 0,
                      summary.consumed.burned_calories_cardio ?? 0,
                    ),
                  }
                  : undefined
              }
            >
              <dt className="font-semibold">{t('dashboard.caloriesExercise')}</dt>
              <dd className="flex flex-col items-end gap-0.5 font-semibold sm:items-end">
                <span>{summary.consumed.burned_calories.toFixed(0)} kcal</span>
                {summary.consumed.burned_calories > 0 ? (
                  <span className="text-xs font-semibold text-black/90">
                    {t('dashboard.exerciseBreakdown', {
                      strength: (summary.consumed.burned_calories_strength ?? 0).toFixed(0),
                      cardio: (summary.consumed.burned_calories_cardio ?? 0).toFixed(0),
                    })}
                  </span>
                ) : null}
              </dd>
            </div>
            {(() => {
              const targetCal = Math.max(0, summary.targets.calories)
              const foodCal = Math.max(0, summary.consumed.calories)
              const burnedCal = Math.max(0, summary.consumed.burned_calories)
              const consumedTotal = foodCal + burnedCal
              const fillPct =
                targetCal > 0
                  ? Math.min(100, (consumedTotal / targetCal) * 100)
                  : consumedTotal > 0
                    ? 100
                    : 0
              const isFullWidth = fillPct >= 99.5
              return (
                <div className="relative overflow-hidden rounded-lg bg-[var(--color-surface-soft)]">
                  {fillPct > 0 ? (
                    <div
                      aria-hidden
                      className={`pointer-events-none absolute inset-y-0 left-0 ${isFullWidth ? 'rounded-lg' : 'rounded-l-lg'}`}
                      style={{
                        width: `${fillPct}%`,
                        background: buildRemainingCaloriesBackground(
                          macroComputation,
                          summary.consumed.burned_calories,
                          summary.consumed.burned_calories_strength ?? 0,
                          summary.consumed.burned_calories_cardio ?? 0,
                        ),
                      }}
                    />
                  ) : null}
                  <div className="relative flex items-center justify-between px-3 py-2 text-black">
                    <dt className="font-semibold">{t('dashboard.caloriesRemaining')}</dt>
                    <dd className="font-semibold">{summary.remaining.calories.toFixed(0)} kcal</dd>
                  </div>
                </div>
              )
            })()}
          </dl>
        </article>
      </div>

      {/* This is the recipes used today section, it shows the recipes used today, if there are no recipes used today, show the no recipes used today message */}
      <article className="rounded-2xl border border-[var(--color-border)] bg-[var(--color-surface)] p-4 shadow-sm sm:p-6">
        <div className="mb-4 flex items-start justify-between gap-3">
          <div className="min-w-0">
            <h2 className="text-sm font-bold uppercase tracking-wide text-[var(--color-primary)]">
              {t('dashboard.recipesUsedTodayTitle')}
            </h2>
            <p className="mt-1 text-sm font-semibold text-[var(--color-secondary)]">
              {t('dashboard.recipesUsedTodaySubtitle')}
            </p>
          </div>
          <Link
            to="/recetas"
            className="inline-flex rounded-lg bg-gradient-to-r from-emerald-600 to-sky-600 px-3 py-1.5 text-xs font-semibold text-white shadow-sm transition hover:from-emerald-700 hover:to-sky-700"
          >
            {t('dashboard.viewRecipes')}
          </Link>
        </div>
        {summary.consumed.recipes.length === 0 ? (
          <p className="text-sm text-[var(--color-secondary)]">{t('dashboard.noRecipesLoggedYetToday')}</p>
        ) : (
          <ul className="space-y-2">
            {summary.consumed.recipes.map((recipe, index) => (
              <li
                key={`${recipe.title}-${index}`}
                className="rounded-lg border border-[var(--color-border)] bg-[var(--color-surface-soft)] px-3 py-2 text-sm text-black transition-colors hover:bg-[var(--color-surface-soft)]/50 sm:text-base"
              >
                <button
                  type="button"
                  className="w-full text-left"
                  onClick={() => {
                    setActiveRecipe(recipe)
                    setActiveRecipeIndex(index)
                    setRemoveRecipeError(null)
                  }}
                >
                  <span className="font-bold">{t('dashboard.recipeNumberPrefix', { index: index + 1 })}</span>{' '}
                  <span className="font-normal">{recipe.title}</span>
                </button>
              </li>
            ))}
          </ul>
        )}
      </article>

      {activeRecipe ? (
        <div
          role="dialog"
          aria-modal="true"
          className="fixed inset-0 z-50 flex items-end justify-center bg-black/50 p-4 sm:items-center"
          onMouseDown={(event) => {
            if (event.target === event.currentTarget) closeRecipeModal()
          }}
        >
          <div className="w-full max-w-2xl overflow-hidden rounded-2xl border border-[var(--color-border)] bg-white shadow-lg">
            <div className="flex items-start justify-between gap-3 border-b border-[var(--color-border)] bg-[var(--color-surface)] p-4">
              <div className="min-w-0">
                <h3 className="truncate text-lg font-semibold text-[var(--color-primary)]">{activeRecipe.title}</h3>
                {typeof activeRecipe.estimated_time_minutes === 'number' &&
                  !isCheatMealTitle(activeRecipe.title) ? (
                  <p className="mt-1 text-sm text-[var(--color-secondary)]">
                    {t('dashboard.estimatedTime', { minutes: activeRecipe.estimated_time_minutes })}
                  </p>
                ) : null}
              </div>
              <button
                type="button"
                className="rounded-lg bg-gradient-to-r from-emerald-600 to-sky-600 px-3 py-1.5 text-sm font-semibold text-[var(--color-white)] shadow-sm transition hover:from-emerald-700 hover:to-sky-700"
                onClick={() => closeRecipeModal()}
              >
                {t('dashboard.close')}
              </button>
            </div>

            <div className="max-h-[70vh] overflow-auto p-4 sm:p-6">
              {activeRecipe.macro_breakdown ? (
                <div className="mb-4 grid gap-2 sm:grid-cols-2">
                  <div className="rounded-xl bg-[var(--color-surface-soft)] p-3">
                    <p className="text-xs font-semibold uppercase tracking-wide text-[var(--color-primary)]">
                      {t('dashboard.caloriesLabel')}
                    </p>
                    <p className="text-lg font-semibold text-[var(--color-secondary)]">
                      {activeRecipe.macro_breakdown.total_kcal.toFixed(0)} kcal
                    </p>
                  </div>
                  <div className="rounded-xl bg-[var(--color-surface-soft)] p-3">
                    <p className="text-xs font-semibold uppercase tracking-wide text-[var(--color-primary)]">
                      {t('dashboard.macrosLabel')}
                    </p>
                    <p className="text-sm text-[var(--color-secondary)]">
                      {t('history.protein')}: {activeRecipe.macro_breakdown.protein_g.toFixed(1)} g
                    </p>
                    <p className="text-sm text-[var(--color-secondary)]">
                      {t('history.carbs')}: {activeRecipe.macro_breakdown.carb_g.toFixed(1)} g
                    </p>
                    <p className="text-sm text-[var(--color-secondary)]">
                      {t('history.fat')}: {activeRecipe.macro_breakdown.fat_g.toFixed(1)} g
                    </p>
                  </div>
                </div>
              ) : null}

              {formatRecipeNotesText(activeRecipe, t) ? (
                <div className="mb-4 rounded-xl bg-[var(--color-surface-soft)] p-3">
                  <p className="text-xs font-semibold uppercase tracking-wide text-[var(--color-primary)]">
                    {t('dashboard.notesLabel')}
                  </p>
                  <p className="mt-1 text-sm leading-relaxed text-[var(--color-secondary)]">
                    {formatRecipeNotesText(activeRecipe, t)}
                  </p>
                </div>
              ) : null}

              <div className="mb-4">
                <h4 className="mb-2 text-sm font-semibold text-[var(--color-primary)]">
                  {t('dashboard.ingredientsTitle')}
                </h4>
                {activeRecipe.ingredients && activeRecipe.ingredients.length > 0 ? (
                  <ul className="space-y-1">
                    {activeRecipe.ingredients.map((ing, idx) => (
                      <li
                        key={`${ing.name}-${idx}`}
                        className="flex items-center justify-between gap-3 rounded-lg border border-[var(--color-border)] bg-white px-3 py-2 text-sm"
                      >
                        <div className="min-w-0">
                          <p className="truncate font-medium text-[var(--color-secondary)]">{ing.name}</p>
                          <p className="text-xs text-[var(--color-secondary)]">{ing.quantity_g} g</p>
                        </div>
                        <span className="whitespace-nowrap rounded-lg bg-[var(--color-surface-soft)] px-2 py-1 text-[11px] font-semibold text-[var(--color-secondary)]">
                          {ingredientSourceLabel(ing.source, t)}
                        </span>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="text-sm text-[var(--color-secondary)]">{t('dashboard.noIngredientBreakdown')}</p>
                )}
              </div>

              <div>
                <h4 className="mb-2 text-sm font-semibold text-[var(--color-primary)]">
                  {t('dashboard.preparationStepsTitle')}
                </h4>
                {activeRecipe.preparation_steps && activeRecipe.preparation_steps.length > 0 ? (
                  <ol className="space-y-2">
                    {activeRecipe.preparation_steps.map((step, idx) => (
                      <li
                        key={`${step}-${idx}`}
                        className="rounded-xl border border-[var(--color-border)] bg-white px-3 py-2 text-sm text-[var(--color-secondary)]"
                      >
                        <span className="mr-2 font-semibold">{idx + 1}.</span>
                        {step}
                      </li>
                    ))}
                  </ol>
                ) : (
                  <p className="text-sm text-[var(--color-secondary)]">{t('dashboard.noStepBreakdown')}</p>
                )}
              </div>
            </div>

            <div className="border-t border-[var(--color-border)] bg-[var(--color-surface)] p-4 sm:px-6">
              {removeRecipeError ? (
                <p className="mb-3 text-sm font-semibold text-red-600">{removeRecipeError}</p>
              ) : null}
              <DeleteButton
                disabled={removeRecipeBusy}
                onClick={() => void handleRemoveConsumedRecipe()}
                className="w-full sm:w-auto"
              >
                {t('recipesPage.delete')}
              </DeleteButton>
            </div>
          </div>
        </div>
      ) : null}
      {/* Background menu tracking injection */}
      {renderMenuNotificationToast()}
    </section>
  )
}
