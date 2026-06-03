import { useEffect, useMemo, useRef, useState } from 'react'
import { CartesianGrid, XAxis, YAxis, Tooltip, ResponsiveContainer, LineChart, Line, Area, BarChart, Bar } from 'recharts'
import { useNavigate } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import { getApiBaseUrl } from '../config/apiBaseUrl'
import { apiFetch } from '../api/apiFetch'
import { useAuth } from '../context/AuthContext'
import { usePreferences } from '../context/PreferencesContext'
import { kgToLb } from '../utils/units'
import { useMenuJobNotification } from '../hooks/useMenuJobNotification'

type MetricKey = 'weight' | 'net_calories' | 'protein' | 'carbs' | 'fat' | 'exercise_burn'
type AggregationMode = 'daily' | 'weekly' | 'monthly'
type TimeframeMode = 'last_7' | 'last_30' | 'last_90' | 'last_1y' | 'custom_month'

interface WeightRecord {
  date: string
  weight_kg: number
}

interface NutritionRecord {
  date: string
  consumed_calories: number
  consumed_protein: number
  consumed_carbs: number
  consumed_fat: number
  burned_calories: number
}

interface ChartPoint {
  label: string
  value: number
  dateISO: string
}

interface ExerciseBurnDayRecord {
  date: string
  burned_strength: number
  burned_cardio: number
}

interface ExerciseBurnPoint {
  label: string
  dateISO: string
  strength: number
  cardio: number
  /** strength + cardio (for empty checks / tooltips) */
  value: number
}

function formatDateISO(date: Date): string {
  const year = date.getFullYear()
  const month = String(date.getMonth() + 1).padStart(2, '0')
  const day = String(date.getDate()).padStart(2, '0')
  return `${year}-${month}-${day}`
}

function parseDateISO(dateISO: string): Date {
  const parts = dateISO.split('-').map((p) => Number.parseInt(p, 10))
  const year = parts[0] ?? 0
  const month = (parts[1] ?? 1) - 1
  const day = parts[2] ?? 1
  return new Date(year, month, day)
}

function formatXAxisLabel(dateISO: string, locale: string): string {
  const d = parseDateISO(dateISO)
  return d.toLocaleDateString(locale, { month: 'short', day: 'numeric' })
}

function formatMonthLabel(monthISO: string, locale: string): string {
  const [yearString, monthString] = monthISO.split('-')
  const year = Number.parseInt(yearString, 10)
  const monthIndex = Number.parseInt(monthString, 10) - 1
  const d = new Date(year, monthIndex, 1)
  return d.toLocaleDateString(locale, { month: 'short', year: 'numeric' })
}

function formatNumber(value: number, locale: string, maximumFractionDigits = 1): string {
  return value.toLocaleString(locale, { maximumFractionDigits })
}

function mondayStart(date: Date): Date {
  const d = new Date(date)
  const day = d.getDay()
  const diffToMonday = (day + 6) % 7
  d.setDate(d.getDate() - diffToMonday)
  d.setHours(0, 0, 0, 0)
  return d
}

function addDays(date: Date, days: number): Date {
  const d = new Date(date)
  d.setDate(d.getDate() + days)
  return d
}

function buildMonthOptions(startISO: string, endISO: string): string[] {
  const start = parseDateISO(startISO)
  const end = parseDateISO(endISO)
  const monthOptions: string[] = []

  const cursor = new Date(start.getFullYear(), start.getMonth(), 1)
  const endCursor = new Date(end.getFullYear(), end.getMonth(), 1)
  cursor.setHours(0, 0, 0, 0)
  endCursor.setHours(0, 0, 0, 0)

  while (cursor <= endCursor) {
    const label = `${cursor.getFullYear()}-${String(cursor.getMonth() + 1).padStart(2, '0')}`
    monthOptions.push(label)
    cursor.setMonth(cursor.getMonth() + 1)
  }

  return monthOptions
}

function groupPoints(
  points: ChartPoint[],
  aggregation: AggregationMode,
  metric: MetricKey,
  locale: string,
): ChartPoint[] {
  if (aggregation === 'daily') {
    return points
  }

  if (aggregation === 'weekly') {
    const groups = new Map<string, ChartPoint[]>()
    for (const point of points) {
      const weekStart = mondayStart(parseDateISO(point.dateISO))
      const weekStartISO = formatDateISO(weekStart)
      const bucket = groups.get(weekStartISO) ?? []
      bucket.push(point)
      groups.set(weekStartISO, bucket)
    }

    const sortedKeys = Array.from(groups.keys()).sort()
    const result: ChartPoint[] = []
    for (const key of sortedKeys) {
      const bucket = groups.get(key) ?? []
      if (bucket.length === 0) continue
      const values = bucket.map((p) => p.value)
      const computedValue =
        metric === 'weight'
          ? values.reduce((sum, v) => sum + v, 0) / values.length
          : values.reduce((sum, v) => sum + v, 0)
      result.push({
        label: formatXAxisLabel(key, locale),
        value: computedValue,
        dateISO: key,
      })
    }
    return result
  }

  const groups = new Map<string, ChartPoint[]>()
  for (const point of points) {
    const d = parseDateISO(point.dateISO)
    const monthISO = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}`
    const bucket = groups.get(monthISO) ?? []
    bucket.push(point)
    groups.set(monthISO, bucket)
  }

  const sortedKeys = Array.from(groups.keys()).sort()
  const result: ChartPoint[] = []
  for (const key of sortedKeys) {
    const bucket = groups.get(key) ?? []
    if (bucket.length === 0) continue
    const values = bucket.map((p) => p.value)
    const computedValue =
      metric === 'weight'
        ? values.reduce((sum, v) => sum + v, 0) / values.length
        : values.reduce((sum, v) => sum + v, 0)
    result.push({
      label: formatMonthLabel(key, locale),
      value: computedValue,
      dateISO: bucket[0]?.dateISO ?? key,
    })
  }
  return result
}

function groupExerciseBurnPoints(
  points: ExerciseBurnPoint[],
  aggregation: AggregationMode,
  locale: string,
): ExerciseBurnPoint[] {
  if (aggregation === 'daily') {
    return points
  }

  if (aggregation === 'weekly') {
    const groups = new Map<string, ExerciseBurnPoint[]>()
    for (const point of points) {
      const weekStart = mondayStart(parseDateISO(point.dateISO))
      const weekStartISO = formatDateISO(weekStart)
      const bucket = groups.get(weekStartISO) ?? []
      bucket.push(point)
      groups.set(weekStartISO, bucket)
    }

    const sortedKeys = Array.from(groups.keys()).sort()
    const result: ExerciseBurnPoint[] = []
    for (const key of sortedKeys) {
      const bucket = groups.get(key) ?? []
      if (bucket.length === 0) continue
      const s = bucket.reduce((sum, p) => sum + p.strength, 0)
      const c = bucket.reduce((sum, p) => sum + p.cardio, 0)
      result.push({
        label: formatXAxisLabel(key, locale),
        dateISO: key,
        strength: s,
        cardio: c,
        value: s + c,
      })
    }
    return result
  }

  const groups = new Map<string, ExerciseBurnPoint[]>()
  for (const point of points) {
    const d = parseDateISO(point.dateISO)
    const monthISO = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}`
    const bucket = groups.get(monthISO) ?? []
    bucket.push(point)
    groups.set(monthISO, bucket)
  }

  const sortedKeys = Array.from(groups.keys()).sort()
  const result: ExerciseBurnPoint[] = []
  for (const key of sortedKeys) {
    const bucket = groups.get(key) ?? []
    if (bucket.length === 0) continue
    const s = bucket.reduce((sum, p) => sum + p.strength, 0)
    const c = bucket.reduce((sum, p) => sum + p.cardio, 0)
    result.push({
      label: formatMonthLabel(key, locale),
      dateISO: bucket[0]?.dateISO ?? key,
      strength: s,
      cardio: c,
      value: s + c,
    })
  }
  return result
}

function isAuthFailureMessage(message: string): boolean {
  const trimmed = message.trim()
  if (trimmed === 'Could not validate credentials.') return true
  const lower = trimmed.toLowerCase()
  return lower.includes('could not validate credentials') || trimmed.includes('HTTP 401')
}

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
    // Ignore parsing errors
  }
  return fallback
}

export function History() {
  const navigate = useNavigate()
  const { isAuthenticated, token, logout } = useAuth()
  const { renderMenuNotificationToast } = useMenuJobNotification()
  const preferences = usePreferences()
  const { t, i18n } = useTranslation()
  const locale = i18n.language === 'es' ? 'es-ES' : 'en-US'

  const [metric, setMetric] = useState<MetricKey>('weight')
  const [timeframeMode, setTimeframeMode] = useState<TimeframeMode>('last_30')
  const [aggregationMode, setAggregationMode] = useState<AggregationMode>('daily')

  const [monthChoice, setMonthChoice] = useState<string>('')
  const [earliestDateISO, setEarliestDateISO] = useState<string>('')

  const [isLoading, setIsLoading] = useState<boolean>(false)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)

  const metricLabel = (key: MetricKey): string => {
    if (key === 'weight') return t('history.weight')
    if (key === 'net_calories') return t('history.netCalories')
    if (key === 'protein') return t('history.protein')
    if (key === 'carbs') return t('history.carbs')
    if (key === 'fat') return t('history.fat')
    return t('history.exerciseBurn')
  }

  const tooltipText = (key: MetricKey, value: number): string => {
    if (key === 'weight') {
      const unitKey = preferences.units === 'imperial' ? 'lbs' : 'kg'
      return `${formatNumber(value, locale, 1)} ${t(`units.${unitKey}`)}`
    }
    if (key === 'net_calories' || key === 'exercise_burn') {
      return `${formatNumber(value, locale)} ${t('units.kcal')}`
    }
    return `${formatNumber(value, locale)} ${t('units.g')}`
  }
  const [points, setPoints] = useState<ChartPoint[]>([])
  const [exerciseBurnPoints, setExerciseBurnPoints] = useState<ExerciseBurnPoint[]>([])
  const [hasLoaded, setHasLoaded] = useState<boolean>(false)
  const hasAppliedDefaultFiltersRef = useRef<boolean>(false)

  useEffect(() => {
    if (!isAuthenticated || !token) {
      navigate('/login')
    }
  }, [isAuthenticated, token, navigate])

  useEffect(() => {
    if (!isAuthenticated || !token) return
    if (hasAppliedDefaultFiltersRef.current) return

    // Default filters required for the initial UX.
    setMetric('weight')
    setTimeframeMode('last_30')
    setAggregationMode('daily')
    setHasLoaded(false)

    hasAppliedDefaultFiltersRef.current = true
  }, [isAuthenticated, token])

  const todayISO = useMemo(() => formatDateISO(new Date()), [])

  useEffect(() => {
    if (!isAuthenticated || !token) {
      return
    }

    const controller = new AbortController()
    const loadWeightForEarliest = async (): Promise<void> => {
      try {
        const response = await apiFetch(`${getApiBaseUrl()}/api/history/weight`, {
          headers: { Authorization: `Bearer ${token}` },
          signal: controller.signal,
        })
        if (!response.ok) {
          const detail = await readErrorDetail(
            response,
            `Failed to load weight history (HTTP ${response.status}).`,
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
          !('records' in payload) ||
          !Array.isArray((payload as { records: unknown[] }).records)
        ) {
          return
        }

        const records = (payload as { records: WeightRecord[] }).records
        const sortedDates = records.map((r) => String(r.date)).sort()
        if (sortedDates.length > 0) {
          const minISO = sortedDates[0] ?? ''
          if (minISO) {
            setEarliestDateISO(minISO)
          }
        }
      } catch {
        // Ignore errors while building month options
      }
    }

    void loadWeightForEarliest()
    return () => controller.abort()
  }, [isAuthenticated, token, logout])

  useEffect(() => {
    const fallbackStart = (() => {
      const end = parseDateISO(todayISO)
      const start = addDays(end, -365)
      return formatDateISO(start)
    })()
    const startISO = earliestDateISO || fallbackStart
    const monthOptions = buildMonthOptions(startISO, todayISO)
    if (monthOptions.length === 0) {
      setMonthChoice('')
      return
    }
    if (!monthChoice) {
      setMonthChoice(monthOptions[monthOptions.length - 1] ?? '')
      return
    }
    if (!monthOptions.includes(monthChoice)) {
      setMonthChoice(monthOptions[monthOptions.length - 1] ?? '')
    }
  }, [earliestDateISO, todayISO, monthChoice])

  const dateRange = useMemo(() => {
    const endDate = parseDateISO(todayISO)
    const endISO = todayISO

    const calculateStartFromDays = (days: number): string => {
      const start = addDays(endDate, -days + 1)
      return formatDateISO(start)
    }

    if (timeframeMode === 'last_7') {
      return { start_date: calculateStartFromDays(7), end_date: endISO }
    }
    if (timeframeMode === 'last_30') {
      return { start_date: calculateStartFromDays(30), end_date: endISO }
    }
    if (timeframeMode === 'last_90') {
      return { start_date: calculateStartFromDays(90), end_date: endISO }
    }
    if (timeframeMode === 'last_1y') {
      return { start_date: calculateStartFromDays(365), end_date: endISO }
    }

    const monthISO = monthChoice || `${endDate.getFullYear()}-${String(endDate.getMonth() + 1).padStart(2, '0')}`
    const [yearString, monthString] = monthISO.split('-')
    const year = Number.parseInt(yearString, 10)
    const monthIndex = Number.parseInt(monthString, 10) - 1
    const start = new Date(year, monthIndex, 1)
    const startISO = formatDateISO(start)

    const monthEnd = new Date(year, monthIndex + 1, 0)
    const monthEndISO = formatDateISO(monthEnd)

    const effectiveEndISO = monthEndISO > endISO ? endISO : monthEndISO
    return { start_date: startISO, end_date: effectiveEndISO }
  }, [timeframeMode, monthChoice, todayISO])

  const chartTitle = metricLabel(metric)

  useEffect(() => {
    if (!isAuthenticated || !token) {
      return
    }

    setIsLoading(true)
    setErrorMessage(null)

    const controller = new AbortController()

    const load = async (): Promise<void> => {
      try {
        const filtered: ChartPoint[] = []
        if (metric === 'weight') {
          const response = await apiFetch(`${getApiBaseUrl()}/api/history/weight`, {
            headers: { Authorization: `Bearer ${token}` },
            signal: controller.signal,
          })
          if (!response.ok) {
            const detail = await readErrorDetail(
              response,
              `Failed to load weight history (HTTP ${response.status}).`,
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
            !('records' in payload) ||
            !Array.isArray((payload as { records: unknown[] }).records)
          ) {
            setPoints([])
            setExerciseBurnPoints([])
            return
          }
          const records = (payload as { records: WeightRecord[] }).records
          const startISO = dateRange.start_date
          const endISO = dateRange.end_date
          for (const record of records) {
            const dateISO = String(record.date)
            if (dateISO < startISO || dateISO > endISO) continue
            const baseKg = Number(record.weight_kg ?? 0)
            const value = preferences.units === 'imperial' ? kgToLb(baseKg) : baseKg
            filtered.push({
              label: formatXAxisLabel(dateISO, locale),
              value,
              dateISO,
            })
          }
          filtered.sort((a, b) => (a.dateISO < b.dateISO ? -1 : 1))
          const aggregated = groupPoints(filtered, aggregationMode, metric, locale)
          setPoints(aggregated)
          setExerciseBurnPoints([])
        } else if (metric === 'exercise_burn') {
          const response = await apiFetch(
            `${getApiBaseUrl()}/api/history/exercise-burn?start_date=${dateRange.start_date}&end_date=${dateRange.end_date}`,
            {
              headers: { Authorization: `Bearer ${token}` },
              signal: controller.signal,
            },
          )
          if (!response.ok) {
            const detail = await readErrorDetail(
              response,
              `Failed to load exercise burn history (HTTP ${response.status}).`,
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
            !('records' in payload) ||
            !Array.isArray((payload as { records: unknown[] }).records)
          ) {
            setExerciseBurnPoints([])
            setPoints([])
            return
          }
          const records = (payload as { records: ExerciseBurnDayRecord[] }).records
          const burnFiltered: ExerciseBurnPoint[] = []
          for (const record of records) {
            const dateISO = String(record.date)
            const s = Number(record.burned_strength ?? 0)
            const c = Number(record.burned_cardio ?? 0)
            burnFiltered.push({
              label: formatXAxisLabel(dateISO, locale),
              dateISO,
              strength: s,
              cardio: c,
              value: s + c,
            })
          }
          burnFiltered.sort((a, b) => (a.dateISO < b.dateISO ? -1 : 1))
          const burnAggregated = groupExerciseBurnPoints(burnFiltered, aggregationMode, locale)
          setExerciseBurnPoints(burnAggregated)
          setPoints([])
        } else {
          const response = await apiFetch(
            `${getApiBaseUrl()}/api/history/advanced?start_date=${dateRange.start_date}&end_date=${dateRange.end_date}`,
            {
              headers: { Authorization: `Bearer ${token}` },
              signal: controller.signal,
            },
          )
          if (!response.ok) {
            const detail = await readErrorDetail(
              response,
              `Failed to load nutrition history (HTTP ${response.status}).`,
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
            !('records' in payload) ||
            !Array.isArray((payload as { records: unknown[] }).records)
          ) {
            setPoints([])
            setExerciseBurnPoints([])
            return
          }
          const records = (payload as { records: NutritionRecord[] }).records
          for (const record of records) {
            const dateISO = String(record.date)
            const value =
              metric === 'net_calories'
                ? Number(record.consumed_calories ?? 0) - Number(record.burned_calories ?? 0)
                : metric === 'protein'
                  ? Number(record.consumed_protein ?? 0)
                  : metric === 'carbs'
                    ? Number(record.consumed_carbs ?? 0)
                    : Number(record.consumed_fat ?? 0)
            filtered.push({
              label: formatXAxisLabel(dateISO, locale),
              value,
              dateISO,
            })
          }
          filtered.sort((a, b) => (a.dateISO < b.dateISO ? -1 : 1))
          const aggregated = groupPoints(filtered, aggregationMode, metric, locale)
          setPoints(aggregated)
          setExerciseBurnPoints([])
        }
      } catch (error) {
        // Avoid displaying an AbortController cleanup as an actual error.
        if (controller.signal.aborted) {
          return
        }
        if (error instanceof DOMException && error.name === 'AbortError') {
          return
        }

        const detail = error instanceof Error ? error.message : 'Unexpected error.'
        setErrorMessage(detail)
        if (isAuthFailureMessage(detail)) {
          logout()
        }
      } finally {
        setIsLoading(false)
        if (!controller.signal.aborted) {
          setHasLoaded(true)
        }
      }
    }

    void load()
    return () => controller.abort()
  }, [
    isAuthenticated,
    token,
    metric,
    dateRange.start_date,
    dateRange.end_date,
    aggregationMode,
    logout,
    preferences.units,
    locale,
  ])

  const emptyState = useMemo(
    () =>
      (metric === 'exercise_burn' ? exerciseBurnPoints.length === 0 : points.length === 0) &&
      hasLoaded &&
      !isLoading,
    [metric, exerciseBurnPoints.length, points.length, hasLoaded, isLoading],
  )

  const xTickMargin = useMemo(() => {
    if (aggregationMode === 'daily') return 10
    if (aggregationMode === 'weekly') return 12
    return 18
  }, [aggregationMode])

  const yAxisFormatter = (value: number): string => {
    if (metric === 'weight') {
      const unitKey = preferences.units === 'imperial' ? 'lbs' : 'kg'
      return `${formatNumber(value, locale, 0)} ${t(`units.${unitKey}`)}`
    }
    if (metric === 'net_calories' || metric === 'exercise_burn') {
      return `${formatNumber(value, locale, 0)} ${t('units.kcal')}`
    }
    return `${formatNumber(value, locale, 0)} ${t('units.g')}`
  }

  return (
    <section className="space-y-6">
      <header className="flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <h1 className="text-xl font-bold text-[var(--color-primary)] sm:text-2xl">{t('history.title')}</h1>
          <p className="mt-1 text-sm font-semibold text-[var(--color-secondary)]">{t('history.subtitle')}</p>
        </div>
        <div className="text-right">
          <p className="text-sm font-semibold text-[var(--color-primary)]">{chartTitle}</p>
          <p className="mt-1 text-xs font-semibold text-[var(--color-secondary)]">{t('history.updatedAutomatically')}</p>
        </div>
      </header>

      <div className="rounded-2xl border border-[var(--color-border)] bg-[var(--color-surface)] p-4 shadow-sm sm:p-6">
        <div className="grid gap-4 lg:grid-cols-3">
          <div className="space-y-2">
            <span className="block text-sm font-semibold text-[var(--color-primary)]">{t('history.metric')}</span>
            <select
              value={metric}
              onChange={(event) => setMetric(event.target.value as MetricKey)}
              className="w-full rounded-xl border border-slate-300 bg-white px-4 py-3 text-sm outline-none transition focus:border-[var(--color-secondary)] focus:ring-2 focus:ring-[var(--color-focus-ring)]"
            >
              <option value="weight">{metricLabel('weight')}</option>
              <option value="net_calories">{metricLabel('net_calories')}</option>
              <option value="protein">{metricLabel('protein')}</option>
              <option value="carbs">{metricLabel('carbs')}</option>
              <option value="fat">{metricLabel('fat')}</option>
              <option value="exercise_burn">{metricLabel('exercise_burn')}</option>
            </select>
          </div>

          <div className="space-y-2">
            <span className="block text-sm font-semibold text-[var(--color-primary)]">{t('history.timeframe')}</span>
            <div className="flex flex-wrap gap-2">
              <button
                type="button"
                onClick={() => setTimeframeMode('last_7')}
                className={`rounded-xl border px-3 py-2 text-sm font-semibold transition ${timeframeMode === 'last_7'
                  ? 'border-[var(--color-active-border)] bg-[var(--color-active-bg)] text-[var(--color-secondary)]'
                  : 'border-slate-200 bg-white text-[var(--color-secondary)] hover:border-slate-300 hover:bg-[var(--color-surface-soft)]'
                  }`}
              >
                {t('history.last7')}
              </button>
              <button
                type="button"
                onClick={() => setTimeframeMode('last_30')}
                className={`rounded-xl border px-3 py-2 text-sm font-semibold transition ${timeframeMode === 'last_30'
                  ? 'border-[var(--color-active-border)] bg-[var(--color-active-bg)] text-[var(--color-secondary)]'
                  : 'border-slate-200 bg-white text-[var(--color-secondary)] hover:border-slate-300 hover:bg-[var(--color-surface-soft)]'
                  }`}
              >
                {t('history.last30')}
              </button>
              <button
                type="button"
                onClick={() => setTimeframeMode('last_90')}
                className={`rounded-xl border px-3 py-2 text-sm font-semibold transition ${timeframeMode === 'last_90'
                  ? 'border-[var(--color-active-border)] bg-[var(--color-active-bg)] text-[var(--color-secondary)]'
                  : 'border-slate-200 bg-white text-[var(--color-secondary)] hover:border-slate-300 hover:bg-[var(--color-surface-soft)]'
                  }`}
              >
                {t('history.last90')}
              </button>
              {metric === 'weight' ? (
                <button
                  type="button"
                  onClick={() => setTimeframeMode('last_1y')}
                  className={`rounded-xl border px-3 py-2 text-sm font-semibold transition ${timeframeMode === 'last_1y'
                    ? 'border-[var(--color-active-border)] bg-[var(--color-active-bg)] text-[var(--color-secondary)]'
                    : 'border-slate-200 bg-white text-[var(--color-secondary)] hover:border-slate-300 hover:bg-[var(--color-surface-soft)]'
                    }`}
                >
                  {t('history.last1y')}
                </button>
              ) : null}
              <button
                type="button"
                onClick={() => setTimeframeMode('custom_month')}
                className={`rounded-xl border px-3 py-2 text-sm font-semibold transition ${timeframeMode === 'custom_month'
                  ? 'border-[var(--color-active-border)] bg-[var(--color-active-bg)] text-[var(--color-secondary)]'
                  : 'border-slate-200 bg-white text-[var(--color-secondary)] hover:border-slate-300 hover:bg-[var(--color-surface-soft)]'
                  }`}
              >
                {t('history.custom')}
              </button>
            </div>

            {timeframeMode === 'custom_month' ? (
              <div className="mt-3">
                <label className="block text-xs font-semibold text-[var(--color-primary)]">
                  {t('history.pickMonth')}
                </label>
                <select
                  value={monthChoice}
                  onChange={(event) => setMonthChoice(event.target.value)}
                  className="mt-2 w-full rounded-xl border border-slate-300 bg-white px-4 py-3 text-sm outline-none transition focus:border-[var(--color-secondary)] focus:ring-2 focus:ring-[var(--color-focus-ring)]"
                >
                  {(() => {
                    const fallbackStart = (() => {
                      const end = parseDateISO(todayISO)
                      const start = addDays(end, -365)
                      return formatDateISO(start)
                    })()
                    const startISO = earliestDateISO || fallbackStart
                    const monthOptions = buildMonthOptions(startISO, todayISO)
                    return monthOptions.map((monthISO) => (
                      <option key={monthISO} value={monthISO}>
                        {formatMonthLabel(monthISO, locale)}
                      </option>
                    ))
                  })()}
                </select>
              </div>
            ) : null}
          </div>

          <div className="space-y-2">
            <span className="block text-sm font-semibold text-[var(--color-primary)]">{t('history.aggregation')}</span>
            <div className="flex flex-wrap gap-2">
              {(['daily', 'weekly', 'monthly'] as AggregationMode[]).map((modeOption) => {
                const selected = aggregationMode === modeOption
                return (
                  <button
                    key={modeOption}
                    type="button"
                    onClick={() => setAggregationMode(modeOption)}
                    className={`rounded-xl border px-3 py-2 text-sm font-semibold transition ${selected
                      ? 'border-[var(--color-active-border)] bg-[var(--color-active-bg)] text-[var(--color-secondary)]'
                      : 'border-slate-200 bg-white text-[var(--color-secondary)] hover:border-slate-300 hover:bg-[var(--color-surface-soft)]'
                      }`}
                  >
                    {modeOption === 'daily'
                      ? t('history.daily')
                      : modeOption === 'weekly'
                        ? t('history.weekly')
                        : t('history.monthly')}
                  </button>
                )
              })}
            </div>
          </div>
        </div>

        <div className="mt-4 flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
          <div className="text-sm text-[var(--color-secondary)]">
            {t('history.dateRange', {
              start: dateRange.start_date,
              end: dateRange.end_date,
            })}
          </div>
        </div>
      </div>

      <div className="rounded-2xl border border-[var(--color-border)] bg-white p-4 shadow-sm sm:p-6">
        {isLoading ? (
          <div className="animate-pulse">
            <div className="h-6 w-1/2 rounded bg-slate-200" />
            <div className="mt-4 h-72 w-full rounded bg-slate-100" />
          </div>
        ) : errorMessage ? (
          <div className="rounded-2xl border border-[var(--color-active-border)] bg-[var(--color-active-bg)] p-6">
            <p className="text-sm font-semibold text-[var(--color-secondary)]">{t('history.analyticsLoadFailed')}</p>
            <p className="mt-2 text-sm font-semibold text-[var(--color-secondary)]">{errorMessage}</p>
          </div>
        ) : emptyState ? (
          <div className="rounded-2xl border border-dashed border-[var(--color-active-border)] bg-[var(--color-active-bg)] p-8 text-center">
            <p className="text-sm font-bold text-[var(--color-primary)]">{t('history.noDataForPeriod')}</p>
            <p className="mt-2 text-sm font-semibold text-[var(--color-secondary)]">{t('history.tryAnotherTimeframeOrMetric')}</p>
          </div>
        ) : (
          <div className="h-[360px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              {metric === 'weight' ? (
                <LineChart data={points} margin={{ top: 10, right: 14, left: 0, bottom: 0 }}>
                  <defs>
                    <linearGradient id="weightGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#10B981" stopOpacity={0.35} />
                      <stop offset="100%" stopColor="#10B981" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis
                    dataKey="label"
                    tickMargin={xTickMargin}
                    tick={{ fontSize: 12 }}
                    interval="preserveStartEnd"
                  />
                  <YAxis
                    tick={{ fontSize: 12 }}
                    tickFormatter={yAxisFormatter}
                    width={60}
                    axisLine={false}
                  />
                  <Tooltip
                    formatter={(value: unknown) => {
                      const numberValue = typeof value === 'number' ? value : Number(value)
                      const safeValue = Number.isFinite(numberValue) ? numberValue : 0
                      return tooltipText(metric, safeValue)
                    }}
                  />
                  <Area
                    type="monotone"
                    dataKey="value"
                    name={t('history.weight')}
                    stroke="none"
                    fill="url(#weightGradient)"
                    isAnimationActive={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="value"
                    stroke="#10B981"
                    strokeWidth={2}
                    dot={false}
                    isAnimationActive={false}
                  />
                </LineChart>
              ) : metric === 'exercise_burn' ? (
                <BarChart data={exerciseBurnPoints} margin={{ top: 10, right: 14, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis
                    dataKey="label"
                    tickMargin={xTickMargin}
                    tick={{ fontSize: 12 }}
                    interval="preserveStartEnd"
                  />
                  <YAxis
                    tick={{ fontSize: 12 }}
                    tickFormatter={yAxisFormatter}
                    width={70}
                    axisLine={false}
                  />
                  <Tooltip
                    formatter={(value: unknown, name: unknown) => {
                      const numberValue = typeof value === 'number' ? value : Number(value)
                      const safeValue = Number.isFinite(numberValue) ? numberValue : 0
                      const label =
                        name === 'strength'
                          ? t('history.exerciseStrengthSeries')
                          : name === 'cardio'
                            ? t('history.exerciseCardioSeries')
                            : String(name)
                      return [tooltipText('exercise_burn', safeValue), label]
                    }}
                  />
                  <Bar
                    dataKey="strength"
                    stackId="ex"
                    fill="var(--color-exercise-strength)"
                    name="strength"
                    radius={[0, 0, 0, 0]}
                    isAnimationActive={false}
                  />
                  <Bar
                    dataKey="cardio"
                    stackId="ex"
                    fill="var(--color-exercise-cardio)"
                    name="cardio"
                    radius={[4, 4, 0, 0]}
                    isAnimationActive={false}
                  />
                </BarChart>
              ) : (
                <BarChart data={points} margin={{ top: 10, right: 14, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis
                    dataKey="label"
                    tickMargin={xTickMargin}
                    tick={{ fontSize: 12 }}
                    interval="preserveStartEnd"
                  />
                  <YAxis
                    tick={{ fontSize: 12 }}
                    tickFormatter={yAxisFormatter}
                    width={70}
                    axisLine={false}
                  />
                  <Tooltip
                    formatter={(value: unknown) => {
                      const numberValue = typeof value === 'number' ? value : Number(value)
                      const safeValue = Number.isFinite(numberValue) ? numberValue : 0
                      return tooltipText(metric, safeValue)
                    }}
                  />
                  {metric === 'protein' ? (
                    <Bar dataKey="value" fill="var(--color-protein)" radius={[4, 4, 0, 0]} isAnimationActive={false} />
                  ) : metric === 'carbs' ? (
                    <Bar dataKey="value" fill="var(--color-carbs)" radius={[4, 4, 0, 0]} isAnimationActive={false} />
                  ) : metric === 'fat' ? (
                    <Bar dataKey="value" fill="var(--color-fat)" radius={[4, 4, 0, 0]} isAnimationActive={false} />
                  ) : (
                    <Bar dataKey="value" fill="var(--color-secondary)" radius={[4, 4, 0, 0]} isAnimationActive={false} />
                  )}
                </BarChart>
              )}
            </ResponsiveContainer>
          </div>
        )}
      </div>
      {/* Background menu tracking injection */}
      {renderMenuNotificationToast()}
    </section>
  )
}
