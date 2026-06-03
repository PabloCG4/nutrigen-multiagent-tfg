import { useCallback, useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import type { TFunction } from 'i18next'
import { getApiBaseUrl } from '../config/apiBaseUrl'
import { apiFetch } from '../api/apiFetch'
import { DeleteButton } from '../components/DeleteButton'
import { useAuth } from '../context/AuthContext'
import { useMenuJobNotification } from '../hooks/useMenuJobNotification'

type ExerciseCategory = 'strength' | 'cardio'
type StrengthIntensity = 'medium' | 'high' | 'very_high'
type CardioType = 'bike' | 'walk' | 'run' | 'swim'

interface ExercisePayload {
  date: string
  category: ExerciseCategory
  duration_minutes: number
  type?: CardioType
  intensity?: StrengthIntensity
  manual_burned_calories?: number
}

// This function formats the current date in the ISO 8601 format (YYYY-MM-DD)
function formatTodayISO(): string {
  const d = new Date()
  const year = d.getFullYear()
  const month = String(d.getMonth() + 1).padStart(2, '0')
  const day = String(d.getDate()).padStart(2, '0')
  return `${year}-${month}-${day}`
}

function isAuthFailureMessage(message: string): boolean {
  const trimmed = message.trim()
  if (trimmed === 'Could not validate credentials.') {
    return true
  }
  const lower = trimmed.toLowerCase()
  return lower.includes('could not validate credentials') || trimmed.includes('HTTP 401')
}

interface ExerciseLogRow {
  id: number
  category: ExerciseCategory
  duration_minutes: number
  cardio_type: string | null
  strength_intensity: string | null
  burned_calories: number
  manual_entry: boolean
}

function strengthIntensityLabel(t: TFunction, value: string): string {
  if (value === 'medium') return t('common.medium')
  if (value === 'high') return t('common.high')
  if (value === 'very_high') return t('common.veryHigh')
  return value
}

function cardioTypeLabel(t: TFunction, value: string): string {
  if (value === 'bike') return t('common.bike')
  if (value === 'walk') return t('common.walk')
  if (value === 'run') return t('common.run')
  if (value === 'swim') return t('common.swim')
  return value
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
    // Ignore parsing errors.
  }
  return fallback
}

export function Exercise() {
  const navigate = useNavigate()
  const { isAuthenticated, token, logout } = useAuth()
  const { renderMenuNotificationToast } = useMenuJobNotification()
  const { t } = useTranslation()

  // Initial values 
  const [category, setCategory] = useState<ExerciseCategory>('strength')
  const [dateISO, setDateISO] = useState<string>(formatTodayISO())

  const [durationMinutes, setDurationMinutes] = useState<string>('30')
  const [strengthIntensity, setStrengthIntensity] = useState<StrengthIntensity>('medium')

  const [cardioType, setCardioType] = useState<CardioType>('walk')
  const [manualCaloriesInput, setManualCaloriesInput] = useState<string>('')

  const [isSubmitting, setIsSubmitting] = useState<boolean>(false)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [successMessage, setSuccessMessage] = useState<string | null>(null)

  const [todayLogs, setTodayLogs] = useState<ExerciseLogRow[]>([])
  const [logsLoading, setLogsLoading] = useState<boolean>(false)
  const [logsError, setLogsError] = useState<string | null>(null)
  const [deleteBusyId, setDeleteBusyId] = useState<number | null>(null)

  // List of sessions logged for **today** (calendar day in local time).
  const loadTodaysExerciseLogs = useCallback(async (): Promise<void> => {
    if (!token) return
    setLogsLoading(true)
    setLogsError(null)
    try {
      const day = formatTodayISO()
      const response = await apiFetch(`${getApiBaseUrl()}/api/exercise?date=${encodeURIComponent(day)}`, {
        headers: { Authorization: `Bearer ${token}` },
      })
      if (!response.ok) {
        const detail = await readErrorDetail(response, `Failed to load exercises (HTTP ${response.status}).`)
        if (isAuthFailureMessage(detail)) {
          logout()
          navigate('/login', { replace: true })
          return
        }
        throw new Error(detail)
      }
      const payload: unknown = await response.json()
      if (
        typeof payload !== 'object' ||
        payload === null ||
        !('logs' in payload) ||
        !Array.isArray((payload as { logs: unknown }).logs)
      ) {
        setTodayLogs([])
        return
      }
      const raw = (payload as { logs: unknown[] }).logs
      const parsed: ExerciseLogRow[] = []
      for (const item of raw) {
        if (typeof item !== 'object' || item === null) continue
        const row = item as Record<string, unknown>
        const id = Number(row.id)
        const cat = row.category === 'cardio' ? 'cardio' : 'strength'
        const dur = Number(row.duration_minutes)
        const kcal = Number(row.burned_calories)
        if (!Number.isFinite(id) || !Number.isFinite(dur) || !Number.isFinite(kcal)) continue
        parsed.push({
          id,
          category: cat,
          duration_minutes: dur,
          cardio_type: typeof row.cardio_type === 'string' ? row.cardio_type : null,
          strength_intensity: typeof row.strength_intensity === 'string' ? row.strength_intensity : null,
          burned_calories: kcal,
          manual_entry: Boolean(row.manual_entry),
        })
      }
      setTodayLogs(parsed)
    } catch (error) {
      const detail = error instanceof Error ? error.message : t('common.error')
      setLogsError(detail)
    } finally {
      setLogsLoading(false)
    }
  }, [token, logout, navigate, t])

  useEffect(() => {
    if (!isAuthenticated) {
      navigate('/login', { replace: true })
    }
  }, [isAuthenticated, navigate])

  useEffect(() => {
    if (!isAuthenticated || !token) return
    void loadTodaysExerciseLogs()
  }, [isAuthenticated, token, loadTodaysExerciseLogs])

  const payload = useMemo<ExercisePayload | null>(() => {
    if (!durationMinutes) {
      return null
    }
    const duration = Number.parseInt(durationMinutes, 10)
    if (!Number.isFinite(duration) || duration <= 0) {
      return null
    }

    const manualTrimmed = manualCaloriesInput.trim()
    if (manualTrimmed !== '') {
      const manualVal = Number.parseFloat(manualTrimmed.replace(',', '.'))
      if (!Number.isFinite(manualVal) || manualVal < 1 || manualVal > 10000) {
        return null
      }
    }

    const base: ExercisePayload =
      category === 'strength'
        ? {
          date: dateISO,
          category,
          duration_minutes: duration,
          intensity: strengthIntensity,
        }
        : {
          date: dateISO,
          category,
          duration_minutes: duration,
          type: cardioType,
        }

    if (manualTrimmed === '') {
      return base
    }
    const manualVal = Number.parseFloat(manualTrimmed.replace(',', '.'))
    return { ...base, manual_burned_calories: manualVal }
  }, [cardioType, category, dateISO, durationMinutes, strengthIntensity, manualCaloriesInput])

  const handleSubmit = async (): Promise<void> => {
    setErrorMessage(null)
    setSuccessMessage(null)

    if (!token) {
      navigate('/login', { replace: true })
      return
    }
    if (!payload) {
      const duration = Number.parseInt(durationMinutes, 10)
      if (!Number.isFinite(duration) || duration <= 0) {
        setErrorMessage(t('exercisePage.validDuration'))
        return
      }
      const manualTrimmed = manualCaloriesInput.trim()
      if (manualTrimmed !== '') {
        setErrorMessage(t('exercisePage.validManualCalories'))
        return
      }
      setErrorMessage(t('exercisePage.validDuration'))
      return
    }

    setIsSubmitting(true)
    // Try to submit the exercise data to the API
    try {
      const response = await apiFetch(`${getApiBaseUrl()}/api/exercise`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify(payload),
      })

      if (!response.ok) {
        const detail = await readErrorDetail(
          response,
          `${t('common.error')} (HTTP ${response.status}).`
        )
        if (isAuthFailureMessage(detail)) {
          logout()
          navigate('/', { replace: true })
          return
        }
        throw new Error(detail)
      }

      // If the submission is successful, show the success message and reset the form
      setSuccessMessage(t('exercisePage.success'))
      setDurationMinutes('30')
      setStrengthIntensity('medium')
      setCardioType('walk')
      setManualCaloriesInput('')
      setDateISO(formatTodayISO())
      await loadTodaysExerciseLogs()
    } catch (error) {
      const detail = error instanceof Error ? error.message : t('common.error')
      setErrorMessage(detail)
    } finally {
      setIsSubmitting(false)
    }
  }

  const handleDeleteLog = async (logId: number): Promise<void> => {
    if (!token) return
    setDeleteBusyId(logId)
    setLogsError(null)
    try {
      const response = await apiFetch(`${getApiBaseUrl()}/api/exercise/${logId}`, {
        method: 'DELETE',
        headers: { Authorization: `Bearer ${token}` },
      })
      if (!response.ok) {
        const detail = await readErrorDetail(response, `Failed to delete exercise (HTTP ${response.status}).`)
        if (isAuthFailureMessage(detail)) {
          logout()
          navigate('/login', { replace: true })
          return
        }
        throw new Error(detail)
      }
      await loadTodaysExerciseLogs()
    } catch (error) {
      const detail = error instanceof Error ? error.message : t('common.error')
      setLogsError(detail)
    } finally {
      setDeleteBusyId(null)
    }
  }

  return (
    <section className="space-y-6">
      <header>
        <h1 className="text-xl font-bold text-[var(--color-primary)] sm:text-2xl">{t('exercisePage.title')}</h1>
        <p className="mt-1 text-sm text-[var(--color-muted)]">
          {t('exercisePage.subtitle')}
        </p>
      </header>

      <div className="grid gap-4 lg:grid-cols-2">
        <button
          type="button"
          onClick={() => setCategory('strength')}
          className={`flex flex-col items-start rounded-2xl border p-5 text-left transition ${category === 'strength'
            ? 'border-[var(--color-active-border)] bg-[var(--color-active-bg)] hover:bg-[var(--color-active-bg)]'
            : 'border-[var(--color-border)] bg-white hover:border-slate-300 hover:bg-[var(--color-surface-soft)]'
            }`}
          aria-pressed={category === 'strength'}
        >
          <span className="mt-2 text-lg font-bold text-[var(--color-primary)]">{t('exercisePage.strengthTitle')}</span>
          <span className="mt-2 text-sm text-[var(--color-secondary)]">
            {t('exercisePage.strengthBody')}
          </span>
        </button>

        <button
          type="button"
          onClick={() => setCategory('cardio')}
          className={`flex flex-col items-start rounded-2xl border p-5 text-left transition ${category === 'cardio'
            ? 'border-[var(--color-active-border)] bg-[var(--color-active-bg)] hover:bg-[var(--color-active-bg)]'
            : 'border-[var(--color-border)] bg-white hover:border-slate-300 hover:bg-[var(--color-surface-soft)]'
            }`}
          aria-pressed={category === 'cardio'}
        >
          <span className="mt-2 text-lg font-bold text-[var(--color-primary)]">{t('exercisePage.cardioTitle')}</span>
          <span className="mt-2 text-sm text-[var(--color-secondary)]">
            {t('exercisePage.cardioBody')}
          </span>
        </button>
      </div>

      {errorMessage ? (
        <div
          className="rounded-2xl border border-[var(--color-active-border)] bg-[var(--color-active-bg)] p-4 text-sm font-bold text-[var(--color-secondary)]"
          role="alert"
        >
          {errorMessage}
        </div>
      ) : null}
      {successMessage ? (
        <div
          className="rounded-2xl border border-[var(--color-active-border)] bg-[var(--color-active-bg)] p-4 text-sm font-bold text-[var(--color-secondary)]"
          role="status"
        >
          {successMessage}
        </div>
      ) : null}

      <form
        className="space-y-6 rounded-2xl border border-[var(--color-border)] bg-white p-6 shadow-sm"
        noValidate
        onSubmit={(event) => {
          event.preventDefault()
          void handleSubmit()
        }}
      >
        <div className="grid gap-4 sm:grid-cols-2">
          <div className="space-y-2">
            <label className="block text-sm font-medium text-[var(--color-primary)]" htmlFor="exercise-date">
              {t('exercisePage.date')}
            </label>
            <input
              id="exercise-date"
              type="date"
              value={dateISO}
              onChange={(event) => setDateISO(event.target.value)}
              className="exercise-date-input w-full rounded-xl border border-slate-300 bg-white px-4 py-3 text-sm text-black outline-none transition focus:border-[var(--color-border-strong)] focus:ring-2 focus:ring-[var(--color-focus-ring)]"
            />
          </div>

          <div className="space-y-2">
            <label className="block text-sm font-medium text-[var(--color-primary)]" htmlFor="exercise-duration">
              {t('exercisePage.durationMinutes')}
            </label>
            <input
              id="exercise-duration"
              type="number"
              min={0}
              step={5}
              value={durationMinutes}
              onChange={(event) => setDurationMinutes(event.target.value)}
              className="w-full rounded-xl border border-slate-300 px-4 py-3 text-sm text-black outline-none transition focus:border-[var(--color-border-strong)] focus:ring-2 focus:ring-[var(--color-focus-ring)]"
            />
          </div>
        </div>

        {category === 'strength' ? (
          <div className="space-y-3">
            <span className="block text-sm font-semibold text-[var(--color-primary)]">{t('exercisePage.intensity')}</span>
            <div className="grid gap-3 sm:grid-cols-3">
              {(
                [
                  { value: 'medium', label: t('common.medium', 'Medium') },
                  { value: 'high', label: t('common.high', 'High') },
                  { value: 'very_high', label: t('common.veryHigh', 'Very high') },
                ] as const
              ).map((option) => {
                const selected = strengthIntensity === option.value
                return (
                  <button
                    key={option.value}
                    type="button"
                    onClick={() => setStrengthIntensity(option.value)}
                    className={`rounded-2xl border px-4 py-3 text-sm font-normal transition ${selected
                      ? 'border-[var(--color-active-border)] bg-[var(--color-active-bg)] text-[var(--color-secondary)]'
                      : 'border-slate-200 bg-white text-[var(--color-secondary)] hover:border-slate-300 hover:bg-[var(--color-surface-soft)]'
                      }`}
                  >
                    {option.label}
                  </button>
                )
              })}
            </div>
            <div className="space-y-2">
              <label className="block text-sm font-medium text-[var(--color-primary)]" htmlFor="exercise-manual-kcal-strength">
                {t('exercisePage.manualCaloriesLabel')}
              </label>
              <input
                id="exercise-manual-kcal-strength"
                type="number"
                min={0}
                max={10000}
                step={25}
                value={manualCaloriesInput}
                onChange={(event) => setManualCaloriesInput(event.target.value)}
                placeholder={t('exercisePage.manualCaloriesPlaceholder')}
                className="w-full rounded-xl border border-slate-300 px-4 py-3 text-sm text-[var(--color-primary)] placeholder:text-[var(--color-border)] outline-none transition focus:border-[var(--color-border-strong)] focus:ring-2 focus:ring-[var(--color-focus-ring)]"
              />
              <p className="text-xs font-semibold text-[var(--color-secondary)]">{t('exercisePage.manualCaloriesHint')}</p>
            </div>
          </div>
        ) : (
          <div className="space-y-3">
            <span className="block text-sm font-semibold text-[var(--color-primary)]">{t('exercisePage.cardioType')}</span>
            <div className="grid gap-3 sm:grid-cols-4">
              {(
                [
                  { value: 'bike', label: t('common.bike', 'Bike') },
                  { value: 'walk', label: t('common.walk', 'Walk') },
                  { value: 'run', label: t('common.run', 'Run') },
                  { value: 'swim', label: t('common.swim', 'Swim') },
                ] as const
              ).map((option) => {
                const selected = cardioType === option.value
                return (
                  <button
                    key={option.value}
                    type="button"
                    onClick={() => setCardioType(option.value)}
                    className={`rounded-2xl border-2 px-4 py-3 text-sm font-normal transition ${selected
                      ? 'border-[var(--color-active-border)] bg-[var(--color-active-bg)] text-[var(--color-secondary)]'
                      : 'border-slate-200 bg-white text-[var(--color-secondary)] hover:border-slate-300 hover:bg-[var(--color-surface-soft)]'
                      }`}
                  >
                    {option.label}
                  </button>
                )
              })}
            </div>
            <div className="space-y-2">
              <label className="block text-sm font-medium text-[var(--color-primary)]" htmlFor="exercise-manual-kcal-cardio">
                {t('exercisePage.manualCaloriesLabel')}
              </label>
              <input
                id="exercise-manual-kcal-cardio"
                type="number"
                min={1}
                max={10000}
                step={1}
                value={manualCaloriesInput}
                onChange={(event) => setManualCaloriesInput(event.target.value)}
                placeholder={t('exercisePage.manualCaloriesPlaceholder')}
                className="w-full rounded-xl border border-slate-300 px-4 py-3 text-sm text-black outline-none transition focus:border-[var(--color-border-strong)] focus:ring-2 focus:ring-[var(--color-focus-ring)]"
              />
              <p className="text-xs font-semibold text-[var(--color-secondary)]">{t('exercisePage.manualCaloriesHint')}</p>
            </div>
          </div>
        )}

        <button
          type="submit"
          disabled={isSubmitting}
          className="w-full rounded-xl bg-gradient-to-r from-emerald-600 to-sky-600 px-4 py-3 text-sm font-semibold uppercase tracking-wide text-white shadow-sm transition hover:from-emerald-700 hover:to-sky-700 disabled:cursor-not-allowed disabled:opacity-60"
        >
          {isSubmitting ? t('exercisePage.saving') : t('exercisePage.logWorkout')}
        </button>
      </form>

      <div className="rounded-2xl border border-[var(--color-border)] bg-[var(--color-surface)] p-4 shadow-sm sm:p-6">
        <h2 className="text-base font-bold text-[var(--color-primary)]">{t('exercisePage.todaySectionTitle')}</h2>
        {logsLoading ? (
          <p className="mt-3 text-sm font-semibold text-[var(--color-secondary)]">{t('exercisePage.todayLoading')}</p>
        ) : null}
        {logsError ? (
          <p className="mt-3 text-sm font-semibold text-[var(--color-secondary)]" role="alert">
            {logsError}
          </p>
        ) : null}
        {!logsLoading && todayLogs.length === 0 && !logsError ? (
          <p className="mt-3 text-sm font-semibold text-[var(--color-secondary)]">{t('exercisePage.todayEmpty')}</p>
        ) : null}
        <ul className="mt-4 space-y-3">
          {todayLogs.map((log) => (
            <li
              key={log.id}
              className="flex flex-col gap-3 rounded-xl border border-[var(--color-border)] bg-white px-3 py-3 sm:flex-row sm:items-start sm:justify-between"
            >
              <div className="min-w-0 flex-1 space-y-1">
                <p className="text-sm font-bold text-[var(--color-primary)]">
                  {log.category === 'strength' ? t('exercisePage.strengthTitle') : t('exercisePage.cardioTitle')}
                </p>
                <div className="text-sm">
                  <span className="font-semibold text-[var(--color-primary)]">{t('exercisePage.fieldDuration')}</span>{' '}
                  <span className="font-semibold text-[var(--color-secondary)]">
                    {t('exercisePage.fieldDurationValue', { minutes: log.duration_minutes })}
                  </span>
                </div>
                {log.category === 'strength' && log.strength_intensity ? (
                  <div className="text-sm">
                    <span className="font-semibold text-[var(--color-primary)]">{t('exercisePage.intensity')}</span>{' '}
                    <span className="font-semibold text-[var(--color-secondary)]">
                      {strengthIntensityLabel(t, log.strength_intensity)}
                    </span>
                  </div>
                ) : null}
                {log.category === 'cardio' && log.cardio_type ? (
                  <div className="text-sm">
                    <span className="font-semibold text-[var(--color-primary)]">{t('exercisePage.cardioType')}</span>{' '}
                    <span className="font-semibold text-[var(--color-secondary)]">
                      {cardioTypeLabel(t, log.cardio_type)}
                    </span>
                  </div>
                ) : null}
                <div className="text-sm">
                  <span className="font-semibold text-[var(--color-primary)]">
                    {log.manual_entry ? t('exercisePage.fieldCalories') : t('exercisePage.fieldCaloriesEstimated')}
                  </span>{' '}
                  <span className="font-semibold text-[var(--color-secondary)]">
                    {t('exercisePage.fieldCaloriesValue', { kcal: log.burned_calories.toFixed(0) })}
                  </span>
                </div>
                {log.manual_entry ? (
                  <p className="text-xs font-semibold text-[var(--color-secondary)]">{t('exercisePage.caloriesManualBadge')}</p>
                ) : null}
              </div>
              <DeleteButton
                type="button"
                disabled={deleteBusyId === log.id}
                aria-label={t('exercisePage.deleteLogAria')}
                onClick={() => void handleDeleteLog(log.id)}
                className="shrink-0 self-start sm:self-center"
              >
                {deleteBusyId === log.id ? t('exercisePage.deleting') : t('exercisePage.deleteLog')}
              </DeleteButton>
            </li>
          ))}
        </ul>
      </div>
      {/* Background menu tracking injection */}
      {renderMenuNotificationToast()}
    </section>
  )
}
