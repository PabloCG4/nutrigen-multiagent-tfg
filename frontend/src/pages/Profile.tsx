import { useEffect, useMemo, useState } from 'react'
import type { FormEvent } from 'react'
import { useNavigate } from 'react-router-dom'
import { getApiBaseUrl } from '../config/apiBaseUrl'
import { apiFetch } from '../api/apiFetch'
import { useTranslation } from 'react-i18next'
import { useAuth } from '../context/AuthContext'
import { usePreferences } from '../context/PreferencesContext'
import { convertHeightInputToCm, convertWeightInputToKg, formatHeight, formatWeight } from '../utils/units'
import { ALLERGY_PRESET_IDS } from '../constants/allergies'

interface ProfileResponse {
  name: string
  location: string
  age: number
  gender: string
  weight: number
  height: number
  activity_level: string
  allergies: string[]
  physical_goal: string
  weight_goal_rate: number
  meals_per_day: number
}

interface ProfileFormState {
  name: string
  location: string
  age: string
  gender: 'male' | 'female'
  weight: string
  height: string
  activity_level: string
  allergies: string[]
  physical_goal: string
  weight_goal_rate: string
  meals_per_day: string
}

const GENDER_OPTIONS: ReadonlyArray<{ value: 'male' | 'female'; labelEs: string }> = [
  { value: 'male', labelEs: 'Hombre' },
  { value: 'female', labelEs: 'Mujer' },
]

const ACTIVITY_OPTIONS: ReadonlyArray<{ value: string; labelEs: string; hintEs: string }> = [
  { value: 'sedentary', labelEs: 'Sedentario', hintEs: 'Poco o ningún ejercicio' },
  { value: 'lightly_active', labelEs: 'Ligero', hintEs: 'Ejercicio ligero 1–3 días/semana' },
  {
    value: 'moderately_active',
    labelEs: 'Moderado',
    hintEs: 'Ejercicio moderado 3–5 días/semana',
  },
  { value: 'very_active', labelEs: 'Activo', hintEs: 'Ejercicio intenso 6–7 días/semana' },
  {
    value: 'extremely_active',
    labelEs: 'Muy activo',
    hintEs: 'Trabajo físico o muy alta actividad',
  },
]

const PHYSICAL_GOAL_OPTIONS: ReadonlyArray<{ value: string; labelEs: string }> = [
  { value: 'weight_loss', labelEs: 'Perder peso' },
  { value: 'maintenance', labelEs: 'Mantener' },
  { value: 'muscle_gain', labelEs: 'Ganar músculo' },
]

function toggleAllergy(list: string[], id: string): string[] {
  if (list.includes(id)) {
    return list.filter((item) => item !== id)
  }
  return [...list, id]
}

function isAuthFailureMessage(message: string): boolean {
  const trimmed = message.trim()
  if (trimmed === 'Could not validate credentials.') {
    return true
  }
  const lower = trimmed.toLowerCase()
  return lower.includes('could not validate credentials') || trimmed.includes('HTTP 401')
}

export function Profile() {
  const navigate = useNavigate()
  const { isAuthenticated, token, logout } = useAuth()
  const preferences = usePreferences()
  const { t } = useTranslation()

  const [isLoading, setIsLoading] = useState<boolean>(true)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [successMessage, setSuccessMessage] = useState<string | null>(null)

  const [form, setForm] = useState<ProfileFormState>({
    name: '',
    location: '',
    age: '',
    gender: 'male',
    weight: '',
    height: '',
    activity_level: 'sedentary',
    allergies: [],
    physical_goal: 'maintenance',
    weight_goal_rate: '0',
    meals_per_day: '3',
  })

  const payloadToSave = useMemo(() => {
    const weightValue = Number.parseFloat(form.weight)
    const heightValue = Number.parseFloat(form.height)
    return {
      name: form.name.trim(),
      location: form.location.trim(),
      age: Number.parseInt(form.age, 10),
      gender: form.gender,
      weight: convertWeightInputToKg(Number.isFinite(weightValue) ? weightValue : 0, preferences.units),
      height: convertHeightInputToCm(Number.isFinite(heightValue) ? heightValue : 0, preferences.units),
      activity_level: form.activity_level,
      allergies: form.allergies,
      physical_goal: form.physical_goal,
      weight_goal_rate: Number.parseFloat(form.weight_goal_rate),
      meals_per_day: Number.parseInt(form.meals_per_day, 10),
    }
  }, [form, preferences.units])

  useEffect(() => {
    if (!isAuthenticated || !token) {
      navigate('/login')
      return
    }

    const controller = new AbortController()

    const loadProfile = async (): Promise<void> => {
      setIsLoading(true)
      setErrorMessage(null)
      try {
        const response = await apiFetch(`${getApiBaseUrl()}/api/profile`, {
          headers: { Authorization: `Bearer ${token}` },
          signal: controller.signal,
        })
        if (!response.ok) {
          let detail = `Failed to load profile (HTTP ${response.status}).`
          try {
            const payload: unknown = await response.json()
            if (
              typeof payload === 'object' &&
              payload !== null &&
              'detail' in payload &&
              typeof (payload as { detail: unknown }).detail === 'string'
            ) {
              detail = (payload as { detail: string }).detail
            }
          } catch {
            // Ignore
          }
          if (isAuthFailureMessage(detail)) {
            logout()
            return
          }
          throw new Error(detail)
        }

        const data: unknown = await response.json()
        if (
          typeof data !== 'object' ||
          data === null ||
          !('name' in data) ||
          !('location' in data)
        ) {
          throw new Error(t('errors.invalidProfileResponse', 'Invalid profile response.'))
        }

        const p = data as ProfileResponse
        const weightKg = Number(p.weight ?? 0)
        const heightCm = Number(p.height ?? 0)
        const displayedWeight = formatWeight(weightKg, preferences.units).value
        const displayedHeight = formatHeight(heightCm, preferences.units).value
        setForm({
          name: p.name ?? '',
          location: String(p.location ?? ''),
          age: String(p.age ?? ''),
          gender: (p.gender === 'female' ? 'female' : 'male') as 'male' | 'female',
          weight: Number.isFinite(displayedWeight) ? displayedWeight.toFixed(1) : '',
          height: Number.isFinite(displayedHeight) ? displayedHeight.toFixed(1) : '',
          activity_level: p.activity_level ?? 'sedentary',
          allergies: Array.isArray(p.allergies) ? p.allergies : [],
          physical_goal: p.physical_goal ?? 'maintenance',
          weight_goal_rate: String(p.weight_goal_rate ?? 0),
          meals_per_day: String(p.meals_per_day ?? 3),
        })
      } catch (error) {
        if (error instanceof DOMException && error.name === 'AbortError') {
          return
        }
        const detail = error instanceof Error ? error.message : t('common.error')
        setErrorMessage(detail)
      } finally {
        setIsLoading(false)
      }
    }

    void loadProfile()
    return () => controller.abort()
  }, [isAuthenticated, token, logout, navigate, preferences.units])

  const handleSubmit = async (event: FormEvent<HTMLFormElement>): Promise<void> => {
    event.preventDefault()
    if (!token) {
      return
    }

    setErrorMessage(null)
    setSuccessMessage(null)

    try {
      const response = await apiFetch(`${getApiBaseUrl()}/api/profile`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify(payloadToSave),
      })

      if (!response.ok) {
        let detail = `Failed to save profile (HTTP ${response.status}).`
        try {
          const payload: unknown = await response.json()
          if (
            typeof payload === 'object' &&
            payload !== null &&
            'detail' in payload &&
            typeof (payload as { detail: unknown }).detail === 'string'
          ) {
            detail = (payload as { detail: string }).detail
          }
        } catch {
          // Ignore
        }
        if (isAuthFailureMessage(detail)) {
          logout()
          return
        }
        throw new Error(detail)
      }

      setSuccessMessage(t('profile.saveSuccess'))
    } catch (error) {
      const detail = error instanceof Error ? error.message : t('common.error')
      setErrorMessage(detail)
    }
  }

  if (!isAuthenticated) {
    return null
  }

  if (isLoading) {
    return (
      <section className="rounded-2xl border border-[var(--color-border)] bg-[var(--color-surface)] p-6 shadow-sm">
        <h1 className="text-xl font-bold text-[var(--color-primary)] sm:text-2xl">{t('profile.title')}</h1>
        <p className="mt-3 text-sm font-semibold text-[var(--color-secondary)]">{t('profile.loading')}</p>
      </section>
    )
  }

  return (
    <section className="space-y-6">
      <header className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold text-[var(--color-primary)] sm:text-2xl">{t('profile.title')}</h1>
          <p className="mt-1 text-sm font-semibold text-[var(--color-secondary)]">
            {t('profile.subtitle')}
          </p>
        </div>
      </header>

      <form onSubmit={handleSubmit} className="space-y-6 rounded-2xl border border-[var(--color-border)] bg-white p-6 shadow-sm">
        {errorMessage ? (
          <p className="rounded-lg border border-[var(--color-active-border)] bg-[var(--color-active-bg)] px-3 py-2 text-sm font-semibold text-[var(--color-secondary)]" role="alert">
            {errorMessage}
          </p>
        ) : null}
        {successMessage ? (
          <p className="rounded-lg border border-[var(--color-active-border)] bg-[var(--color-active-bg)] px-3 py-2 text-sm font-semibold text-[var(--color-secondary)]">
            {successMessage}
          </p>
        ) : null}

        <div className="grid gap-4 sm:grid-cols-2">
          <div className="space-y-2">
            <label className="block text-sm font-medium text-[var(--color-primary)]" htmlFor="profile-name">
              {t('profile.nameLabel')}
            </label>
            <input
              id="profile-name"
              value={form.name}
              onChange={(event) => setForm((prev) => ({ ...prev, name: event.target.value }))}
              className="w-full rounded-xl border border-slate-300 px-4 py-3 text-sm outline-none transition focus:border-[var(--color-secondary)] focus:ring-2 focus:ring-[var(--color-focus-ring)]"
              placeholder={t('onboarding.namePlaceholder')}
            />
          </div>

          <div className="space-y-2">
            <label className="block text-sm font-medium text-[var(--color-primary)]" htmlFor="profile-location">
              {t('profile.locationLabel')}
            </label>
            <input
              id="profile-location"
              value={form.location}
              onChange={(event) => setForm((prev) => ({ ...prev, location: event.target.value }))}
              className="w-full rounded-xl border border-slate-300 px-4 py-3 text-sm outline-none transition focus:border-[var(--color-secondary)] focus:ring-2 focus:ring-[var(--color-focus-ring)]"
              placeholder={t('onboarding.locationPlaceholder')}
            />
          </div>
        </div>

        <div className="grid gap-4 sm:grid-cols-2">
          <div className="space-y-2">
            <label className="block text-sm font-medium text-[var(--color-primary)]" htmlFor="profile-age">
              {t('profile.ageLabel')}
            </label>
            <input
              id="profile-age"
              type="number"
              min={0}
              max={130}
              value={form.age}
              onChange={(event) => setForm((prev) => ({ ...prev, age: event.target.value }))}
              className="w-full rounded-xl border border-slate-300 px-4 py-3 text-sm outline-none transition focus:border-[var(--color-secondary)] focus:ring-2 focus:ring-[var(--color-focus-ring)]"
            />
          </div>

          <div className="space-y-2">
            <span className="block text-sm font-medium text-[var(--color-primary)]">{t('profile.genderLabel')}</span>
            <div className="grid grid-cols-2 gap-3">
              {GENDER_OPTIONS.map((option) => (
                <button
                  key={option.value}
                  type="button"
                  onClick={() => setForm((prev) => ({ ...prev, gender: option.value }))}
                  className={`rounded-2xl border px-4 py-3 text-center text-sm font-semibold transition ${form.gender === option.value
                      ? 'border-[var(--color-active-border)] bg-[var(--color-active-bg)] text-[var(--color-secondary)]'
                      : 'border-slate-200 bg-white text-[var(--color-secondary)] hover:border-slate-300 hover:bg-[var(--color-surface-soft)]'
                    }`}
                >
                  {option.labelEs}
                </button>
              ))}
            </div>
          </div>
        </div>

        <div className="grid gap-4 sm:grid-cols-2">
          <div className="space-y-2">
            <label className="block text-sm font-medium text-[var(--color-primary)]" htmlFor="profile-weight">
              {`${t('history.weight')} (${t(preferences.units === 'imperial' ? 'units.lbs' : 'units.kg')})`}
            </label>
            <input
              id="profile-weight"
              type="number"
              step="0.1"
              min={0}
              value={form.weight}
              onChange={(event) => setForm((prev) => ({ ...prev, weight: event.target.value }))}
              className="w-full rounded-xl border border-slate-300 px-4 py-3 text-sm outline-none transition focus:border-[var(--color-secondary)] focus:ring-2 focus:ring-[var(--color-focus-ring)]"
            />
          </div>

          <div className="space-y-2">
            <label className="block text-sm font-medium text-[var(--color-primary)]" htmlFor="profile-height">
              {`${t('profile.heightLabel', { defaultValue: 'Height' })} (${t(preferences.units === 'imperial' ? 'units.in' : 'units.cm')})`}
            </label>
            <input
              id="profile-height"
              type="number"
              step="0.1"
              min={0}
              value={form.height}
              onChange={(event) => setForm((prev) => ({ ...prev, height: event.target.value }))}
              className="w-full rounded-xl border border-slate-300 px-4 py-3 text-sm outline-none transition focus:border-[var(--color-secondary)] focus:ring-2 focus:ring-[var(--color-focus-ring)]"
            />
          </div>
        </div>

        <div className="space-y-3">
          <span className="block text-sm font-semibold text-[var(--color-primary)]">{t('profile.activityLevelLabel')}</span>
          <div className="grid gap-3 md:grid-cols-2">
            {ACTIVITY_OPTIONS.map((option) => {
              const selected = form.activity_level === option.value
              return (
                <button
                  key={option.value}
                  type="button"
                  onClick={() => setForm((prev) => ({ ...prev, activity_level: option.value }))}
                  className={`rounded-2xl border px-4 py-3 text-left transition ${selected
                      ? 'border-[var(--color-active-border)] bg-[var(--color-active-bg)]'
                      : 'border-slate-200 bg-white hover:border-slate-300 hover:bg-[var(--color-surface-soft)]'
                    }`}
                >
                  <span className="block text-sm font-semibold text-[var(--color-primary)]">{option.labelEs}</span>
                  <span className="block text-xs font-semibold text-[var(--color-secondary)]">{option.hintEs}</span>
                </button>
              )
            })}
          </div>
        </div>

        <div className="space-y-3">
          <span className="block text-sm font-semibold text-[var(--color-primary)]">{t('profile.allergiesLabel')}</span>
          <p className="text-xs font-semibold text-[var(--color-secondary)]">{t('profile.allergiesBody')}</p>
          <div className="flex flex-wrap gap-2">
            {ALLERGY_PRESET_IDS.map((id) => {
              const selected = form.allergies.includes(id)
              return (
                <button
                  key={id}
                  type="button"
                  onClick={() =>
                    setForm((prev) => ({
                      ...prev,
                      allergies: toggleAllergy(prev.allergies, id),
                    }))
                  }
                  className={`rounded-full border px-3 py-1.5 text-sm font-medium transition ${selected
                      ? 'border-[var(--color-active-border)] bg-[var(--color-active-bg)] text-[var(--color-secondary)]'
                      : 'border-slate-200 bg-white text-[var(--color-secondary)] hover:border-slate-300 hover:bg-[var(--color-surface-soft)]'
                    }`}
                >
                  {t(`common.allergens.${id}`)}
                </button>
              )
            })}
          </div>
        </div>

        <div className="space-y-3">
          <span className="block text-sm font-semibold text-[var(--color-primary)]">{t('profile.physicalGoalLabel')}</span>
          <div className="grid gap-3 sm:grid-cols-3">
            {PHYSICAL_GOAL_OPTIONS.map((option) => {
              const selected = form.physical_goal === option.value
              return (
                <button
                  key={option.value}
                  type="button"
                  onClick={() => setForm((prev) => ({ ...prev, physical_goal: option.value }))}
                  className={`rounded-2xl border px-4 py-3 text-center transition ${selected
                      ? 'border-[var(--color-active-border)] bg-[var(--color-active-bg)] text-[var(--color-secondary)]'
                      : 'border-slate-200 bg-white text-[var(--color-secondary)] hover:border-slate-300 hover:bg-[var(--color-surface-soft)]'
                    }`}
                >
                  {option.labelEs}
                </button>
              )
            })}
          </div>

          <div className="mt-4 grid gap-4 sm:grid-cols-2">
            <div className="space-y-2">
              <label className="block text-sm font-medium text-[var(--color-primary)]" htmlFor="profile-weight-rate">
                {t('profile.weightGoalRateLabel')}
              </label>
              <input
                id="profile-weight-rate"
                type="number"
                min={-1.5}
                max={1.5}
                step={0.05}
                value={form.weight_goal_rate}
                onChange={(event) =>
                  setForm((prev) => ({ ...prev, weight_goal_rate: event.target.value }))
                }
                className="w-full rounded-xl border border-slate-300 px-4 py-3 text-sm outline-none transition focus:border-[var(--color-secondary)] focus:ring-2 focus:ring-[var(--color-focus-ring)]"
                placeholder={t('common.zero')}
              />
            </div>

            <div className="space-y-2">
              <label className="block text-sm font-medium text-[var(--color-primary)]" htmlFor="profile-meals-per-day">
                {t('profile.mealsPerDayLabel', 'Comidas al día')}
              </label>
              <select
                id="profile-meals-per-day"
                value={form.meals_per_day}
                onChange={(event) =>
                  setForm((prev) => ({ ...prev, meals_per_day: event.target.value }))
                }
                className="w-full rounded-xl border border-slate-300 px-4 py-3 text-sm bg-white outline-none transition focus:border-[var(--color-secondary)] focus:ring-2 focus:ring-[var(--color-focus-ring)]"
              >
                {[1, 2, 3, 4, 5, 6].map((num) => (
                  <option key={num} value={String(num)}>
                    {num}
                  </option>
                ))}
              </select>
            </div>
          </div>
        </div>

        <div className="pt-2">
          <button
            type="submit"
            className="w-full rounded-xl bg-gradient-to-r from-emerald-600 to-sky-600 px-4 py-3 text-sm font-semibold uppercase tracking-wide text-white shadow-sm transition hover:from-emerald-700 hover:to-sky-700"
          >
            {t('profile.saveChanges')}
          </button>
        </div>
      </form>
    </section>
  )
}
