import { useEffect, useMemo, useState, type Dispatch, type SetStateAction } from 'react'
import { useNavigate } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import { getApiBaseUrl } from '../config/apiBaseUrl'
import { apiFetch } from '../api/apiFetch'
import { useAuth } from '../context/AuthContext'
import { ALLERGY_PRESET_IDS } from '../constants/allergies'
const TOTAL_STEPS = 4

/** Payload shape must match backend `ProfileUpdate` in `src/backend/main.py`. */
export interface ProfilePayload {
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

/** Internal form state while collecting input (allows empty fields between steps). */
interface OnboardingFormState {
  name: string
  location: string
  age: string
  gender: 'male' | 'female' | ''
  weight: string
  height: string
  activity_level: string
  allergies: string[]
  physical_goal: string
  weight_goal_rate: string
  meals_per_day: string
}

const ACTIVITY_OPTIONS: ReadonlyArray<{
  value: string
  labelEs: string
  hintEs: string
}> = [
    {
      value: 'sedentary',
      labelEs: 'Sedentario',
      hintEs: 'Poco o ningún ejercicio',
    },
    {
      value: 'lightly_active',
      labelEs: 'Ligero',
      hintEs: 'Ejercicio ligero 1–3 días/semana',
    },
    {
      value: 'moderately_active',
      labelEs: 'Moderado',
      hintEs: 'Ejercicio moderado 3–5 días/semana',
    },
    {
      value: 'very_active',
      labelEs: 'Activo',
      hintEs: 'Ejercicio intenso 6–7 días/semana',
    },
    {
      value: 'extremely_active',
      labelEs: 'Muy activo',
      hintEs: 'Ejercicio muy intenso o trabajo físico',
    },
  ]

const PHYSICAL_GOAL_OPTIONS: ReadonlyArray<{
  value: string
  labelEs: string
  hintEs: string
}> = [
    {
      value: 'weight_loss',
      labelEs: 'Perder peso',
      hintEs: 'Déficit calórico controlado',
    },
    {
      value: 'maintenance',
      labelEs: 'Mantener',
      hintEs: 'Equilibrio y hábitos sostenibles',
    },
    {
      value: 'muscle_gain',
      labelEs: 'Ganar músculo',
      hintEs: 'Superávit moderado y proteína',
    },
  ]

const WEIGHT_GOAL_PRESETS: ReadonlyArray<{ value: number; labelEs: string }> = [
  { value: -1, labelEs: '-1.0 kg/sem' },
  { value: -0.5, labelEs: '-0.5 kg/sem' },
  { value: -0.25, labelEs: '-0.25 kg/sem' },
  { value: 0, labelEs: 'Mantener' },
  { value: 0.25, labelEs: '+0.25 kg/sem' },
  { value: 0.5, labelEs: '+0.5 kg/sem' },
  { value: 1, labelEs: '+1.0 kg/sem' },
]

function initialFormState(): OnboardingFormState {
  return {
    name: '',
    location: '',
    age: '',
    gender: '',
    weight: '',
    height: '',
    activity_level: '',
    allergies: [],
    physical_goal: '',
    weight_goal_rate: '',
    meals_per_day: '3',
  }
}

async function parseApiError(response: Response, fallback: string): Promise<string> {
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
    return fallback
  }
  return fallback
}

function toggleAllergy(list: string[], id: string): string[] {
  if (list.includes(id)) {
    return list.filter((item) => item !== id)
  }
  return [...list, id]
}

export function Onboarding() {
  const navigate = useNavigate()
  const { isAuthenticated, token } = useAuth()
  const { t } = useTranslation()
  const [step, setStep] = useState<number>(1)
  const [form, setForm] = useState<OnboardingFormState>(initialFormState)
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)

  useEffect(() => {
    if (!isAuthenticated || !token) {
      navigate('/login', { replace: true })
    }
  }, [isAuthenticated, token, navigate])

  const progressPercent = useMemo(
    () => Math.round((step / TOTAL_STEPS) * 100),
    [step],
  )

  const validateStep = (currentStep: number): string | null => {
    if (currentStep === 1) {
      const ageNum = Number.parseInt(form.age, 10)
      if (!form.name.trim()) {
        return 'Introduce tu nombre.'
      }
      if (!form.location.trim()) {
        return 'Indica tu ubicación.'
      }
      if (!Number.isFinite(ageNum) || ageNum < 10 || ageNum > 120) {
        return 'Introduce una edad válida (10–120).'
      }
      if (form.gender !== 'male' && form.gender !== 'female') {
        return 'Selecciona una opción de género.'
      }
    }
    if (currentStep === 2) {
      const weightKg = Number.parseFloat(form.weight)
      const heightCm = Number.parseFloat(form.height)

      if (!Number.isFinite(weightKg) || weightKg < 10 || weightKg > 400) {
        return t('onboarding.validationWeight')
      }
      if (!Number.isFinite(heightCm) || heightCm < 50 || heightCm > 250) {
        return t('onboarding.validationHeight')
      }
    }
    if (currentStep === 3) {
      if (!form.activity_level) {
        return 'Selecciona tu nivel de actividad.'
      }
    }
    if (currentStep === 4) {
      if (!form.physical_goal) {
        return 'Selecciona tu objetivo físico.'
      }
      const rate = Number.parseFloat(form.weight_goal_rate)
      if (!Number.isFinite(rate) || rate < -1.5 || rate > 1.5) {
        return 'El ritmo de peso debe estar entre -1.5 y +1.5 kg/semana.'
      }
      const mealsNum = Number.parseInt(form.meals_per_day, 10)
      if (!Number.isFinite(mealsNum) || mealsNum < 1 || mealsNum > 6) {
        return 'El número de comidas debe estar entre 1 y 6.'
      }
    }
    return null
  }

  const goNext = (): void => {
    setErrorMessage(null)
    const message = validateStep(step)
    if (message) {
      setErrorMessage(message)
      return
    }
    if (step < TOTAL_STEPS) {
      setStep((previous) => previous + 1)
    }
  }

  const goBack = (): void => {
    setErrorMessage(null)
    if (step > 1) {
      setStep((previous) => previous - 1)
    }
  }

  const buildPayload = (): ProfilePayload => {
    const ageNum = Number.parseInt(form.age, 10)
    const weightNum = Number.parseFloat(form.weight)
    const heightNum = Number.parseFloat(form.height)
    const rateNum = Number.parseFloat(form.weight_goal_rate)
    const mealsNum = Number.parseInt(form.meals_per_day, 10)
    return {
      name: form.name.trim(),
      location: form.location.trim(),
      age: ageNum,
      gender: form.gender,
      weight: weightNum,
      height: heightNum,
      activity_level: form.activity_level,
      allergies: form.allergies,
      physical_goal: form.physical_goal,
      weight_goal_rate: rateNum,
      meals_per_day: mealsNum,
    }
  }

  const handleFinish = async (): Promise<void> => {
    setErrorMessage(null)
    const message = validateStep(4)
    if (message) {
      setErrorMessage(message)
      return
    }
    if (!token) {
      setErrorMessage(t('onboarding.sessionExpired'))
      return
    }

    setIsSubmitting(true)
    try {
      const payload = buildPayload()
      const response = await apiFetch(`${getApiBaseUrl()}/api/profile`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify(payload),
      })

      if (!response.ok) {
        throw new Error(await parseApiError(response, 'Could not save profile.'))
      }

      navigate('/', { replace: true })
    } catch (error) {
      const detail = error instanceof Error ? error.message : t('common.error')
      setErrorMessage(detail)
    } finally {
      setIsSubmitting(false)
    }
  }

  if (!isAuthenticated || !token) {
    return null
  }

  return (
    <div className="flex min-h-screen flex-col bg-slate-100">
      <header className="px-6 py-5 sm:px-10">
        <span className="text-lg font-bold uppercase tracking-tight text-[var(--color-primary)]">
          {t('brand.name')}
        </span>
      </header>

      <div className="flex flex-1 items-center justify-center px-4 pb-16 pt-4 sm:px-6">
        <div className="w-full max-w-lg rounded-2xl bg-white p-8 shadow-lg sm:p-10">
          <div className="mb-8 h-1.5 w-full overflow-hidden rounded-full bg-slate-200">
            <div
              className="h-full rounded-full bg-gradient-to-r from-emerald-600 to-sky-600 transition-all duration-300 ease-out"
              style={{ width: `${progressPercent}%` }}
            />
          </div>

          {step === 1 ? (
            <StepBasics form={form} setForm={setForm} />
          ) : null}
          {step === 2 ? (
            <StepBodyMetrics form={form} setForm={setForm} />
          ) : null}
          {step === 3 ? (
            <StepLifestyle form={form} setForm={setForm} />
          ) : null}
          {step === 4 ? (
            <StepGoals form={form} setForm={setForm} />
          ) : null}

          {errorMessage ? (
            <p
              className="mt-6 rounded-lg border border-[var(--color-active-border)] bg-[var(--color-active-bg)] px-3 py-2 text-sm font-semibold text-[var(--color-secondary)]"
              role="alert"
            >
              {errorMessage}
            </p>
          ) : null}

          <div className="mt-10 flex flex-col gap-3 sm:flex-row sm:justify-between">
            <button
              type="button"
              onClick={goBack}
              disabled={step === 1 || isSubmitting}
              className="order-2 rounded-lg border border-[var(--color-secondary)] px-8 py-3 text-sm font-semibold uppercase tracking-wide text-[var(--color-secondary)] transition hover:bg-[var(--color-surface-soft)] disabled:cursor-not-allowed disabled:opacity-40 sm:order-1"
            >
              {t('onboarding.back')}
            </button>
            {step < TOTAL_STEPS ? (
              <button
                type="button"
                onClick={goNext}
                disabled={isSubmitting}
                className="order-1 rounded-lg bg-gradient-to-r from-emerald-600 to-sky-600 px-8 py-3 text-sm font-semibold uppercase tracking-wide text-white shadow-sm transition hover:from-emerald-700 hover:to-sky-700 disabled:cursor-not-allowed disabled:opacity-60 sm:order-2 sm:ml-auto"
              >
                {t('onboarding.next')}
              </button>
            ) : (
              <button
                type="button"
                onClick={() => void handleFinish()}
                disabled={isSubmitting}
                className="order-1 rounded-lg bg-gradient-to-r from-emerald-600 to-sky-600 px-8 py-3 text-sm font-semibold uppercase tracking-wide text-white shadow-sm transition hover:from-emerald-700 hover:to-sky-700 disabled:cursor-not-allowed disabled:opacity-60 sm:order-2 sm:ml-auto"
              >
                {isSubmitting ? t('onboarding.saving') : t('onboarding.finish')}
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

interface StepProps {
  form: OnboardingFormState
  setForm: Dispatch<SetStateAction<OnboardingFormState>>
}

function StepBasics({ form, setForm }: StepProps) {
  const { t } = useTranslation()
  return (
    <div className="space-y-6 text-center">
      <h1 className="text-2xl font-bold text-[var(--color-primary)] sm:text-3xl">
        {t('onboarding.basicsTitle')}
      </h1>
      <p className="text-sm font-semibold text-[var(--color-secondary)] sm:text-base">
        {t('onboarding.basicsBody')}
      </p>

      <div className="space-y-2 text-left">
        <label htmlFor="onboarding-name" className="text-sm font-semibold text-[var(--color-primary)]">
          {t('onboarding.nameLabel')}
        </label>
        <input
          id="onboarding-name"
          type="text"
          required
          value={form.name}
          onChange={(event) =>
            setForm((previous) => ({ ...previous, name: event.target.value }))
          }
          className="w-full rounded-xl border border-slate-300 px-4 py-3 text-base text-[var(--color-primary)] placeholder:text-[var(--color-border)] outline-none transition focus:border-[var(--color-secondary)] focus:ring-2 focus:ring-[var(--color-focus-ring)]"
          placeholder={t('onboarding.namePlaceholder')}
        />
      </div>

      <div className="space-y-2 text-left">
        <label htmlFor="onboarding-age" className="text-sm font-semibold text-[var(--color-primary)]">
          {t('onboarding.ageLabel')}
        </label>
        <input
          id="onboarding-age"
          type="number"
          min={10}
          max={120}
          inputMode="numeric"
          value={form.age}
          onChange={(event) => setForm((previous) => ({ ...previous, age: event.target.value }))}
          className="w-full rounded-xl border border-slate-300 px-4 py-3 text-center text-lg text-[var(--color-primary)] placeholder:text-[var(--color-border)] outline-none transition focus:border-[var(--color-secondary)] focus:ring-2 focus:ring-[var(--color-focus-ring)]"
          placeholder={t('onboarding.agePlaceholder')}
        />
      </div>

      <div className="space-y-3 text-left">
        <span className="text-sm font-semibold text-[var(--color-primary)]">{t('onboarding.genderLabel')}</span>
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          {(
            [
              { value: 'male' as const, label: t('onboarding.genderMale') },
              { value: 'female' as const, label: t('onboarding.genderFemale') },
            ] as const
          ).map((option) => (
            <button
              key={option.value}
              type="button"
              onClick={() => setForm((previous) => ({ ...previous, gender: option.value }))}
              className={`rounded-2xl border px-4 py-4 text-center text-base font-semibold transition ${form.gender === option.value
                ? 'border-[var(--color-active-border)] bg-[var(--color-active-bg)] text-[var(--color-secondary)]'
                : 'border-slate-200 bg-white text-[var(--color-secondary)] hover:border-slate-300'
                }`}
            >
              {option.label}
            </button>
          ))}
        </div>
        <p className="text-xs font-semibold text-[var(--color-secondary)]">
          {t('onboarding.genderHint')}
        </p>
      </div>

      <div className="space-y-2 text-left">
        <label htmlFor="onboarding-location" className="text-sm font-semibold text-[var(--color-primary)]">
          {t('onboarding.locationLabel')}
        </label>
        <input
          id="onboarding-location"
          type="text"
          value={form.location}
          onChange={(event) =>
            setForm((previous) => ({ ...previous, location: event.target.value }))
          }
          className="w-full rounded-xl border border-slate-300 px-4 py-3 text-base text-[var(--color-primary)] placeholder:text-[var(--color-border)] outline-none transition focus:border-[var(--color-secondary)] focus:ring-2 focus:ring-[var(--color-focus-ring)]"
          placeholder={t('onboarding.locationPlaceholder')}
        />
      </div>
    </div>
  )
}

function StepBodyMetrics({ form, setForm }: StepProps) {
  const { t } = useTranslation()

  return (
    <div className="space-y-6 text-center">
      <h1 className="text-2xl font-bold text-[var(--color-primary)] sm:text-3xl">{t('onboarding.bodyMetricsTitle')}</h1>
      <p className="text-sm font-semibold text-[var(--color-secondary)] sm:text-base">
        {t('onboarding.bodyMetricsBody')}
      </p>

      <div className="grid gap-4 sm:grid-cols-2">
        <div className="space-y-2 text-left">
          <label htmlFor="onboarding-weight" className="text-sm font-semibold text-[var(--color-primary)]">
            {t('onboarding.weightLabel')} (kg)
          </label>
          <input
            id="onboarding-weight"
            type="number"
            min={10}
            max={400}
            step="0.1"
            inputMode="decimal"
            value={form.weight}
            onChange={(event) =>
              setForm((previous) => ({ ...previous, weight: event.target.value }))
            }
            className="w-full rounded-xl border border-slate-300 px-4 py-3 text-lg text-[var(--color-primary)] placeholder:text-[var(--color-border)] outline-none transition focus:border-[var(--color-secondary)] focus:ring-2 focus:ring-[var(--color-focus-ring)]"
            placeholder={t('onboarding.weightPlaceholder')}
          />
        </div>
        <div className="space-y-2 text-left">
          <label htmlFor="onboarding-height" className="text-sm font-semibold text-[var(--color-primary)]">
            {t('onboarding.heightLabel')} (cm)
          </label>
          <input
            id="onboarding-height"
            type="number"
            min={50}
            max={250}
            step="1"
            inputMode="decimal"
            value={form.height}
            onChange={(event) =>
              setForm((previous) => ({ ...previous, height: event.target.value }))
            }
            className="w-full rounded-xl border border-slate-300 px-4 py-3 text-lg text-[var(--color-primary)] placeholder:text-[var(--color-border)] outline-none transition focus:border-[var(--color-secondary)] focus:ring-2 focus:ring-[var(--color-focus-ring)]"
            placeholder={t('onboarding.heightPlaceholder')}
          />
        </div>
      </div>
    </div>
  )
}

function StepLifestyle({ form, setForm }: StepProps) {
  const { t } = useTranslation()
  return (
    <div className="space-y-6 text-center">
      <h1 className="text-2xl font-bold text-[var(--color-primary)] sm:text-3xl">{t('onboarding.lifestyleTitle')}</h1>
      <p className="text-sm font-semibold text-[var(--color-secondary)] sm:text-base">
        {t('onboarding.lifestyleBody')}
      </p>

      <div className="space-y-3 text-left">
        <span className="text-sm font-bold text-[var(--color-primary)]">{t('onboarding.activityLevelLabel')}</span>
        <div className="grid gap-3">
          {ACTIVITY_OPTIONS.map((option) => (
            <button
              key={option.value}
              type="button"
              onClick={() =>
                setForm((previous) => ({ ...previous, activity_level: option.value }))
              }
              className={`rounded-2xl border px-4 py-3 text-left transition ${form.activity_level === option.value
                ? 'border-[var(--color-active-border)] bg-[var(--color-active-bg)]'
                : 'border-slate-200 bg-white hover:border-slate-300'
                }`}
            >
              <span className="block font-semibold text-[var(--color-primary)]">{option.labelEs}</span>
              <span className="block text-xs font-semibold text-[var(--color-secondary)]">{option.hintEs}</span>
            </button>
          ))}
        </div>
      </div>

      <div className="space-y-3 text-left">
        <span className="text-sm font-bold text-[var(--color-primary)]">{t('onboarding.allergiesLabel')}</span>
        <p className="text-xs font-semibold text-[var(--color-secondary)]">{t('onboarding.allergiesBody')}</p>
        <div className="flex flex-wrap gap-2">
          {ALLERGY_PRESET_IDS.map((id) => {
            const selected = form.allergies.includes(id)
            return (
              <button
                key={id}
                type="button"
                onClick={() =>
                  setForm((previous) => ({
                    ...previous,
                    allergies: toggleAllergy(previous.allergies, id),
                  }))
                }
                className={`rounded-full border px-3 py-1.5 text-sm font-medium transition ${selected
                  ? 'border-[var(--color-active-border)] bg-[var(--color-active-bg)] text-[var(--color-secondary)]'
                  : 'border-slate-200 bg-white text-[var(--color-secondary)] hover:border-slate-300'
                  }`}
              >
                {t(`common.allergens.${id}`)}
              </button>
            )
          })}
        </div>
      </div>
    </div>
  )
}

function StepGoals({ form, setForm }: StepProps) {
  const { t } = useTranslation()
  return (
    <div className="space-y-6 text-center">
      <h1 className="text-2xl font-bold text-[var(--color-primary)] sm:text-3xl">
        {t('onboarding.goalsTitle')}
      </h1>
      <p className="text-sm font-semibold text-[var(--color-secondary)] sm:text-base">
        {t('onboarding.goalsBody')}
      </p>

      <div className="space-y-3 text-left">
        <span className="text-sm font-bold text-[var(--color-primary)]">{t('onboarding.physicalGoalLabel')}</span>
        <div className="grid gap-3 sm:grid-cols-3">
          {PHYSICAL_GOAL_OPTIONS.map((option) => (
            <button
              key={option.value}
              type="button"
              onClick={() =>
                setForm((previous) => ({ ...previous, physical_goal: option.value }))
              }
              className={`rounded-2xl border px-3 py-4 text-center transition ${form.physical_goal === option.value
                ? 'border-[var(--color-active-border)] bg-[var(--color-active-bg)]'
                : 'border-slate-200 bg-white hover:border-slate-300'
                }`}
            >
              <span className="block text-sm font-semibold text-[var(--color-primary)]">{option.labelEs}</span>
              <span className="mt-1 block text-xs font-semibold text-[var(--color-secondary)]">{option.hintEs}</span>
            </button>
          ))}
        </div>
      </div>

      <div className="space-y-3 text-left">
        <span className="text-sm font-bold text-[var(--color-primary)]">
          {t('onboarding.weightGoalRateLabel')}
        </span>
        <p className="text-xs font-semibold text-[var(--color-secondary)]">
          {t('onboarding.weightGoalRateBody')}
        </p>
        <div className="flex flex-wrap gap-2">
          {WEIGHT_GOAL_PRESETS.map((preset) => {
            const selected = form.weight_goal_rate === String(preset.value)
            return (
              <button
                key={preset.value}
                type="button"
                onClick={() =>
                  setForm((previous) => ({
                    ...previous,
                    weight_goal_rate: String(preset.value),
                  }))
                }
                className={`rounded-full border px-3 py-1.5 text-sm font-medium transition ${selected
                  ? 'border-[var(--color-active-border)] bg-[var(--color-active-bg)] text-[var(--color-secondary)]'
                  : 'border-slate-200 bg-white text-[var(--color-secondary)] hover:border-slate-300'
                  }`}
              >
                {preset.labelEs}
              </button>
            )
          })}
        </div>
        <div className="pt-2">
          <label htmlFor="onboarding-rate-custom" className="sr-only">
            {t('onboarding.weightGoalRateCustomLabel')}
          </label>
          <input
            id="onboarding-rate-custom"
            type="number"
            min={-1.5}
            max={1.5}
            step={0.05}
            value={form.weight_goal_rate}
            onChange={(event) =>
              setForm((previous) => ({ ...previous, weight_goal_rate: event.target.value }))
            }
            className="w-full rounded-xl border border-slate-300 px-4 py-2 text-sm text-[var(--color-primary)] placeholder:text-[var(--color-border)] outline-none transition focus:border-[var(--color-secondary)] focus:ring-2 focus:ring-[var(--color-focus-ring)]"
            placeholder={t('onboarding.weightGoalRateCustomPlaceholder')}
          />
        </div>
      </div>

      {/* New consistent layout row for Meals per Day selection */}
      <div className="space-y-2 text-left pt-2">
        <label htmlFor="onboarding-meals-per-day" className="text-sm font-bold text-[var(--color-primary)]">
          {t('profile.mealsPerDayLabel', 'Comidas al día')}
        </label>
        <select
          id="onboarding-meals-per-day"
          value={form.meals_per_day}
          onChange={(event) =>
            setForm((previous) => ({ ...previous, meals_per_day: event.target.value }))
          }
          className="w-full rounded-xl border border-slate-300 bg-white px-4 py-3 text-sm text-[var(--color-primary)] outline-none transition focus:border-[var(--color-secondary)] focus:ring-2 focus:ring-[var(--color-focus-ring)]"
        >
          {[1, 2, 3, 4, 5, 6].map((num) => (
            <option key={num} value={String(num)}>
              {num}
            </option>
          ))}
        </select>
      </div>
    </div>
  )
}
