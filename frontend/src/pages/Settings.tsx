import { useCallback, useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { getApiBaseUrl } from '../config/apiBaseUrl'
import { apiFetch } from '../api/apiFetch'
import { useAuth } from '../context/AuthContext'
import { usePreferences } from '../context/PreferencesContext'
import { useTranslation } from 'react-i18next'
import { DeleteButton } from '../components/DeleteButton'

type LanguageValue = 'es' | 'en'
type UnitsValue = 'metric' | 'imperial'

type SectionFeedback = { type: 'success' | 'error'; message: string } | null

function isAuthFailureMessage(message: string): boolean {
  const trimmed = message.trim()
  if (trimmed === 'Could not validate credentials.') {
    return true
  }
  const lower = trimmed.toLowerCase()
  return lower.includes('could not validate credentials') || trimmed.includes('HTTP 401')
}

async function parseDetail(response: Response, fallback: string): Promise<string> {
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
    // Ignore
  }
  return fallback
}

function SectionAlert({ feedback }: { feedback: SectionFeedback }) {
  if (!feedback) {
    return null
  }
  return (
    <p
      className="rounded-lg border border-[var(--color-active-border)] bg-[var(--color-active-bg)] px-3 py-2 text-sm font-semibold text-[var(--color-secondary)]"
      role={feedback.type === 'error' ? 'alert' : undefined}
    >
      {feedback.message}
    </p>
  )
}

export function Settings() {
  const navigate = useNavigate()
  const { isAuthenticated, token, logout } = useAuth()
  const { t } = useTranslation()
  const preferences = usePreferences()

  const [isLoading, setIsLoading] = useState<boolean>(true)
  const [passwordCredential, setPasswordCredential] = useState<boolean | null>(null)

  const [preferencesFeedback, setPreferencesFeedback] = useState<SectionFeedback>(null)
  const [accountFeedback, setAccountFeedback] = useState<SectionFeedback>(null)
  const [securityFeedback, setSecurityFeedback] = useState<SectionFeedback>(null)

  const [draftLanguage, setDraftLanguage] = useState<LanguageValue>(preferences.language)
  const [draftUnits, setDraftUnits] = useState<UnitsValue>(preferences.units)

  const [emailNew, setEmailNew] = useState<string>('')
  const [emailNewConfirm, setEmailNewConfirm] = useState<string>('')
  const [emailCurrentPassword, setEmailCurrentPassword] = useState<string>('')

  const [passwordCurrentPassword, setPasswordCurrentPassword] = useState<string>('')
  const [passwordNew, setPasswordNew] = useState<string>('')
  const [passwordNewConfirm, setPasswordNewConfirm] = useState<string>('')

  const [deletePassword, setDeletePassword] = useState<string>('')
  const [deleteError, setDeleteError] = useState<string | null>(null)
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState<boolean>(false)

  const loadPasswordCredential = useCallback(async (): Promise<void> => {
    if (!token) {
      return
    }
    try {
      const response = await apiFetch(`${getApiBaseUrl()}/api/profile`, {
        method: 'GET',
        headers: { Authorization: `Bearer ${token}` },
      })
      if (!response.ok) {
        setPasswordCredential(true)
        return
      }
      const data: unknown = await response.json()
      if (
        typeof data === 'object' &&
        data !== null &&
        'password_credential' in data &&
        typeof (data as { password_credential: unknown }).password_credential === 'boolean'
      ) {
        setPasswordCredential((data as { password_credential: boolean }).password_credential)
      } else {
        setPasswordCredential(true)
      }
    } catch {
      setPasswordCredential(true)
    }
  }, [token])

  useEffect(() => {
    if (!isAuthenticated || !token) {
      navigate('/login')
      return
    }
    setIsLoading(false)
    setDraftLanguage(preferences.language)
    setDraftUnits(preferences.units)
    void loadPasswordCredential()
  }, [isAuthenticated, token, navigate, preferences.language, preferences.units, loadPasswordCredential])

  const preferencesPayload = useMemo(() => {
    return { language: draftLanguage, units: draftUnits }
  }, [draftLanguage, draftUnits])

  const handleSavePreferences = async (): Promise<void> => {
    if (!token) {
      return
    }
    setPreferencesFeedback(null)
    try {
      const response = await apiFetch(`${getApiBaseUrl()}/api/settings/preferences`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify(preferencesPayload),
      })
      if (!response.ok) {
        const detail = await parseDetail(
          response,
          `Failed to update preferences (HTTP ${response.status}).`,
        )
        if (isAuthFailureMessage(detail)) {
          logout()
          return
        }
        throw new Error(detail)
      }
      preferences.setLanguage(draftLanguage)
      preferences.setUnits(draftUnits)
      setPreferencesFeedback({ type: 'success', message: t('settings.preferencesSaved') })
    } catch (error) {
      const detail = error instanceof Error ? error.message : 'Unexpected error.'
      setPreferencesFeedback({ type: 'error', message: detail })
    }
  }

  const handleUpdateEmail = async (): Promise<void> => {
    if (!token) {
      return
    }
    setAccountFeedback(null)
    const trimmed = emailNew.trim()
    const trimmedConfirm = emailNewConfirm.trim()
    if (trimmed !== trimmedConfirm) {
      setAccountFeedback({ type: 'error', message: t('settings.emailMismatch') })
      return
    }
    try {
      const response = await apiFetch(`${getApiBaseUrl()}/api/settings/email`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          new_email: trimmed,
          current_password: passwordCredential === false ? '' : emailCurrentPassword,
        }),
      })
      if (!response.ok) {
        const detail = await parseDetail(
          response,
          `Failed to update email (HTTP ${response.status}).`,
        )
        if (isAuthFailureMessage(detail)) {
          logout()
          return
        }
        throw new Error(detail)
      }
      setAccountFeedback({ type: 'success', message: t('settings.emailUpdated') })
      setEmailNew('')
      setEmailNewConfirm('')
      setEmailCurrentPassword('')
    } catch (error) {
      const detail = error instanceof Error ? error.message : 'Unexpected error.'
      setAccountFeedback({ type: 'error', message: detail })
    }
  }

  const handleChangePassword = async (): Promise<void> => {
    if (!token) {
      return
    }
    setSecurityFeedback(null)
    if (passwordNew !== passwordNewConfirm) {
      setSecurityFeedback({ type: 'error', message: t('settings.passwordMismatch') })
      return
    }
    try {
      const response = await apiFetch(`${getApiBaseUrl()}/api/settings/password`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          current_password: passwordCredential === false ? '' : passwordCurrentPassword,
          new_password: passwordNew,
        }),
      })
      if (!response.ok) {
        const detail = await parseDetail(
          response,
          `Failed to update password (HTTP ${response.status}).`,
        )
        if (isAuthFailureMessage(detail)) {
          logout()
          return
        }
        throw new Error(detail)
      }
      setSecurityFeedback({ type: 'success', message: t('settings.passwordUpdated') })
      setPasswordCurrentPassword('')
      setPasswordNew('')
      setPasswordNewConfirm('')
      setPasswordCredential(true)
    } catch (error) {
      const detail = error instanceof Error ? error.message : t('common.error')
      setSecurityFeedback({ type: 'error', message: detail })
    }
  }

  const handleDeleteAccount = async (): Promise<void> => {
    if (!token) {
      return
    }
    if (passwordCredential === true && !deletePassword.trim()) {
      setDeleteError(t('settings.passwordRequired'))
      return
    }

    setDeleteError(null)

    try {
      const response = await apiFetch(`${getApiBaseUrl()}/api/settings/account`, {
        method: 'DELETE',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          current_password: passwordCredential === false ? '' : deletePassword,
        }),
      })
      if (!response.ok) {
        const detail = await parseDetail(
          response,
          `Failed to delete account (HTTP ${response.status}).`,
        )
        if (isAuthFailureMessage(detail)) {
          logout()
          return
        }
        throw new Error(detail)
      }
      setIsDeleteModalOpen(false)
      logout()
    } catch (error) {
      const detail = error instanceof Error ? error.message : t('common.error')
      setDeleteError(detail)
    }
  }

  const openDeleteModal = (): void => {
    setDeleteError(null)
    setDeletePassword('')
    setIsDeleteModalOpen(true)
  }

  if (!isAuthenticated) {
    return null
  }

  if (isLoading) {
    return (
      <section className="rounded-2xl border border-[var(--color-border)] bg-[var(--color-surface)] p-6 shadow-sm">
        <h1 className="text-xl font-bold text-[var(--color-primary)] sm:text-2xl">{t('settings.title')}</h1>
        <p className="mt-3 text-sm font-semibold text-[var(--color-secondary)]">{t('settings.failedToLoad')}</p>
      </section>
    )
  }

  const showEmailPasswordField = passwordCredential !== false
  const showCurrentPasswordForChange = passwordCredential !== false

  return (
    <section className="space-y-6">
      <header>
        <h1 className="text-xl font-bold text-[var(--color-primary)] sm:text-2xl">{t('settings.title')}</h1>
        <p className="mt-1 text-sm font-semibold text-[var(--color-secondary)]">{t('settings.subtitle')}</p>
      </header>

      <div className="space-y-6">
        <div className="space-y-4 rounded-2xl border border-[var(--color-border)] bg-white p-6 shadow-sm">
          <h2 className="text-sm font-bold uppercase tracking-wide text-[var(--color-primary)]">
            {t('settings.preferences')}
          </h2>
          <SectionAlert feedback={preferencesFeedback} />

          <div className="space-y-3">
            <span className="block text-sm font-semibold text-[var(--color-primary)]">{t('settings.language')}</span>
            <div className="grid gap-3 sm:grid-cols-2">
              {(['es', 'en'] as LanguageValue[]).map((value) => {
                const selected = draftLanguage === value
                return (
                  <button
                    key={value}
                    type="button"
                    onClick={() => setDraftLanguage(value)}
                    className={`rounded-2xl border px-4 py-3 text-sm font-semibold transition ${selected
                      ? 'border-[var(--color-active-border)] bg-[var(--color-active-bg)] text-[var(--color-secondary)]'
                      : 'border-slate-200 bg-white text-[var(--color-secondary)] hover:border-slate-300 hover:bg-[var(--color-surface-soft)]'
                      }`}
                  >
                    {value === 'es' ? t('settings.spanish') : t('settings.english')}
                  </button>
                )
              })}
            </div>
          </div>

          <div className="space-y-3">
            <span className="block text-sm font-semibold text-[var(--color-primary)]">{t('settings.units')}</span>
            <div className="grid gap-3 sm:grid-cols-2">
              {(['metric', 'imperial'] as UnitsValue[]).map((value) => {
                const selected = draftUnits === value
                return (
                  <button
                    key={value}
                    type="button"
                    onClick={() => setDraftUnits(value)}
                    className={`rounded-2xl border px-4 py-3 text-sm font-semibold transition ${selected
                      ? 'border-[var(--color-active-border)] bg-[var(--color-active-bg)] text-[var(--color-secondary)]'
                      : 'border-slate-200 bg-white text-[var(--color-secondary)] hover:border-slate-300 hover:bg-[var(--color-surface-soft)]'
                      }`}
                  >
                    {value === 'metric' ? t('settings.metric') : t('settings.imperial')}
                  </button>
                )
              })}
            </div>
          </div>

          <div className="pt-2">
            <button
              type="button"
              onClick={() => void handleSavePreferences()}
              className="w-full rounded-xl bg-gradient-to-r from-emerald-600 to-sky-600 px-4 py-3 text-sm font-semibold uppercase tracking-wide text-white shadow-sm transition hover:from-emerald-700 hover:to-sky-700"
            >
              {t('settings.savePreferences')}
            </button>
          </div>
        </div>

        <div className="grid gap-4 lg:grid-cols-2">
          <div className="space-y-4 rounded-2xl border border-[var(--color-border)] bg-white p-6 shadow-sm">
            <h2 className="text-sm font-bold uppercase tracking-wide text-[var(--color-primary)]">
              {t('settings.accountTitle')}
            </h2>
            <SectionAlert feedback={accountFeedback} />

            <div className="space-y-3">
              <label className="block text-sm font-medium text-[var(--color-primary)]" htmlFor="new-email">
                {t('settings.newEmailLabel')}
              </label>
              <input
                id="new-email"
                value={emailNew}
                onChange={(event) => setEmailNew(event.target.value)}
                className="w-full rounded-xl border border-slate-300 px-4 py-3 text-sm text-[var(--color-primary)] placeholder:text-[var(--color-border)] outline-none transition focus:border-[var(--color-secondary)] focus:ring-2 focus:ring-[var(--color-focus-ring)]"
                placeholder={t('settings.newEmailPlaceholder')}
                autoComplete="email"
              />
            </div>

            <div className="space-y-3">
              <label className="block text-sm font-medium text-[var(--color-primary)]" htmlFor="new-email-confirm">
                {t('settings.confirmNewEmailLabel')}
              </label>
              <input
                id="new-email-confirm"
                value={emailNewConfirm}
                onChange={(event) => setEmailNewConfirm(event.target.value)}
                className="w-full rounded-xl border border-slate-300 px-4 py-3 text-sm text-[var(--color-primary)] placeholder:text-[var(--color-border)] outline-none transition focus:border-[var(--color-secondary)] focus:ring-2 focus:ring-[var(--color-focus-ring)]"
                placeholder={t('settings.newEmailPlaceholder')}
                autoComplete="off"
              />
            </div>

            {showEmailPasswordField ? (
              <div className="space-y-3">
                <label
                  className="block text-sm font-medium text-[var(--color-primary)]"
                  htmlFor="email-current-password"
                >
                  {t('settings.currentPasswordLabel')}
                </label>
                <input
                  id="email-current-password"
                  type="password"
                  value={emailCurrentPassword}
                  onChange={(event) => setEmailCurrentPassword(event.target.value)}
                  className="w-full rounded-xl border border-slate-300 px-4 py-3 text-sm text-[var(--color-primary)] placeholder:text-[var(--color-border)] outline-none transition focus:border-[var(--color-secondary)] focus:ring-2 focus:ring-[var(--color-focus-ring)]"
                  placeholder={t('auth.passwordPlaceholder')}
                  autoComplete="current-password"
                />
              </div>
            ) : null}

            <button
              type="button"
              onClick={() => void handleUpdateEmail()}
              className="w-full rounded-xl bg-gradient-to-r from-emerald-600 to-sky-600 px-4 py-3 text-sm font-semibold uppercase tracking-wide text-white shadow-sm transition hover:from-emerald-700 hover:to-sky-700"
            >
              {t('settings.updateEmail')}
            </button>
          </div>

          <div className="space-y-4 rounded-2xl border border-[var(--color-border)] bg-white p-6 shadow-sm">
            <h2 className="text-sm font-bold uppercase tracking-wide text-[var(--color-primary)]">
              {t('settings.securityTitle')}
            </h2>
            <SectionAlert feedback={securityFeedback} />

            {showCurrentPasswordForChange ? (
              <div className="space-y-3">
                <label
                  className="block text-sm font-medium text-[var(--color-primary)]"
                  htmlFor="password-current-password"
                >
                  {t('settings.currentPasswordLabel')}
                </label>
                <input
                  id="password-current-password"
                  type="password"
                  value={passwordCurrentPassword}
                  onChange={(event) => setPasswordCurrentPassword(event.target.value)}
                  className="w-full rounded-xl border border-slate-300 px-4 py-3 text-sm text-[var(--color-primary)] placeholder:text-[var(--color-border)] outline-none transition focus:border-[var(--color-secondary)] focus:ring-2 focus:ring-[var(--color-focus-ring)]"
                  placeholder={t('auth.passwordPlaceholder')}
                  autoComplete="current-password"
                />
              </div>
            ) : (
              <p className="text-sm font-semibold text-[var(--color-secondary)]">{t('settings.oauthSetPasswordHint')}</p>
            )}

            <div className="space-y-3">
              <label
                className="block text-sm font-medium text-[var(--color-primary)]"
                htmlFor="password-new"
              >
                {t('settings.newPasswordLabel')}
              </label>
              <input
                id="password-new"
                type="password"
                value={passwordNew}
                onChange={(event) => setPasswordNew(event.target.value)}
                className="w-full rounded-xl border border-slate-300 px-4 py-3 text-sm text-[var(--color-primary)] placeholder:text-[var(--color-border)] outline-none transition focus:border-[var(--color-secondary)] focus:ring-2 focus:ring-[var(--color-focus-ring)]"
                placeholder={t('auth.passwordPlaceholder')}
                autoComplete="new-password"
              />
            </div>

            <div className="space-y-3">
              <label
                className="block text-sm font-medium text-[var(--color-primary)]"
                htmlFor="password-new-confirm"
              >
                {t('settings.confirmNewPasswordLabel')}
              </label>
              <input
                id="password-new-confirm"
                type="password"
                value={passwordNewConfirm}
                onChange={(event) => setPasswordNewConfirm(event.target.value)}
                className="w-full rounded-xl border border-slate-300 px-4 py-3 text-sm text-[var(--color-primary)] placeholder:text-[var(--color-border)] outline-none transition focus:border-[var(--color-secondary)] focus:ring-2 focus:ring-[var(--color-focus-ring)]"
                placeholder={t('auth.passwordPlaceholder')}
                autoComplete="new-password"
              />
            </div>

            <button
              type="button"
              onClick={() => void handleChangePassword()}
              className="w-full rounded-xl bg-gradient-to-r from-emerald-600 to-sky-600 px-4 py-3 text-sm font-semibold uppercase tracking-wide text-white shadow-sm transition hover:from-emerald-700 hover:to-sky-700"
            >
              {passwordCredential === false ? t('settings.setPassword') : t('settings.changePassword')}
            </button>
          </div>
        </div>

        <div className="space-y-4 rounded-2xl border border-[var(--color-border)] bg-white p-6 shadow-sm">
          <h2 className="text-sm font-bold uppercase tracking-wide text-[var(--color-primary)]">
            {t('settings.dangerZoneTitle')}
          </h2>

          <p className="text-sm font-semibold text-[var(--color-secondary)]">
            {t('settings.dangerZoneBody')}
          </p>

          <button
            type="button"
            onClick={() => openDeleteModal()}
            className="w-full rounded-xl bg-[var(--color-red-600)] px-4 py-3 text-sm font-semibold uppercase tracking-wide text-white shadow-sm transition hover:bg-[var(--color-red-700)]"
          >
            {t('settings.deleteAccount')}
          </button>
        </div>
      </div>

      {isDeleteModalOpen ? (
        <div className="fixed inset-0 z-[70] flex items-center justify-center bg-slate-950/60 p-4">
          <div className="w-full max-w-md rounded-2xl bg-white p-6 shadow-xl">
            <h2 className="text-lg font-bold text-[var(--color-primary)]">{t('settings.confirmAccountDeletionTitle')}</h2>
            <p className="mt-2 text-sm font-semibold text-[var(--color-secondary)]">
              {passwordCredential === false
                ? t('settings.confirmAccountDeletionBodyOAuth')
                : t('settings.confirmAccountDeletionBody')}
            </p>
            {deleteError ? (
              <p className="mt-3 rounded-lg border border-[var(--color-active-border)] bg-[var(--color-active-bg)] px-3 py-2 text-sm font-semibold text-[var(--color-secondary)]" role="alert">
                {deleteError}
              </p>
            ) : null}
            {passwordCredential !== false ? (
              <div className="mt-4 space-y-2">
                <label className="block text-sm font-medium text-[var(--color-primary)]" htmlFor="delete-password">
                  {t('settings.currentPasswordLabel')}
                </label>
                <input
                  id="delete-password"
                  type="password"
                  value={deletePassword}
                  onChange={(event) => setDeletePassword(event.target.value)}
                  className="w-full rounded-xl border border-slate-300 px-4 py-3 text-sm text-[var(--color-primary)] placeholder:text-[var(--color-border)] outline-none transition focus:border-[var(--color-secondary)] focus:ring-2 focus:ring-[var(--color-focus-ring)]"
                  placeholder={t('auth.passwordPlaceholder')}
                  autoComplete="current-password"
                />
              </div>
            ) : null}
            <div className="mt-6 flex justify-end gap-2">
              <button
                type="button"
                onClick={() => setIsDeleteModalOpen(false)}
                className="rounded-lg border border-slate-300 px-4 py-2 text-sm font-semibold text-[var(--color-secondary)] transition hover:bg-[var(--color-surface-soft)]"
              >
                {t('common.cancel')}
              </button>
              <DeleteButton onClick={() => void handleDeleteAccount()}>
                {t('settings.delete')}
              </DeleteButton>
            </div>
          </div>
        </div>
      ) : null}
    </section>
  )
}
