import { useState } from 'react'
import type { FormEvent } from 'react'
import { useNavigate } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import { getApiBaseUrl } from '../config/apiBaseUrl'
import { useAuth } from '../context/AuthContext'

type AuthMode = 'login' | 'register'

export function Login() {
  const navigate = useNavigate()
  const { login, register } = useAuth()
  const { t } = useTranslation()

  const [mode, setMode] = useState<AuthMode>('login')
  const [email, setEmail] = useState<string>('')
  const [name, setName] = useState<string>('')
  const [password, setPassword] = useState<string>('')
  const [confirmPassword, setConfirmPassword] = useState<string>('')
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false)

  // New state for GDPR compliance and privacy modal control
  const [acceptedTerms, setAcceptedTerms] = useState<boolean>(false)
  const [showPrivacyModal, setShowPrivacyModal] = useState<boolean>(false)

  const submitLabel = mode === 'login' ? t('auth.login') : t('auth.register')
  const switchLabel = mode === 'login' ? t('auth.switchToRegister') : t('auth.switchToLogin')

  const handleSubmit = async (event: FormEvent<HTMLFormElement>): Promise<void> => {
    event.preventDefault()
    setErrorMessage(null)
    setIsSubmitting(true)

    try {
      if (mode === 'register') {
        if (!acceptedTerms) {
          setErrorMessage(t('auth.privacyConsentRequired'))
          setIsSubmitting(false)
          return
        }
        if (password !== confirmPassword) {
          setErrorMessage(t('auth.passwordsDoNotMatch'))
          setIsSubmitting(false)
          return
        }
        await register(email.trim(), password, name.trim())
        return
      }
      await login(email.trim(), password)
      navigate('/')
    } catch (error) {
      const detail = error instanceof Error ? error.message : t('auth.authenticationFailed')
      setErrorMessage(detail)
    } finally {
      setIsSubmitting(false)
    }
  }

  const handleGoogleSignIn = (): void => {
    // Structural guard clause to block authorization flow if consent is missing during registration
    if (mode === 'register' && !acceptedTerms) {
      return
    }
    window.location.href = `${getApiBaseUrl()}/api/auth/google`
  }

  return (
    <section className="min-h-screen">
      <div className="grid min-h-screen grid-cols-1 lg:grid-cols-[1.25fr_1fr]">
        <aside className="hidden bg-gradient-to-br from-emerald-700 via-teal-600 to-sky-600 lg:flex lg:items-center lg:justify-center">
          <div className="px-8 text-center text-white">
            <p className="text-sm font-bold uppercase tracking-[0.3em] text-white/85">{t('auth.leftBadge')}</p>
            <h1 className="mt-4 text-5xl font-semibold leading-tight">{t('auth.leftTitle')}</h1>
            <p className="mt-6 text-base font-bold text-white/85">
              {t('auth.leftSubtitle')}
            </p>
          </div>
        </aside>

        <div className="flex items-center justify-center bg-slate-100 px-4 py-8 sm:px-6">
          <article className="w-full max-w-md rounded-2xl border border-slate-200 bg-white p-6 shadow-sm sm:p-8">
            <h2 className="text-2xl font-bold text-[var(--color-primary)]">{submitLabel}</h2>
            <p className="mt-2 text-sm font-semibold text-[var(--color-secondary)]">
              {mode === 'login'
                ? t('auth.loginBody')
                : t('auth.registerBody')}
            </p>

            <div className="mt-5 grid grid-cols-2 rounded-xl bg-slate-100 p-1">
              <button
                type="button"
                onClick={() => setMode('login')}
                className={`rounded-lg px-3 py-2 text-sm font-medium transition-colors ${mode === 'login' ? 'bg-white text-[var(--color-primary)] shadow-sm' : 'text-[var(--color-primary)]'
                  }`}
              >
                {t('auth.login')}
              </button>
              <button
                type="button"
                onClick={() => setMode('register')}
                className={`rounded-lg px-3 py-2 text-sm font-medium transition-colors ${mode === 'register' ? 'bg-white text-[var(--color-primary)] shadow-sm' : 'text-[var(--color-primary)]'
                  }`}
              >
                {t('auth.register')}
              </button>
            </div>

            <form className="mt-6 space-y-4" onSubmit={handleSubmit}>
              <div>
                <label htmlFor="email" className="mb-1 block text-sm font-semibold text-[var(--color-primary)]">
                  {t('auth.emailLabel')}
                </label>
                <input
                  id="email"
                  type="email"
                  required
                  value={email}
                  onChange={(event) => setEmail(event.target.value)}
                  className="w-full rounded-xl border border-slate-300 px-3 py-2 text-sm text-[var(--color-primary)] placeholder:text-[var(--color-border)] outline-none transition focus:border-[var(--color-border)] focus:ring-2 focus:ring-[var(--color-focus-ring)]"
                  placeholder={t('auth.emailPlaceholder')}
                />
              </div>

              {mode === 'register' ? (
                <div>
                  <label
                    htmlFor="name"
                    className="mb-1 block text-sm font-semibold text-[var(--color-primary)]"
                  >
                    {t('auth.nameLabel')}
                  </label>
                  <input
                    id="name"
                    type="text"
                    required
                    minLength={1}
                    value={name}
                    onChange={(event) => setName(event.target.value)}
                    className="w-full rounded-xl border border-slate-300 px-3 py-2 text-sm text-[var(--color-primary)] placeholder:text-[var(--color-border)] outline-none transition focus:border-[var(--color-border)] focus:ring-2 focus:ring-[var(--color-focus-ring)]"
                    placeholder={t('auth.namePlaceholder')}
                  />
                </div>
              ) : null}

              <div>
                <label
                  htmlFor="password"
                  className="mb-1 block text-sm font-semibold text-[var(--color-primary)]"
                >
                  {t('auth.passwordLabel')}
                </label>
                <input
                  id="password"
                  type="password"
                  required
                  minLength={8}
                  value={password}
                  onChange={(event) => setPassword(event.target.value)}
                  className="w-full rounded-xl border border-slate-300 px-3 py-2 text-sm text-[var(--color-primary)] placeholder:text-[var(--color-border)] outline-none transition focus:border-[var(--color-border)] focus:ring-2 focus:ring-[var(--color-focus-ring)]"
                  placeholder={t('auth.passwordPlaceholder')}
                />
              </div>

              {mode === 'register' ? (
                <div>
                  <label
                    htmlFor="confirmPassword"
                    className="mb-1 block text-sm font-semibold text-[var(--color-primary)]"
                  >
                    {t('auth.confirmPasswordLabel')}
                  </label>
                  <input
                    id="confirmPassword"
                    type="password"
                    required
                    minLength={8}
                    value={confirmPassword}
                    onChange={(event) => setConfirmPassword(event.target.value)}
                    className="w-full rounded-xl border border-slate-300 px-3 py-2 text-sm text-[var(--color-primary)] placeholder:text-[var(--color-border)] outline-none transition focus:border-[var(--color-border)] focus:ring-2 focus:ring-[var(--color-focus-ring)]"
                    placeholder={t('auth.passwordPlaceholder')}
                  />
                </div>
              ) : null}

              {/* GDPR Mandatory Consent Checkbox */}
              {mode === 'register' ? (
                <div className="flex items-start gap-2 py-1">
                  <input
                    id="acceptedTerms"
                    type="checkbox"
                    checked={acceptedTerms}
                    onChange={(event) => setAcceptedTerms(event.target.checked)}
                    className="mt-1 h-4 w-4 rounded border-slate-300 text-emerald-600 focus:ring-emerald-500"
                  />
                  <label htmlFor="acceptedTerms" className="text-xs font-medium text-[var(--color-secondary)] leading-normal">
                    {t('auth.privacyCheckboxPrefix')}{' '}
                    <button
                      type="button"
                      onClick={() => setShowPrivacyModal(true)}
                      className="text-emerald-600 underline hover:text-emerald-700 font-semibold inline-block transition"
                    >
                      {t('auth.privacyCheckboxLink')}
                    </button>
                  </label>
                </div>
              ) : null}

              {errorMessage ? (
                <p className="rounded-lg border border-[var(--color-active-border)] bg-[var(--color-active-bg)] px-3 py-2 text-sm font-bold text-[var(--color-secondary)]">
                  {errorMessage}
                </p>
              ) : null}

              <button
                type="submit"
                disabled={isSubmitting}
                className="w-full rounded-xl bg-gradient-to-r from-emerald-600 to-sky-600 px-4 py-2 text-sm font-semibold text-[var(--color-white)] shadow-sm transition hover:from-emerald-700 hover:to-sky-700 disabled:cursor-not-allowed disabled:opacity-60"
              >
                {isSubmitting ? t('auth.processing') : submitLabel}
              </button>
            </form>

            <div className="relative my-6">
              <div className="absolute inset-0 flex items-center">
                <div className="w-full border-t border-slate-200" />
              </div>
              <div className="relative flex justify-center text-xs uppercase">
                <span className="bg-white px-2 font-semibold text-[var(--color-secondary)]">{t('auth.or')}</span>
              </div>
            </div>

            <button
              type="button"
              onClick={handleGoogleSignIn}
              disabled={(mode === 'register' && !acceptedTerms) || isSubmitting}
              className="flex w-full items-center justify-center gap-2 rounded-xl border border-slate-300 bg-white px-4 py-2.5 text-sm font-medium text-[var(--color-secondary)] shadow-sm transition hover:bg-[var(--color-surface-soft)] disabled:cursor-not-allowed disabled:opacity-50"
            >
              {/* Google logo */}
              <svg className="h-5 w-5" viewBox="0 0 24 24" aria-hidden="true">
                <path
                  fill="currentColor"
                  d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
                />
                <path
                  fill="#34A853"
                  d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
                />
                <path
                  fill="#FBBC05"
                  d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"
                />
                <path
                  fill="#EA4335"
                  d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"
                />
              </svg>
              {t('auth.continueWithGoogle')}
            </button>

            <button
              type="button"
              onClick={() => setMode((prev) => (prev === 'login' ? 'register' : 'login'))}
              className="mt-5 text-sm font-medium text-[var(--color-secondary)] hover:text-[var(--color-secondary-strong)]"
            >
              {switchLabel}
            </button>
          </article>
        </div>
      </div>

      {/* Modal Viewport Overlay for Privacy Policy (Art. 9 GDPR) */}
      {showPrivacyModal ? (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/50 p-4 backdrop-blur-sm animate-fade-in">
          <div className="w-full max-w-lg rounded-2xl border border-slate-200 bg-white p-6 shadow-xl max-h-[80vh] flex flex-col">
            <h3 className="text-lg font-bold text-[var(--color-primary)]">
              {t('auth.privacyModalTitle')}
            </h3>

            <div className="mt-4 overflow-y-auto text-sm text-[var(--color-secondary)] pr-2 space-y-4 leading-relaxed">
              <p>{t('auth.privacyModalBody1')}</p>
              <p>{t('auth.privacyModalBody2')}</p>
              <p>{t('auth.privacyModalBody3')}</p>
            </div>

            <div className="mt-6 flex justify-end">
              <button
                type="button"
                onClick={() => setShowPrivacyModal(false)}
                className="rounded-xl bg-gradient-to-r from-emerald-600 to-teal-600 px-4 py-2 text-sm font-semibold text-white shadow-sm hover:from-emerald-700 hover:to-teal-700 transition"
              >
                {t('auth.privacyModalClose')}
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </section>
  )
}