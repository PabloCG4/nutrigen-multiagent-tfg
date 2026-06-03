/* eslint-disable react-refresh/only-export-components */
import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react'
import type { ReactNode } from 'react'
import { useTranslation } from 'react-i18next'
import i18next from 'i18next'
import { apiFetch } from '../api/apiFetch'
import { getApiBaseUrl } from '../config/apiBaseUrl'
import { useAuth } from './AuthContext'

export type LanguageValue = 'es' | 'en'
export type UnitsValue = 'metric' | 'imperial'

interface PreferencesState {
  language: LanguageValue
  units: UnitsValue
  // userOverrideLanguage is a boolean that indicates if the user has overridden the language. If its false, the language is the one of the browser.
  userOverrideLanguage: boolean
}

// We define all that the app can see
interface PreferencesContextValue extends PreferencesState {
  setLanguage: (value: LanguageValue) => void
  setUnits: (value: UnitsValue) => void
  refreshFromBackend: () => Promise<void>
  syncToBackend: () => Promise<void>
}

const PREFERENCES_STORAGE_KEY = 'user_preferences_v1'

// check if the value given by the local storage (hard disk of the browser) is a valid PreferencesState.
function isPreferencesState(value: unknown): value is PreferencesState {
  if (typeof value !== 'object' || value === null) return false
  const v = value as { language?: unknown; units?: unknown; userOverrideLanguage?: unknown }
  const langOk = v.language === 'es' || v.language === 'en'
  const unitsOk = v.units === 'metric' || v.units === 'imperial'
  const overrideOk = typeof v.userOverrideLanguage === 'boolean'
  return langOk && unitsOk && overrideOk
}

function loadStoredPreferences(): PreferencesState | null {
  try {
    // tries to read the "directory" of the local storage (hard disk of the browser) that contains the preferences.
    const raw = localStorage.getItem(PREFERENCES_STORAGE_KEY)
    if (!raw) return null
    const parsed: unknown = JSON.parse(raw)
    // check if the value is a valid PreferencesState.
    return isPreferencesState(parsed) ? parsed : null
  } catch {
    return null
  }
}

// tries to read an older version of the preferences if the new one is not found.
function loadStoredPreferencesCompat(): PreferencesState | null {
  const stored = loadStoredPreferences()
  if (stored) return stored
  try {
    const raw = localStorage.getItem(PREFERENCES_STORAGE_KEY)
    if (!raw) return null
    const parsed: unknown = JSON.parse(raw)
    if (typeof parsed !== 'object' || parsed === null) return null
    const v = parsed as { language?: unknown; units?: unknown }
    const langOk = v.language === 'es' || v.language === 'en'
    const unitsOk = v.units === 'metric' || v.units === 'imperial'
    if (!langOk || !unitsOk) return null
    return {
      language: v.language as LanguageValue,
      units: v.units as UnitsValue,
      userOverrideLanguage: false,
    }
  } catch {
    return null
  }
}

function storePreferences(value: PreferencesState): void {
  // setItem is a function from the package localStorage that stores the PreferencesState in the local storage (hard disk of the browser).
  localStorage.setItem(PREFERENCES_STORAGE_KEY, JSON.stringify(value))
}

const PreferencesContext = createContext<PreferencesContextValue | undefined>(undefined)

// Defines that this component is a wrapper that provides the PreferencesContext to the children.
interface PreferencesProviderProps {
  children: ReactNode
}

export function PreferencesProvider({ children }: PreferencesProviderProps) {
  // obtained from the AuthContext.
  const { token, isAuthenticated, logout } = useAuth()
  const { i18n } = useTranslation()

  // cache to avoid re-rendering the component when the preferences are not changed.
  const stored = useMemo(() => loadStoredPreferencesCompat(), [])
  // if theres language in the stored preferences, use it. Otherwise, use the language of the browser.
  const [language, setLanguageState] = useState<LanguageValue>(stored?.language ?? (i18n.language === 'en' ? 'en' : 'es'))
  // if theres units in the stored preferences, use it. Otherwise, use "metric" as default.
  const [units, setUnitsState] = useState<UnitsValue>(stored?.units ?? 'metric')
  const [userOverrideLanguage, setUserOverrideLanguage] = useState<boolean>(stored?.userOverrideLanguage ?? false)

  // if the variable language is changed, change the language of the app.
  useEffect(() => {
    void i18n.changeLanguage(language)
  }, [i18n, language])

  // if the variable units or the language is changed, change the units or the language of the app.
  useEffect(() => {
    storePreferences({ language, units, userOverrideLanguage })
  }, [language, units, userOverrideLanguage])

  const refreshFromBackend = useCallback(async (): Promise<void> => {
    if (!token || !isAuthenticated) return // you need to be authenticated to get the preferences from the backend.
    const response = await apiFetch(`${getApiBaseUrl()}/api/profile`, { // get the profile from the backend.
      method: 'GET',
      headers: { Authorization: `Bearer ${token}` }, // the token is obtained from the AuthContext.
    })
    if (!response.ok) {
      // if the token is invalid or the profile is not found, logout the user.
      const payload: unknown = await response.json().catch(() => null)
      const detail =
        typeof payload === 'object' && payload !== null && 'detail' in payload
          ? String((payload as { detail?: unknown }).detail ?? '')
          : ''
      if (detail.toLowerCase().includes('could not validate credentials')) {
        logout()
      }
      return
    }
    const data: unknown = await response.json()
    if (typeof data !== 'object' || data === null) return // if the data is not a valid PreferencesState, return.
    const p = data as { language?: unknown; units?: unknown } // cast the data to a PreferencesState.
    if (!userOverrideLanguage && (p.language === 'es' || p.language === 'en')) {
      setLanguageState(p.language)
    }
    if (p.units === 'metric' || p.units === 'imperial') {
      setUnitsState(p.units)
    }
  }, [token, isAuthenticated, logout, userOverrideLanguage])

  const syncToBackend = useCallback(async (): Promise<void> => {
    if (!token || !isAuthenticated) return
    await apiFetch(`${getApiBaseUrl()}/api/settings/preferences`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({ language, units }),
    })
  }, [token, isAuthenticated, language, units])

  // Decides when to refresh the preferences from the backend. When is it? When the user is authenticated and has not overridden the language.
  useEffect(() => {
    if (!token || !isAuthenticated) {
      return // if the user is not authenticated, return.
    }
    if (userOverrideLanguage) {
      return // if the user has overridden the language, return.
    }
    // void used to avoid the warning about the promise not being awaited, whe dont need to wait for the promise to be resolved here because we are not using the result.
    void refreshFromBackend()
  }, [token, isAuthenticated, userOverrideLanguage, refreshFromBackend])

  const setLanguage = useCallback((value: LanguageValue): void => {
    setLanguageState(value)
    // set the userOverrideLanguage to true to indicate that the user has overridden the language, avoiding to refresh the preferences from the backend.
    setUserOverrideLanguage(true)
  }, [])

  const setUnits = useCallback((value: UnitsValue): void => {
    setUnitsState(value)
  }, [])

  const value = useMemo<PreferencesContextValue>(
    () => ({
      language,
      units,
      userOverrideLanguage,
      setLanguage,
      setUnits,
      refreshFromBackend,
      syncToBackend,
    }),
    [language, units, userOverrideLanguage, setLanguage, setUnits, refreshFromBackend, syncToBackend],
  )

  return <PreferencesContext.Provider value={value}>{children}</PreferencesContext.Provider>
}

export function usePreferences(): PreferencesContextValue {
  const context = useContext(PreferencesContext)
  if (!context) {
    throw new Error(i18next.t('errors.usePreferencesProvider'))
  }
  return context
}

