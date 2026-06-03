import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react'
import type { ReactNode } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import i18next from 'i18next'

import { apiFetch } from '../api/apiFetch'
const AUTH_TOKEN_KEY = 'auth_token'

// AuthContextValue is the type of the context value. It keeps functions. Promise is a type that represents a value that may be available now or in the future, and is void because the functions do not return a value.
interface AuthContextValue {
  isAuthenticated: boolean
  login: (email: string, password: string) => Promise<void>
  register: (email: string, password: string, name: string) => Promise<void>
  logout: () => void
  token: string | null
}

// createContext is a function imported from react made to create a context object. It is used to pass the context value to the components.
const AuthContext = createContext<AuthContextValue | undefined>(undefined)

interface AuthProviderProps {
  children: ReactNode
}

// catch the error from FastAPI and return the error message.
async function parseError(response: Response, fallback: string): Promise<string> {
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


export function AuthProvider({ children }: AuthProviderProps) {
  const navigate = useNavigate()
  const location = useLocation()
  // only reads the token from the local storage once.
  const [token, setToken] = useState<string | null>(() => localStorage.getItem(AUTH_TOKEN_KEY))

  // OAuth redirect: backend sends users to /login?token=...&new_user=1. If that parameters are present, UseEffect cleans the url and redirects the user to the home page or the onboarding page.
  useEffect(() => {
    const isOAuthCallbackRoute =
      location.pathname === '/login' || location.pathname.endsWith('/login')

    if (!isOAuthCallbackRoute) {
      return
    }
    // URLSearchParams is a class that parses the query string from the URL and returns a map of key-value pairs.
    const params = new URLSearchParams(location.search)
    const tokenParam = params.get('token')
    if (!tokenParam) {
      return
    }
    localStorage.setItem(AUTH_TOKEN_KEY, tokenParam)
    // Update token state after parsing the OAuth redirect query string.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setToken(tokenParam)

    // Backend marks new accounts with `new_user=1` in the redirect query string. How does it know if the user is new? By the email address put in Google OAuth.
    // This logic also tolerates `new_user=true` for safety.
    const newUserParam = params.get('new_user')
    const isNewUser =
      newUserParam === '1' || newUserParam?.toLowerCase() === 'true'

    navigate(isNewUser ? '/onboarding' : '/', { replace: true })
  }, [location.pathname, location.search, navigate])

  // async because the response is not immediate, and the web cannot wait for it because it would block the main thread.
  const login = useCallback(async (email: string, password: string): Promise<void> => {
    const body = new URLSearchParams({
      username: email,
      password,
    })
    const response = await apiFetch('/api/login', {
      method: 'POST',
      // parameters are sent in html form urlencoded format, demanded like that by Oauth2 security standards
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: body.toString(),
    })

    if (!response.ok) {
      throw new Error(await parseError(response, 'Invalid credentials.'))
    }

    const payload: unknown = await response.json()
    if (
      typeof payload !== 'object' ||
      payload === null ||
      !('access_token' in payload) ||
      typeof (payload as { access_token: unknown }).access_token !== 'string'
    ) {
      throw new Error(i18next.t('errors.invalidLoginResponse'))
    }

    const accessToken = (payload as { access_token: string }).access_token
    // store the token in the local storage to keep the user logged in even if the page is refreshed.
    localStorage.setItem(AUTH_TOKEN_KEY, accessToken)
    // update the token state to keep the user logged in even if the page is refreshed.
    setToken(accessToken)
  }, [])

  const register = useCallback(
    async (email: string, password: string, name: string): Promise<void> => {
      const registerResponse = await apiFetch('/api/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        // stringify is a function that converts the object to a JSON string.
        body: JSON.stringify({ email, password, name }),
      })

      if (!registerResponse.ok) {
        throw new Error(await parseError(registerResponse, 'Registration failed.'))
      }

      // to not make the user wait for the login to finish, we call the login function inside the register function.
      await login(email, password)
      navigate('/onboarding')
    },
    [login, navigate],
  )

  const logout = useCallback((): void => {
    localStorage.removeItem(AUTH_TOKEN_KEY)
    setToken(null)
    // redirect the user to the home page.
    navigate('/')
  }, [navigate])

  // useMemo is a cache that stores the result of the function and returns it if the dependencies have not changed.
  const value = useMemo<AuthContextValue>(
    () => ({ // () => is a function that returns an object.
      isAuthenticated: Boolean(token),
      login,
      register,
      logout,
      token,
    }),
    [login, logout, register, token],
  )

  // children is the components of the app (navbar, footer, etc.)
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

// Fast refresh only supports component exports in this file; `useAuth` is a hook.
// eslint-disable-next-line react-refresh/only-export-components
export function useAuth(): AuthContextValue {
  const context = useContext(AuthContext)
  if (!context) {
    throw new Error(i18next.t('errors.useAuthProvider'))
  }
  return context
}

