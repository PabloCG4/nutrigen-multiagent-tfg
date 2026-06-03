import { useState, useEffect, useRef, type ReactNode } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import { apiFetch } from '../api/apiFetch'
import { getApiBaseUrl } from '../config/apiBaseUrl'
import { decodeJwtSub, encodeStorageScopeId, getMenuJobKey } from '../utils/recipesStorageScope'

export function useMenuJobNotification() {
  const navigate = useNavigate()
  const location = useLocation()
  const { token, isAuthenticated } = useAuth()
  const [showToast, setShowToast] = useState<boolean>(false)
  const pollIntervalRef = useRef<number | null>(null)

  useEffect(() => {
    // Disable background polling if the user is currently on the recipes page
    if (!isAuthenticated || !token || location.pathname === '/recetas') {
      if (pollIntervalRef.current !== null) {
        window.clearInterval(pollIntervalRef.current)
        pollIntervalRef.current = null
      }
      return
    }

    const sub = decodeJwtSub(token)
    if (!sub) return
    const storageScopeId = encodeStorageScopeId(sub)
    const menuJobKey = getMenuJobKey(storageScopeId)

    const checkJobStatus = async (): Promise<void> => {
      const rawJob = localStorage.getItem(menuJobKey)
      if (!rawJob) return

      try {
        const payload = JSON.parse(rawJob)
        if (!payload || !payload.jobId) return

        // Track notification state using sessionStorage to avoid mutating core recipe keys
        const notificationKey = `menu-notification-${payload.jobId}`
        if (sessionStorage.getItem(notificationKey)) {
          if (pollIntervalRef.current !== null) {
            window.clearInterval(pollIntervalRef.current)
            pollIntervalRef.current = null
          }
          return
        }

        if (payload.state === 'queued' || payload.state === 'running') {
          const response = await apiFetch(`${getApiBaseUrl()}/api/menu-jobs/${encodeURIComponent(payload.jobId)}`, {
            method: 'GET',
            headers: { Authorization: `Bearer ${token}` },
          })

          if (response.ok) {
            const result = await response.json()
            const serverState = result.data?.state

            if (serverState === 'done') {
              sessionStorage.setItem(notificationKey, 'true')
              setShowToast(true)
              if (pollIntervalRef.current !== null) {
                window.clearInterval(pollIntervalRef.current)
                pollIntervalRef.current = null
              }
            } else if (serverState === 'error') {
              sessionStorage.setItem(notificationKey, 'true')
              if (pollIntervalRef.current !== null) {
                window.clearInterval(pollIntervalRef.current)
                pollIntervalRef.current = null
              }
            }
          }
        }
      } catch {
        // Safe protection context
      }
    }

    void checkJobStatus()
    pollIntervalRef.current = window.setInterval(checkJobStatus, 3000)

    return () => {
      if (pollIntervalRef.current !== null) {
        window.clearInterval(pollIntervalRef.current)
        pollIntervalRef.current = null
      }
    }
  }, [isAuthenticated, token, location.pathname])

  const renderMenuNotificationToast = (): ReactNode => {
    if (!showToast) return null

    return (
      <div className="fixed bottom-4 right-4 z-50 w-full max-w-sm rounded-2xl border border-slate-200 bg-white p-4 shadow-xl">
        <div className="flex flex-col gap-3">
          <p className="text-sm font-semibold text-[var(--color-primary)] leading-normal">
            Tu menú está listo. Haz clic aquí para acceder a él.
          </p>
          <div className="flex justify-end gap-2">
            <button
              type="button"
              onClick={() => setShowToast(false)}
              className="rounded-xl border border-slate-200 px-3 py-1.5 text-xs font-semibold text-[var(--color-secondary)] transition hover:bg-[var(--color-surface-soft)]"
            >
              Cancelar
            </button>
            <button
              type="button"
              onClick={() => {
                setShowToast(false)
                // Send navigation state to trigger automated scrolling on target layout
                navigate('/recetas', { state: { scrollToBottom: true } })
              }}
              className="rounded-xl bg-gradient-to-r from-emerald-600 to-teal-600 px-4 py-1.5 text-xs font-semibold text-white shadow-sm hover:from-emerald-700 hover:to-teal-700 transition"
            >
              Ver menú
            </button>
          </div>
        </div>
      </div>
    )
  }

  return { renderMenuNotificationToast }
}