/**
 * Resolves the backend API origin for HTTP requests and browser redirects (e.g. OAuth).
 * Production builds require VITE_API_URL to be set when running `vite build`.
 */

function trimTrailingSlashes(value: string): string {
  return value.replace(/\/+$/, '')
}

export function getApiBaseUrl(): string {
  const raw = import.meta.env.VITE_API_URL
  if (typeof raw === 'string') {
    const trimmed = raw.trim()
    if (trimmed.length > 0) {
      return trimTrailingSlashes(trimmed)
    }
  }
  if (import.meta.env.DEV) {
    const developmentFallbackUrl = ['http://', 'localhost', ':8000'].join('')
    return developmentFallbackUrl
  }
  throw new Error(
    'VITE_API_URL is not set. Configure the backend base URL for this deployment.',
  )
}
