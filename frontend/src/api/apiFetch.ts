import { getApiBaseUrl } from '../config/apiBaseUrl'

const NGROK_SKIP_BROWSER_WARNING_HEADER_NAME = 'ngrok-skip-browser-warning'
const NGROK_SKIP_BROWSER_WARNING_HEADER_VALUE = 'true'

export async function apiFetch(
  pathOrUrl: string,
  options: RequestInit = {},
): Promise<Response> {
  const mergedHeaders = new Headers(options.headers)
  mergedHeaders.set(
    NGROK_SKIP_BROWSER_WARNING_HEADER_NAME,
    NGROK_SKIP_BROWSER_WARNING_HEADER_VALUE,
  )

  const isAbsoluteUrl =
    pathOrUrl.startsWith('http://') || pathOrUrl.startsWith('https://')
  const url = isAbsoluteUrl ? pathOrUrl : `${getApiBaseUrl()}${pathOrUrl}`

  return fetch(url, {
    ...options,
    headers: mergedHeaders,
  })
}

