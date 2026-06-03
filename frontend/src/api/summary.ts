import { getApiBaseUrl } from '../config/apiBaseUrl'
import type { SummaryResponse } from '../types/summary'
import { apiFetch } from './apiFetch'

export async function fetchSummary(token: string, signal?: AbortSignal): Promise<SummaryResponse> {
  const response = await apiFetch(`${getApiBaseUrl()}/api/summary`, {
    method: 'GET',
    headers: {
      Authorization: `Bearer ${token}`,
    },
    signal,
  })

  if (!response.ok) {
    const fallback = `Failed to load summary (HTTP ${response.status}).`
    let detail = fallback
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
      detail = fallback
    }
    throw new Error(detail)
  }

  return (await response.json()) as SummaryResponse
}
