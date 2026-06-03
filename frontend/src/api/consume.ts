import { getApiBaseUrl } from '../config/apiBaseUrl'
import { apiFetch } from './apiFetch'

export async function removeConsumedRecipe(
  token: string,
  payload: { date: string; index: number },
  signal?: AbortSignal,
): Promise<void> {
  const response = await apiFetch(`${getApiBaseUrl()}/api/consume/remove`, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${token}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
    signal,
  })

  if (!response.ok) {
    const fallback = `Failed to remove recipe (HTTP ${response.status}).`
    let detail = fallback
    try {
      const body: unknown = await response.json()
      if (
        typeof body === 'object' &&
        body !== null &&
        'detail' in body &&
        typeof (body as { detail: unknown }).detail === 'string'
      ) {
        detail = (body as { detail: string }).detail
      }
    } catch {
      detail = fallback
    }
    throw new Error(detail)
  }
}
