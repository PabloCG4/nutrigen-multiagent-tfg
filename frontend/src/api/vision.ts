import { apiFetch } from './apiFetch'
import { getApiBaseUrl } from '../config/apiBaseUrl'

export interface VisionAlternative {
  name: string
  barcode: string
}

export interface VisionDetectedProduct {
  detected_name: string
  detected_barcode: string
  alternatives: VisionAlternative[]
}

interface VisionEnqueueResponse {
  status: string
  task_id: string
}

interface VisionStatusResponse {
  state: 'PENDING' | 'RECEIVED' | 'STARTED' | 'SUCCESS' | 'FAILURE'
  result?: { products?: unknown[] } | null
  error?: string | null
}

interface ProductAutocompleteSuggestion {
  name: string
  barcode: string
}

interface ProductMacrosBatchResponse {
  macros: Record<string, ProductAutocompleteSuggestion>
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

function extractTopBarcodesFromRawProduct(raw: any, max = 6): string[] {
  const out: string[] = []
  const matches = raw?.top_matches
  if (Array.isArray(matches)) {
    for (const m of matches.slice(0, max)) {
      const code = typeof m?.barcode === 'string' ? m.barcode.trim() : ''
      if (code) out.push(code)
    }
  }
  return out
}

async function fetchBarcodeNames(
  token: string,
  barcodes: string[],
): Promise<Record<string, string>> {
  if (barcodes.length === 0) return {}

  const response = await apiFetch(`${getApiBaseUrl()}/api/products/macros-by-barcodes`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify({ barcodes }),
  })
  if (!response.ok) {
    return {}
  }
  const payload: unknown = await response.json()
  if (
    typeof payload !== 'object' ||
    payload === null ||
    !('macros' in payload) ||
    typeof (payload as { macros: unknown }).macros !== 'object' ||
    (payload as { macros: unknown }).macros === null
  ) {
    return {}
  }
  const macros = (payload as ProductMacrosBatchResponse).macros
  const out: Record<string, string> = {}
  for (const [barcode, rec] of Object.entries(macros)) {
    if (rec && typeof rec.name === 'string') out[barcode] = rec.name
  }
  return out
}

function coerceProductsToDetected(
  rawProducts: unknown[],
  barcodeToName: Record<string, string>,
): VisionDetectedProduct[] {
  const out: VisionDetectedProduct[] = []
  for (const raw of rawProducts) {
    if (typeof raw !== 'object' || raw === null) continue
    const r: any = raw

    // If backend already returns the UI shape, keep it.
    if (
      typeof r.detected_name === 'string' &&
      typeof r.detected_barcode === 'string' &&
      Array.isArray(r.alternatives)
    ) {
      out.push({
        detected_name: r.detected_name,
        detected_barcode: r.detected_barcode,
        alternatives: r.alternatives
          .filter((a: any) => a && typeof a.barcode === "string")
          .map((a: any) => ({ name: String(a.name ?? ''), barcode: String(a.barcode) })),
      })
      continue
    }

    // Raw vision pipeline shape: derive from top_matches barcodes.
    const barcodes = extractTopBarcodesFromRawProduct(r, 6)
    if (barcodes.length === 0) continue
    const main = barcodes[0]!
    const detectedName = barcodeToName[main] || main
    const alternatives: VisionAlternative[] = barcodes.slice(0, 5).map((code) => ({
      barcode: code,
      name: barcodeToName[code] || code,
    }))
    while (alternatives.length < 5) alternatives.push({ barcode: '', name: '' })

    out.push({
      detected_name: detectedName,
      detected_barcode: main,
      alternatives,
    })
  }
  return out
}

export async function processVisionImageWithPolling(
  token: string,
  file: File,
  options: { pollIntervalMs?: number; timeoutMs?: number } = {},
): Promise<VisionDetectedProduct[]> {
  const { pollIntervalMs = 2000, timeoutMs = 4 * 60 * 1000 } = options
  const formData = new FormData()
  formData.append('image', file)

  const enqueueResponse = await apiFetch(`${getApiBaseUrl()}/api/vision/process`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${token}` },
    body: formData,
  })
  if (!enqueueResponse.ok) {
    const text = await enqueueResponse.text().catch(() => '')
    throw new Error(text || `Vision enqueue failed (HTTP ${enqueueResponse.status}).`)
  }

  const enqueuePayload: unknown = await enqueueResponse.json()
  if (
    typeof enqueuePayload !== 'object' ||
    enqueuePayload === null ||
    !('task_id' in enqueuePayload) ||
    typeof (enqueuePayload as { task_id: unknown }).task_id !== 'string'
  ) {
    throw new Error('Invalid vision enqueue response (missing task_id).')
  }
  const taskId = (enqueuePayload as VisionEnqueueResponse).task_id

  const startedAt = Date.now()
  while (true) {
    if (Date.now() - startedAt > timeoutMs) {
      throw new Error('Vision task timed out.')
    }

    const statusResp = await apiFetch(`${getApiBaseUrl()}/api/vision/status/${encodeURIComponent(taskId)}`, {
      method: 'GET',
      headers: { Authorization: `Bearer ${token}` },
    })
    if (!statusResp.ok) {
      const text = await statusResp.text().catch(() => '')
      throw new Error(text || `Vision status failed (HTTP ${statusResp.status}).`)
    }

    const statusPayload: unknown = await statusResp.json()
    const state = (statusPayload as VisionStatusResponse)?.state
    if (state === 'PENDING' || state === 'RECEIVED' || state === 'STARTED') {
      await sleep(pollIntervalMs)
      continue
    }
    if (state === 'FAILURE') {
      const err = (statusPayload as VisionStatusResponse)?.error
      throw new Error(err || 'Vision processing failed.')
    }
    if (state !== 'SUCCESS') {
      await sleep(pollIntervalMs)
      continue
    }

    const resultProducts = (statusPayload as VisionStatusResponse)?.result?.products
    const rawProducts = Array.isArray(resultProducts) ? resultProducts : []

    // Resolve names for raw vision pipeline output (barcode -> name).
    const barcodeSet = new Set<string>()
    for (const raw of rawProducts) {
      for (const code of extractTopBarcodesFromRawProduct(raw as any, 6)) barcodeSet.add(code)
    }
    const barcodeToName = await fetchBarcodeNames(token, Array.from(barcodeSet))
    return coerceProductsToDetected(rawProducts, barcodeToName)
  }
}

