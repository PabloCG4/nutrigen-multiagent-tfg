/**
 * Mirrors backend normalize_product_text (Python NFKD + strip combining + collapse spaces).
 */

export function normalizeProductText(value: string): string {
  const lowered = value.trim().toLowerCase()
  if (lowered === '') return ''
  const nfkd = lowered.normalize('NFKD')
  const withoutAccents = nfkd.replace(/\p{M}/gu, '')
  return withoutAccents.replace(/\s+/g, ' ').trim()
}

function recentMatchRank(normalizedName: string, normalizedQuery: string): number {
  if (normalizedName === normalizedQuery) return 0
  if (normalizedName.startsWith(normalizedQuery)) return 1
  return 2
}

/** Recent picks whose normalized name contains the normalized query; exact then prefix then rest, stable order. */
export function filterAndSortRecentMatches<T extends { name: string }>(
  recents: T[],
  queryRaw: string,
): T[] {
  const nq = normalizeProductText(queryRaw)
  if (nq === '') return []

  const withIndex = recents
    .map((s, i) => ({ s, i }))
    .filter(({ s }) => normalizeProductText(s.name).includes(nq))

  withIndex.sort((a, b) => {
    const na = normalizeProductText(a.s.name)
    const nb = normalizeProductText(b.s.name)
    const ra = recentMatchRank(na, nq)
    const rb = recentMatchRank(nb, nq)
    if (ra !== rb) return ra - rb
    return a.i - b.i
  })

  return withIndex.map(({ s }) => s)
}
