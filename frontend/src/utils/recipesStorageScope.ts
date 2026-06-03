/**
 * Per-account localStorage keys for the Recipes page (JWT `sub` scope).
 * Does not validate the JWT signature; only reads the payload to derive stable storage scope.
 */

const LEGACY_KEYS = ['recipes.menuSession.v1', 'recipes.menuJob.v1', 'recipes.recentProducts.v1'] as const

export function decodeJwtSub(token: string | null): string | null {
  if (!token) return null
  try {
    const parts = token.split('.')
    if (parts.length !== 3) return null
    const base64 = parts[1].replace(/-/g, '+').replace(/_/g, '/')
    const padded = base64.padEnd(base64.length + (4 - (base64.length % 4)) % 4, '=')
    const json = JSON.parse(atob(padded)) as { sub?: unknown }
    return typeof json.sub === 'string' && json.sub.length > 0 ? json.sub : null
  } catch {
    return null
  }
}

export function encodeStorageScopeId(sub: string): string {
  return encodeURIComponent(sub)
}

export function getMenuSessionKey(scopeId: string): string {
  return `recipes.menuSession.v2:${scopeId}`
}

export function getMenuJobKey(scopeId: string): string {
  return `recipes.menuJob.v2:${scopeId}`
}

export function getRecentProductsKey(scopeId: string): string {
  return `recipes.recentProducts.v2:${scopeId}`
}

/** One-time migration: remove unscoped v1 keys so data does not leak across accounts. */
export function removeLegacyRecipesStorageKeys(): void {
  for (const k of LEGACY_KEYS) {
    try {
      localStorage.removeItem(k)
    } catch {
      // ignore quota / private mode
    }
  }
}
