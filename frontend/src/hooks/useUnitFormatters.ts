import { useMemo } from 'react'
import { useTranslation } from 'react-i18next'
import { usePreferences } from '../context/PreferencesContext'
import type { FormattedValue } from '../utils/units'
import { formatHeight, formatWeight } from '../utils/units'

interface UnitFormatters {
  units: 'metric' | 'imperial'
  formatWeightLocalized: (valueKg: number) => FormattedValue
  formatHeightLocalized: (valueCm: number) => FormattedValue
  unitLabel: (unitKey: 'kg' | 'lbs' | 'cm' | 'in' | 'kcal' | 'g') => string
}

export function useUnitFormatters(): UnitFormatters {
  const preferences = usePreferences()
  const { t } = useTranslation()

  // useMemo is a cache that stores the result of the function and returns it if the dependencies have not changed.
  return useMemo<UnitFormatters>(() => {
    return {
      units: preferences.units,
      // formatWeightLocalized is a function that formats the weight in the units of the user.
      formatWeightLocalized: (valueKg: number) => {
        const formatted = formatWeight(valueKg, preferences.units)
        return { ...formatted, unitLabel: t(`units.${formatted.unitLabel}`) }
      },
      formatHeightLocalized: (valueCm: number) => {
        const formatted = formatHeight(valueCm, preferences.units)
        return { ...formatted, unitLabel: t(`units.${formatted.unitLabel}`) }
      },
      // unitLabel is a function that returns the label of the unit.
      unitLabel: (unitKey) => t(`units.${unitKey}`),
    }
  }, [preferences.units, t]) // dependencies of the hook are the units and the translation.
}

