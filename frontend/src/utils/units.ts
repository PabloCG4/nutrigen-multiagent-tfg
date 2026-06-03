export type UnitsValue = 'metric' | 'imperial'

export interface FormattedValue {
  value: number
  unitLabel: string
}

const KG_TO_LB = 2.2046226218
const CM_TO_IN = 0.3937007874

export function kgToLb(valueKg: number): number {
  return valueKg * KG_TO_LB
}

export function lbToKg(valueLb: number): number {
  return valueLb / KG_TO_LB
}

export function cmToIn(valueCm: number): number {
  return valueCm * CM_TO_IN
}

export function inToCm(valueIn: number): number {
  return valueIn / CM_TO_IN
}

export function formatWeight(valueKg: number, units: UnitsValue): FormattedValue {
  if (units === 'imperial') {
    return { value: kgToLb(valueKg), unitLabel: 'lbs' }
  }
  return { value: valueKg, unitLabel: 'kg' }
}

export function formatHeight(valueCm: number, units: UnitsValue): FormattedValue {
  if (units === 'imperial') {
    return { value: cmToIn(valueCm), unitLabel: 'in' }
  }
  return { value: valueCm, unitLabel: 'cm' }
}

export function convertWeightInputToKg(value: number, units: UnitsValue): number {
  return units === 'imperial' ? lbToKg(value) : value
}

export function convertHeightInputToCm(value: number, units: UnitsValue): number {
  return units === 'imperial' ? inToCm(value) : value
}

