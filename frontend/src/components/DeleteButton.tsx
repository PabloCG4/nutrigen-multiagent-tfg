import { Trash2 } from 'lucide-react'
import type { ButtonHTMLAttributes, ReactNode } from 'react'

type DeleteButtonProps = {
  children: ReactNode
  /** Shown when `aria-label` is not passed */
  'aria-label'?: string
} & Omit<ButtonHTMLAttributes<HTMLButtonElement>, 'children'>

const baseClass =
  'inline-flex items-center justify-center gap-2 rounded-lg bg-gradient-to-r from-red-600 to-red-700 px-3 py-2 text-sm font-semibold text-white shadow-sm transition hover:from-red-700 hover:to-red-800 disabled:cursor-not-allowed disabled:opacity-60'

/**
 * Destructive action button: red gradient, trash icon, white label (used for delete/remove flows).
 */
export function DeleteButton({ children, className = '', disabled, type = 'button', ...rest }: DeleteButtonProps) {
  return (
    <button type={type} disabled={disabled} className={`${baseClass} ${className}`.trim()} {...rest}>
      <Trash2 className="h-4 w-4 shrink-0" aria-hidden />
      {children}
    </button>
  )
}
