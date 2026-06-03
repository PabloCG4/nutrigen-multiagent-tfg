import { useTranslation } from 'react-i18next'

export function Footer() {
  const { t } = useTranslation()

  return (
    <footer className="mt-auto border-t border-[var(--color-border)] bg-gradient-to-r from-emerald-600 to-sky-600 text-white">
      <div className="mx-auto w-full max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
        <div className="flex flex-col gap-6 sm:flex-row sm:items-start sm:justify-between">
          <div>
            <div className="text-lg font-extrabold tracking-wide">{t('brand.name')}</div>
            <p className="mt-2 max-w-md text-sm text-white/90">
              {t('footer.description')}
            </p>
          </div>

          <div className="text-sm">
            <div className="font-semibold">{t('footer.contact')}</div>
            <a
              className="mt-2 inline-block text-white/90 underline decoration-white/40 underline-offset-4 hover:text-white"
              href="mailto:chicagonzalezpablo@gmail.com"
            >
              chicagonzalezpablo@gmail.com
            </a>
            <div className="mt-3 text-xs text-white/80">
              © {new Date().getFullYear()} {t('brand.name')}
            </div>
          </div>
        </div>
      </div>
    </footer>
  )
}

