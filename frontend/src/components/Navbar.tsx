import { Menu, UserCircle, X } from 'lucide-react'
import { useEffect, useState } from 'react'
import { Link, NavLink, useLocation, useNavigate } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import { useAuth } from '../context/AuthContext'
import logoUrl from '../assets/logo.png'

// Definition of the Navbar component and its properties.
export function Navbar() {
  const navigate = useNavigate()
  const location = useLocation()
  const { t } = useTranslation()
  // isAuthenticated is a boolean that indicates if the user is authenticated, logout is its setter
  const { isAuthenticated, logout } = useAuth()
  const [showAuthModal, setShowAuthModal] = useState<boolean>(false)
  const [isProfileDropdownOpen, setIsProfileDropdownOpen] = useState<boolean>(false)
  const [mobileMenuOpen, setMobileMenuOpen] = useState<boolean>(false)

  // navItems is an array of objects that contains the labels and the paths of the navigation links.
  const navItems: Array<{ label: string; to: string }> = [
    { label: t('nav.home'), to: '/' },
    { label: t('nav.history'), to: '/historial' },
    { label: t('nav.exercise'), to: '/ejercicio' },
    { label: t('nav.recipes'), to: '/recetas' },
  ]


  const handleProtectedClick = (path: string): void => {
    setMobileMenuOpen(false)
    // if the path is the home page or the user is authenticated, navigate to the path.
    if (path === '/' || isAuthenticated) {
      navigate(path)
      return
    }
    // if the path is not the home page and the user is not authenticated, show the authentication modal.
    setShowAuthModal(true)
  }

  // useEffect is used to close the profile dropdown when the user clicks outside of it.
  useEffect(() => {
    if (!isProfileDropdownOpen) {
      return
    }

    // when the user clicks outside of the profile dropdown, it closes.
    const onDocumentClick = (): void => {
      setIsProfileDropdownOpen(false)
    }

    document.addEventListener('click', onDocumentClick)
    return () => document.removeEventListener('click', onDocumentClick)
  }, [isProfileDropdownOpen])

  useEffect(() => {
    if (!mobileMenuOpen) {
      return
    }
    // used to prevent the body from scrolling when the mobile menu (navbar version for mobile devices) is open.
    const previousOverflow = document.body.style.overflow
    document.body.style.overflow = 'hidden' // hidden = no scroll.
    return () => {
      document.body.style.overflow = previousOverflow
    }
  }, [mobileMenuOpen])

  // navLinkClass is used to style the navigation links or buttons
  const navLinkClass = (isActive: boolean): string =>
    `block w-full rounded-xl px-4 py-3 text-left text-sm font-semibold uppercase tracking-wide transition-colors md:inline-block md:w-auto md:rounded-full md:px-4 md:py-2 ${isActive
      ? 'bg-gradient-to-r from-emerald-600 to-sky-600 text-white'
      : 'text-[var(--color-primary)] hover:bg-[var(--color-surface-soft)] hover:text-[var(--color-primary)]'
    }`

  const navButtonClass = (isActive: boolean): string =>
    navLinkClass(isActive)

  return (
    <>
      <header className="fixed inset-x-0 top-0 z-50 border-b border-[var(--color-border)] bg-[var(--color-surface)]/95 backdrop-blur">
        <nav
          className="relative mx-auto flex h-16 w-full max-w-7xl items-center gap-2 px-4 sm:h-18 sm:gap-3 sm:px-6 lg:px-8"
          aria-label={t('nav.ariaMainNavigation', 'Main navigation')}
        >
          {/* logo button: when the user clicks on the logo, it navigates to the home page. */}
          <button
            type="button"
            onClick={() => handleProtectedClick('/')}
            className="inline-flex shrink-0 items-center justify-center transition-opacity hover:opacity-80"
          >
            <img
              src={logoUrl}
              alt={t('brand.name')}
              // h-10 w-auto object-contain: when the screen is mobile, the logo is 10px high and the width is auto.
              className="h-12 w-auto max-w-[140px] object-contain sm:h-14 sm:max-w-[300px]"
            />
          </button>


          <ul className="ml-1 hidden min-w-0 flex-1 items-center justify-center gap-2 md:flex">
            {/* navItems.map is used to iterate over the navItems array and create a list of navigation links. */}
            {navItems.map((item) => (
              <li key={item.to} className="shrink-0">
                {/* if the path is the home page, show the navigation link on the browser, on the navbar */}
                {item.to === '/' ? (
                  // NavLink is used to navigate to the path and style the link.
                  <NavLink
                    to={item.to}
                    className={({ isActive }) => navLinkClass(isActive)}
                  >
                    {item.label}
                  </NavLink>
                  // if the path is not the home page, show the navigation link on the browser, on the navbar.
                ) : (
                  <button
                    type="button"
                    onClick={() => handleProtectedClick(item.to)}
                    // navButtonClass is used to style the navigation links or buttons
                    className={navButtonClass(location.pathname === item.to)}
                  >
                    {/* name of the button */}
                    {item.label}
                  </button>
                )}
              </li>
            ))}
          </ul>

          {/* profile dropdown: when the user clicks on the profile dropdown, it shows the profile menu. */}
          <div className="ml-auto flex shrink-0 items-center gap-2">
            {isAuthenticated ? (
              <div
                // relative: the profile dropdown is relative to the parent element, which is the div.
                // shrink-0: the profile dropdown is not allowed to shrink.
                className="relative shrink-0"
                onClick={(event) => {
                  // clicking inside the menu does not close it.
                  event.stopPropagation()
                }}
              >
                {/* profile dropdown button: when the user clicks on the profile dropdown button, it shows the profile menu. */}
                <button
                  type="button"
                  onClick={() => setIsProfileDropdownOpen((previous) => !previous)}
                  className="inline-flex shrink-0 items-center justify-center rounded-full font-semibold text-[var(--color-primary)] border border-[var(--color-primary)] px-3 py-2 text-xs font-medium uppercase tracking-wide transition-colors hover:bg-gradient-to-r hover:from-emerald-600 hover:to-sky-600 hover:text-white sm:px-4 sm:text-sm"
                  aria-label={t('nav.openProfileMenu')}
                >
                  {/* icon of the profile dropdown */}
                  <UserCircle className="h-6 w-6 sm:h-7 sm:w-7" />
                </button>

                {isProfileDropdownOpen ? (
                  // shadow-lg: a shadow effect like a 3D effect.
                  <div className="absolute right-0 mt-1 w-40 rounded-2xl border border-[var(--color-border)] bg-white p-2 shadow-lg">
                    <button
                      type="button"
                      onClick={() => {
                        navigate('/perfil')
                        setIsProfileDropdownOpen(false)
                      }}
                      className="w-full rounded-xl px-3 py-2 text-right text-sm font-semibold text-[var(--color-primary)] transition hover:bg-[var(--color-surface-soft)]"
                    >
                      {t('nav.profile')}
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        navigate('/settings')
                        setIsProfileDropdownOpen(false)
                      }}
                      className="mt-1 w-full rounded-xl px-3 py-2 text-right text-sm font-semibold text-[var(--color-primary)] transition hover:bg-[var(--color-surface-soft)]"
                    >
                      {t('nav.settings')}
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        setIsProfileDropdownOpen(false)
                        logout()
                      }}
                      className="mt-1 w-full rounded-xl px-3 py-2 text-right text-sm font-bold text-[var(--color-app-text)] transition hover:bg-[var(--color-surface-soft)]"
                    >
                      {t('nav.signOut')}
                    </button>
                  </div>
                ) : null}
              </div>
            ) : (
              <Link
                to="/login"
                onClick={() => setMobileMenuOpen(false)}
                className="inline-flex shrink-0 items-center gap-1 rounded-full border-2 border-[var(--color-primary)] px-3 py-2 text-xs font-medium uppercase tracking-wide text-[var(--color-primary)] transition-colors hover:bg-gradient-to-r hover:from-emerald-600 hover:to-sky-600 hover:text-white sm:gap-2 sm:px-4 sm:text-sm"
                aria-label={t('nav.goToLogin')}
              >
                <UserCircle className="h-4 w-4 shrink-0 sm:h-5 sm:w-5" />
                <span className="inline-block">{t('nav.signIn')}</span>
              </Link>
            )}

            <button
              type="button"
              className="inline-flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border border-[var(--color-primary)] text-[var(--color-app-text)] transition-colors hover:bg-[var(--color-surface-soft)] md:hidden"
              onClick={() => setMobileMenuOpen((open) => !open)}
              aria-expanded={mobileMenuOpen}
              aria-controls="mobile-nav-panel"
              aria-label={mobileMenuOpen ? t('common.closeMenu') : t('common.openMenu')}
            >
              {mobileMenuOpen ? <X className="h-5 w-5 text-[var(--color-primary)] font-semibold" /> : <Menu className="h-5 w-5 text-[var(--color-primary)] font-semibold" />}
            </button>
          </div>

          {mobileMenuOpen ? (
            <>
              <button
                type="button"
                className="fixed inset-0 top-16 z-40 bg-slate-950/50 md:hidden"
                aria-label={t('common.closeMenu')}
                onClick={() => setMobileMenuOpen(false)}
              />
              <div
                id="mobile-nav-panel"
                className="absolute left-0 right-0 top-full z-50 max-h-[min(70vh,calc(100dvh-4rem))] overflow-y-auto border-b border-[var(--color-border)] bg-[var(--color-surface)] shadow-lg md:hidden"
              >
                <ul className="flex flex-col gap-1 p-4">
                  {navItems.map((item) => (
                    <li key={item.to}>
                      {item.to === '/' ? (
                        <NavLink
                          to={item.to}
                          onClick={() => setMobileMenuOpen(false)}
                          className={({ isActive }) => navLinkClass(isActive)}
                        >
                          {item.label}
                        </NavLink>
                      ) : (
                        <button
                          type="button"
                          onClick={() => handleProtectedClick(item.to)}
                          className={navButtonClass(location.pathname === item.to)}
                        >
                          {item.label}
                        </button>
                      )}
                    </li>
                  ))}
                </ul>
              </div>
            </>
          ) : null}
        </nav>
      </header>

      {showAuthModal ? (
        // bg-slate-950/60: a dark overlay to make the modal more visible making the background darker.
        <div className="fixed inset-0 z-[60] flex items-center justify-center bg-slate-950/60 p-4">
          <div className="w-full max-w-md rounded-2xl bg-white p-6 shadow-xl">
            <h2 className="text-lg font-bold text-[var(--color-primary)]">{t('nav.accessRequiredTitle')}</h2>
            <p className="mt-2 text-sm font-semibold text-[var(--color-secondary)]">{t('nav.accessRequiredBody')}</p>
            <div className="mt-5 flex justify-end gap-2">
              <button
                type="button"
                onClick={() => setShowAuthModal(false)}
                className="rounded-lg border border-[var(--color-border)] px-4 py-2 text-sm text-[var(--color-secondary)] hover:bg-[var(--color-surface-soft)]"
              >
                {t('common.cancel')}
              </button>
              <Link
                to="/login"
                onClick={() => setShowAuthModal(false)}
                className="rounded-lg bg-gradient-to-r from-emerald-600 to-sky-600 px-4 py-2 text-sm font-semibold text-white shadow-sm transition hover:from-emerald-700 hover:to-sky-700"
              >
                {t('nav.goToLogin')}
              </Link>
            </div>
          </div>
        </div>
      ) : null}
    </>
  )
}
