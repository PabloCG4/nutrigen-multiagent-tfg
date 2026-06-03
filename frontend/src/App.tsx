import { BrowserRouter, Route, Routes, useLocation } from 'react-router-dom'
import { Navbar } from './components/Navbar'
import { Footer } from './components/Footer'
import { AuthProvider } from './context/AuthContext'
import { PreferencesProvider } from './context/PreferencesContext'
import { Dashboard } from './pages/Dashboard'
import { Exercise } from './pages/Exercise'
import { History } from './pages/History'
import { Login } from './pages/Login'
import { Onboarding } from './pages/Onboarding'
import { Profile } from './pages/Profile'
import { Recipes } from './pages/Recipes'
import { Settings } from './pages/Settings'

function AppShell() {
  // useLocation reads the actual URL path.
  const location = useLocation()
  // this pages use the full screen and don't show the navbar or footer.
  const isFullBleedRoute =
    location.pathname === '/login' || location.pathname === '/onboarding'

  return (
    // flex is used to align the items in a row or column. 
    // flex-col: up the navbar, down the main content, down the footer.
    <div className="flex min-h-screen flex-col bg-[var(--color-app-bg)] text-[var(--color-app-text)]">
      {isFullBleedRoute ? null : <Navbar />}
      <main
        className={
          isFullBleedRoute
            ? ''
            // sm:px-6: padding-left and padding-right for small screens.
            // lg:px-8: padding-left and padding-right for large screens.
            : 'mx-auto w-full max-w-7xl px-4 pb-8 pt-24 sm:px-6 lg:px-8'
        }
      >
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/historial" element={<History />} />
          <Route path="/ejercicio" element={<Exercise />} />
          <Route path="/recetas" element={<Recipes />} />
          <Route path="/perfil" element={<Profile />} />
          <Route path="/settings" element={<Settings />} />
          <Route path="/login" element={<Login />} />
          <Route path="/onboarding" element={<Onboarding />} />
        </Routes>
      </main>
      {/* if the page is not full bleed, show the footer. */}
      {isFullBleedRoute ? null : <Footer />}
    </div>
  )
}

function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <PreferencesProvider>
          <AppShell />
        </PreferencesProvider>
      </AuthProvider>
    </BrowserRouter>
  )
}

// export the App component so it can be imported in main.tsx.
export default App
